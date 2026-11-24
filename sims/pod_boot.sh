#!/usr/bin/env bash
# Unattended pod job: set up, record, wait a grace period, stop the pod. Started from
# the pod's start command (see the queue/create call), which clones this repo into
# /workspace first and then runs:
#
#   nohup bash /workspace/free-energy-resource-allocation/sims/pod_boot.sh > /workspace/boot.log 2>&1 &
#
# Pod env vars (all optional):
#   FERA_BRANCH   repo branch to stay on           (default: main)
#   FERA_CONFIG   config for prep/run               (default: configs/mdm2.yaml)
#   FERA_RUNS     run.py argument sets separated by "|", run in order after prep.
#                 Default = the MDM2 pilot: 4 bound-vanish + 2 free-vanish tapes of rep 1.
#   FERA_GRACE_MIN  minutes to stay up after finishing, so tapes can be pulled (default 60)
#   FERA_WAIT_PREP=1  don't run prep here: wait until a finished prep tree has been
#                 pushed in (sims/pods.sh copy-prep writes its manifest.json last-ish;
#                 we wait for the marker file PREP_READY), so every replicate uses the
#                 same equilibrations and restraint.
#   FERA_SMOKE=1  before the real prep, run the whole pipeline on a throwaway copy of the
#                 config (20 ps equilibrations, 2 replicates, one 0.02 ns tape per leg)
#                 so a crash in a late prep step shows up in minutes, not hours (default 1;
#                 skipped on pods that wait for a pushed prep tree).
#   FERA_FAIL_GRACE_MIN  if any step fails, stay up this long so the logs can be read
#                 before stopping (default 240).
#   The pod stops itself with `runpodctl stop` using the pod-scoped API key RunPod puts
#   in every pod (RUNPOD_API_KEY); it is NOT left running when the work is done.
REPO=/workspace/free-energy-resource-allocation
BRANCH=${FERA_BRANCH:-main}
CFG=${FERA_CONFIG:-configs/mdm2.yaml}
RUNS=${FERA_RUNS:-"--rep 1 --leg bound --stage vanish --max-runs 4|--rep 1 --leg free --stage vanish --max-runs 2"}
GRACE=${FERA_GRACE_MIN:-60}
FAIL_GRACE=${FERA_FAIL_GRACE_MIN:-240}

stamp() { echo "[$(date '+%F %T')] $*"; }
stamp "boot: pod ${RUNPOD_POD_ID:-?}, branch $BRANCH, config $CFG"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

if [ -n "${RUNPOD_API_KEY:-}" ]; then
    runpodctl config --apiKey "$RUNPOD_API_KEY" > /dev/null 2>&1 \
        && stamp "runpodctl authorized: pod will stop itself" \
        || stamp "WARNING: runpodctl config failed; pod may not stop itself"
else
    stamp "WARNING: no RUNPOD_API_KEY; pod cannot stop itself"
fi

git -C "$REPO" fetch -q origin "$BRANCH" && git -C "$REPO" checkout -q "$BRANCH" \
    && git -C "$REPO" reset -q --hard "origin/$BRANCH"
stamp "code at $(git -C "$REPO" rev-parse --short HEAD)"

bash "$REPO/sims/pod_setup.sh" || stamp "WARNING: pod_setup.sh exited $?"

fail() {   # stay up so the logs can be read, then stop
    stamp "FAILED: $*; staying up ${FAIL_GRACE} min for debugging"
    touch /workspace/FAILED
    sleep "$((FAIL_GRACE * 60))"
    stamp "stopping pod (after failure)"
    runpodctl stop pod "$RUNPOD_POD_ID"
    exit 1
}
rm -f /workspace/FAILED

first=1
if [ "${FERA_SMOKE:-1}" = 1 ] && [ "${FERA_WAIT_PREP:-0}" != 1 ]; then
    SMOKE_CFG=/workspace/smoke.yaml
    /workspace/mamba/envs/fera/bin/python - "$REPO/$CFG" "$SMOKE_CFG" <<'PY'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
c.update(name=c["name"] + "_smoke", replicates=2, equil_ps=20, ns_per_tape=0.02,
         energy_frequency=1000, buffered_coords_freq=1000)
yaml.safe_dump(c, open(sys.argv[2], "w"))
PY
    stamp "smoke test: prep + 1 tape per leg on $SMOKE_CFG"
    PREP=1 STOP_POD=0 bash "$REPO/sims/overnight.sh" "$SMOKE_CFG" --rep 1 --leg bound --stage vanish --max-runs 1 \
        && PREP=0 STOP_POD=0 bash "$REPO/sims/overnight.sh" "$SMOKE_CFG" --rep 1 --leg free --stage vanish --max-runs 1 \
        || fail "smoke test (log: /workspace/runs/*_smoke/overnight_*.log)"
    stamp "smoke test passed"
fi
if [ "${FERA_WAIT_PREP:-0}" = 1 ]; then
    NAME=$(grep -E '^name:' "$REPO/$CFG" | awk '{print $2}')
    stamp "waiting for prep tree at /workspace/runs/$NAME/PREP_READY"
    until [ -f "/workspace/runs/$NAME/PREP_READY" ]; do sleep 30; done
    stamp "prep tree present"
    first=0
fi
IFS='|' read -ra SETS <<< "$RUNS"
for args in "${SETS[@]}"; do
    stamp "recording: run.py $args"
    # Prep only before the first set (it's resumable either way); overnight.sh logs
    # to /workspace/runs/<name>/overnight_<pod>.log.
    # shellcheck disable=SC2086
    PREP=$first STOP_POD=0 bash "$REPO/sims/overnight.sh" "$CFG" $args \
        || fail "run.py $args (log: /workspace/runs/*/overnight_*.log)"
    first=0
done

stamp "all sets done; staying up ${GRACE} min so results can be pulled"
sleep "$((GRACE * 60))"
stamp "stopping pod"
runpodctl stop pod "$RUNPOD_POD_ID" || stamp "WARNING: stop failed; stop the pod by hand"
