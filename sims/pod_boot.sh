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
#   RUNPOD_API_KEY  set from a RunPod secret ({{ RUNPOD_SECRET_<name> }}); without it the
#                 pod can NOT stop itself and keeps billing until stopped by hand.
REPO=/workspace/free-energy-resource-allocation
BRANCH=${FERA_BRANCH:-main}
CFG=${FERA_CONFIG:-configs/mdm2.yaml}
RUNS=${FERA_RUNS:-"--rep 1 --leg bound --stage vanish --max-runs 4|--rep 1 --leg free --stage vanish --max-runs 2"}
GRACE=${FERA_GRACE_MIN:-60}

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

first=1
IFS='|' read -ra SETS <<< "$RUNS"
for args in "${SETS[@]}"; do
    stamp "recording: run.py $args"
    # Prep only before the first set (it's resumable either way); overnight.sh logs
    # to /workspace/runs/<name>/overnight_<pod>.log.
    # shellcheck disable=SC2086
    PREP=$first STOP_POD=0 bash "$REPO/sims/overnight.sh" "$CFG" $args
    first=0
done

stamp "all sets done; staying up ${GRACE} min so results can be pulled"
sleep "$((GRACE * 60))"
stamp "stopping pod"
runpodctl stop pod "$RUNPOD_POD_ID" || stamp "WARNING: stop failed; stop the pod by hand"
