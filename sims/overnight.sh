#!/usr/bin/env bash
# Unattended tape recording on one pod: prep -> run.py -> stop the pod so it doesn't
# bill idle.
#
#   nohup bash sims/overnight.sh configs/mdm2.yaml --rep 1 > /dev/null 2>&1 &
#   nohup bash sims/overnight.sh                            > /dev/null 2>&1 &   # toluene
#
# Arguments after the config go to run.py (e.g. --rep 2 to record one replicate per
# pod). Prep runs only when PREP=1 (default) and is skipped per step if done, so on
# pods that received a finished prep tree it only re-checks the folders.
# Re-running is safe: prep and run.py both skip finished work.
# No `set -e`: the pod is stopped at the end whether or not a step failed.

REPO=/workspace/free-energy-resource-allocation
CFG=${1:-configs/toluene.yaml}
shift || true

export MAMBA_ROOT_PREFIX=/workspace/mamba
eval "$(/workspace/bin/micromamba shell hook -s bash)"
micromamba activate fera
cd "$REPO"

# Log next to the run folders, one file per pod. (Name read with the env's python,
# which has pyyaml; the pod image's system python may not.)
NAME=$(python -c "import yaml; print(yaml.safe_load(open('$CFG'))['name'])")
LOG=/workspace/runs/$NAME/overnight_${RUNPOD_POD_ID:-local}.log
mkdir -p "$(dirname "$LOG")"
exec >> "$LOG" 2>&1

stamp() { echo "[$(date '+%F %T')] $*"; }

stamp "start (pod ${RUNPOD_POD_ID:-?}, config $CFG, run args: $*)"
stamp "python $(which python) | somd $(which somd-freenrg) | gmx $(which gmx)"
if ! runpodctl get pod "$RUNPOD_POD_ID" > /dev/null 2>&1; then
    stamp "WARNING: runpodctl is not authorized right now; the final 'stop pod' may fail."
    stamp "WARNING: fix before the run ends with: runpodctl config --apiKey <key>"
fi

if [ "${PREP:-1}" = 1 ]; then
    stamp "prep: begin"
    python -u sims/prep.py "$CFG"
    stamp "prep: exit $?"
fi

stamp "run: begin"
python -u sims/run.py "$CFG" --jobs 1 "$@"
stamp "run: exit $?"

if [ "${STOP_POD:-1}" = 1 ]; then
    stamp "stopping pod"
    runpodctl stop pod "$RUNPOD_POD_ID"
fi
