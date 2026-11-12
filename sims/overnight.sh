#!/usr/bin/env bash
# Overnight tape recording: prep -> run every window (one SOMD process at a time,
# replicate-major, resumable) -> stop the pod so it doesn't bill idle.
#
#   nohup bash sims/overnight.sh > /dev/null 2>&1 &
#
# Re-running is safe: prep and run.py both skip finished work.
# No `set -e`: the pod is stopped at the end whether or not a step failed.

REPO=/workspace/free-energy-resource-allocation
CFG=$REPO/configs/toluene.yaml
LOG=/workspace/runs/toluene/overnight.log
mkdir -p "$(dirname "$LOG")"
exec >> "$LOG" 2>&1

stamp() { echo "[$(date '+%F %T')] $*"; }

export MAMBA_ROOT_PREFIX=/workspace/mamba
eval "$(/workspace/bin/micromamba shell hook -s bash)"
micromamba activate fera
cd "$REPO"

stamp "start (pod ${RUNPOD_POD_ID:-?}, commit $(git rev-parse --short HEAD))"
stamp "python $(which python) | somd $(which somd-freenrg) | gmx $(which gmx)"
if ! runpodctl get pod "$RUNPOD_POD_ID" > /dev/null 2>&1; then
    stamp "WARNING: runpodctl is not authorized right now; the final 'stop pod' may fail."
    stamp "WARNING: fix before the run ends with: runpodctl config --apiKey <key>"
fi

stamp "prep: begin"
python -u sims/prep.py "$CFG"
stamp "prep: exit $?"

stamp "run: begin"
python -u sims/run.py "$CFG" --jobs 1
stamp "run: exit $?"

stamp "stopping pod"
runpodctl stop pod $RUNPOD_POD_ID
