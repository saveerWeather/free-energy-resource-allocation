#!/usr/bin/env bash
# Laptop-side helpers for recording on Community pods, which can't mount a network
# volume: code goes up with rsync, tapes come back down with rsync.
#
# A pods file lists one pod per line: <label> <ip> <port>   (from the pod's direct SSH,
# `ssh root@<ip> -p <port>`; it changes on every pod stop/start). Blank lines and
# lines starting with # are ignored.
#
#   sims/pods.sh push-code  <podsfile>                 # repo -> /workspace on every pod
#   sims/pods.sh copy-prep  <podsfile> <name> <label>  # finished prep tree: pod <label> -> laptop -> others
#   sims/pods.sh pull       <podsfile> <name> [secs]   # tapes -> data/<name>/, every secs (default 1800)
#   sims/pods.sh status     <podsfile> <name>          # last lines of each pod's run log
#
# `pull` loops forever; run it as:  caffeinate -i sims/pods.sh pull pods.txt mdm2_pip2
# (ssh/rsync inside the `while read` loops get </dev/null or -n so they don't eat
# the rest of the pods list from stdin.)
set -uo pipefail

cd "$(dirname "$0")/.."
SSH="ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -o ServerAliveInterval=30"

pods() { grep -v -E '^\s*(#|$)' "$1"; }
stamp() { echo "[$(date '+%F %T')] $*"; }

cmd=${1:-}; shift || true
case "$cmd" in
push-code)
    pods "$1" | while read -r label ip port; do
        stamp "push code -> $label"
        rsync -az -e "$SSH -p $port" --exclude .git --exclude data/ --exclude notebooks/ \
            --exclude report/ --exclude viz/ --exclude '*.pdf' ./ \
            "root@$ip:/workspace/free-energy-resource-allocation/" < /dev/null
    done ;;
copy-prep)
    podsfile=$1; name=$2; src=$3
    stage=data/$name/_prep
    mkdir -p "$stage"
    read -r _ sip sport < <(pods "$podsfile" | awk -v l="$src" '$1 == l')
    [ -n "${sip:-}" ] || { echo "no pod labelled $src in $podsfile"; exit 1; }
    stamp "pull prep tree <- $src"
    # Everything prep wrote (inputs, equilibrations, restraint, run folders with their
    # cfg/pert/symlinks), none of the recording outputs.
    PREP_EXCLUDES=(--exclude 'simfile*.dat' --exclude 'somd.out' --exclude 'somd.err'
                   --exclude '*.dcd' --exclude '*.s3' --exclude '*.s3.previous'
                   --exclude 'moves.dat' --exclude 'gradients.dat' --exclude 'latest.pdb'
                   --exclude 'run_log*.jsonl' --exclude 'overnight_*.log')
    rsync -az -e "$SSH -p $sport" "${PREP_EXCLUDES[@]}" \
        "root@$sip:/workspace/runs/$name/" "$stage/" < /dev/null
    pods "$podsfile" | while read -r label ip port; do
        [ "$label" = "$src" ] && continue
        stamp "push prep tree -> $label"
        $SSH -n -p "$port" "root@$ip" "mkdir -p /workspace/runs/$name"
        rsync -az -e "$SSH -p $port" "${PREP_EXCLUDES[@]}" "$stage/" \
            "root@$ip:/workspace/runs/$name/" < /dev/null \
            && $SSH -n -p "$port" "root@$ip" "touch /workspace/runs/$name/PREP_READY"
    done ;;
pull)
    podsfile=$1; name=$2; every=${3:-1800}
    dest=data/$name
    mkdir -p "$dest"
    while true; do
        pods "$podsfile" | while read -r label ip port; do
            # Tapes + what's needed to read them; no trajectories or restart files.
            if rsync -az -e "$SSH -p $port" --prune-empty-dirs \
                --include '*/' --include 'manifest.json' --include 'run_log*.jsonl' \
                --include 'overnight_*.log' --include 'simfile.dat' --include 'somd.cfg' \
                --include 'somd.out' --include 'somd.err' --include 'restraint_1.*' \
                --exclude '*' "root@$ip:/workspace/runs/$name/" "$dest/" < /dev/null; then
                n=$(find "$dest" -name simfile.dat | wc -l | tr -d ' ')
                stamp "pulled $label ($n tapes on laptop)"
            else
                stamp "WARNING: pull from $label ($ip:$port) failed"
            fi
        done
        sleep "$every"
    done ;;
status)
    pods "$1" | while read -r label ip port; do
        echo "== $label"
        $SSH -n -p "$port" "root@$ip" "tail -n 3 /workspace/runs/$2/overnight_*.log 2>/dev/null || echo 'no log yet'"
    done ;;
*)
    sed -n '2,16p' "$0"; exit 1 ;;
esac
