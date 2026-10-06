#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/WunderFunder-creative"
kind=$1
for seed in 3 4; do
 while systemctl --user is-active --quiet "wunder-wave36-${kind}-${seed}"; do sleep 10;done
 test -f "cache/pred_w36_${kind}_s${seed}_valid.npy"
done
if [ "$kind" = comp ]; then while systemctl --user is-active --quiet wunder-wave33-evaluate; do sleep 10;done;fi
WNN_CPU_PCT=6 WNN_MEM_MAX=24G scripts/limited .venv/bin/python "research/iterations/2026-10-05-wave36/evaluate_${kind}.py" > "research/iterations/2026-10-05-wave36/evaluate-${kind}.log" 2>&1
