#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/WunderFunder-creative"
kind=$1
for seed in 3 4; do
 while systemctl --user is-active --quiet "wunder-wave45-${kind}-${seed}"; do sleep 10;done
 test -f "cache/pred_w45_${kind}_s${seed}_valid.npy"
done
WNN_CPU_PCT=6 WNN_MEM_MAX=24G scripts/limited .venv/bin/python "research/iterations/2026-10-05-wave45/evaluate_${kind}.py" > "research/iterations/2026-10-05-wave45/evaluate-${kind}.log" 2>&1
