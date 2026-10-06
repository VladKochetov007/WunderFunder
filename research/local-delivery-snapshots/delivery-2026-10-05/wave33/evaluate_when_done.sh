#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/WunderFunder-creative"
for mode in uniform soft; do
 for seed in 3 4; do
  while systemctl --user is-active --quiet "wunder-wave33-${mode}-${seed}"; do sleep 10; done
  test -f "cache/pred_w33_${mode}_s${seed}_train.npy"
 done
done
WNN_CPU_PCT=6 WNN_MEM_MAX=24G scripts/limited .venv/bin/python research/iterations/2026-10-05-wave33/evaluate.py > research/iterations/2026-10-05-wave33/evaluate.log 2>&1
