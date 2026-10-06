#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/WunderFunder-creative"
for s in 5 6; do
 while systemctl --user is-active --quiet "wunder-wave37-parent-${s}"; do sleep 10;done
 test -f "cache/pred_w21final_distilled_s${s}_train.npy"
 test -f "cache/pred_w21final_distilled_s${s}_valid.npy"
done
WNN_CPU_PCT=6 WNN_MEM_MAX=24G scripts/limited .venv/bin/python research/iterations/2026-10-05-wave37/prepare_teacher.py > research/iterations/2026-10-05-wave37/prepare.log 2>&1
