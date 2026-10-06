#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/WunderFunder-creative"
while systemctl --user is-active --quiet wunder-wave18-finish-composite-parents; do sleep 10; done
while systemctl --user is-active --quiet wunder-wave18-composite-student-3 || systemctl --user is-active --quiet wunder-wave18-composite-student-4; do sleep 10; done
WNN_CPU_PCT=6 WNN_MEM_MAX=24G scripts/limited .venv/bin/python research/iterations/2026-10-05-wave18/evaluate_composite.py > research/iterations/2026-10-05-wave18/evaluate-auto.log 2>&1
