#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/WunderFunder-creative"
while systemctl --user is-active --quiet wunder-wave19-student-3 || systemctl --user is-active --quiet wunder-wave19-student-4; do sleep 10; done
WNN_CPU_PCT=6 WNN_MEM_MAX=24G scripts/limited .venv/bin/python research/iterations/2026-10-05-wave19/evaluate.py > research/iterations/2026-10-05-wave19/evaluate-auto.log 2>&1
