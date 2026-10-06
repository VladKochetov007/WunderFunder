#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/WunderFunder-creative"
while systemctl --user is-active --quiet wunder-wave29-scaled-3 || systemctl --user is-active --quiet wunder-wave29-scaled-4 || systemctl --user is-active --quiet wunder-wave28-evaluate; do sleep 10; done
WNN_CPU_PCT=6 WNN_MEM_MAX=24G scripts/limited .venv/bin/python research/iterations/2026-10-05-wave29/evaluate.py > research/iterations/2026-10-05-wave29/evaluate.log 2>&1
