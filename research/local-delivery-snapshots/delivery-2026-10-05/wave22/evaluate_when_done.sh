#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/WunderFunder-creative"
while systemctl --user is-active --quiet wunder-wave22-finish-parents; do sleep 10; done
while systemctl --user is-active --quiet wunder-wave22-wide-3 || systemctl --user is-active --quiet wunder-wave22-wide-4 || systemctl --user is-active --quiet wunder-wave22-control; do sleep 10; done
WNN_CPU_PCT=6 WNN_MEM_MAX=24G scripts/limited .venv/bin/python research/iterations/2026-10-05-wave22/evaluate.py > research/iterations/2026-10-05-wave22/evaluate.log 2>&1
