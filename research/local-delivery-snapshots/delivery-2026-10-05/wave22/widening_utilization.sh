#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/WunderFunder-creative"
D="$PWD/research/iterations/2026-10-05-wave22"
while systemctl --user is-active --quiet wunder-wave22-wide-3 || systemctl --user is-active --quiet wunder-wave22-wide-4; do sleep 10; done
export WUNDER_RESEARCH_ROOT="$PWD" PYTHONPATH="$PWD/research/iterations/2026-10-05-wave6/scripts:$HOME/WunderFunder/src" WNN_CPU_PCT=6 WNN_MEM_MAX=8G
scripts/limited .venv/bin/python "$D/widening_utilization.py" > "$D/widening-utilization.log" 2>&1
