#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/WunderFunder-creative"
D="$PWD/research/iterations/2026-10-05-wave22"
while systemctl --user is-active --quiet wunder-wave22-finish-parents; do sleep 10; done
export WUNDER_RESEARCH_ROOT="$PWD" PYTHONPATH="$PWD/research/iterations/2026-10-05-wave6/scripts:$HOME/WunderFunder/src"
WNN_CPU_PCT=6 WNN_MEM_MAX=16G scripts/limited .venv/bin/python "$D/native_assets.py" > "$D/native-assets.log" 2>&1
cd "$D"
"$HOME/ds" project job execute -p bt1dqf9jg0tuhggcrj4k -c native-ds.yaml > datasphere.log 2>&1
