#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/WunderFunder-creative"
D="$PWD/research/iterations/2026-10-05-wave26"
while systemctl --user is-active --quiet wunder-wave26-native-build; do sleep 10; done
for level in v2 avx2 v3; do test -f "$D/parity-$level.json"; done
WNN_CPU_PCT=6 WNN_MEM_MAX=12G scripts/limited "$HOME/venv311/bin/python" "$D/compatibility.py" > "$D/compatibility.log" 2>&1
