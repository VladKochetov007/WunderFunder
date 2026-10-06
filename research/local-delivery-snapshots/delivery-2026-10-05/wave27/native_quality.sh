#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/WunderFunder-creative"
D="$PWD/research/iterations/2026-10-05-wave27"
while systemctl --user is-active --quiet wunder-wave27-evaluate || systemctl --user is-active --quiet wunder-wave22-native-quality; do sleep 10; done
export WUNDER_RESEARCH_ROOT="$PWD" PYTHONPATH="$PWD/research/iterations/2026-10-05-wave27/scripts:$HOME/WunderFunder/src" WNN_CPU_PCT=6 WNN_MEM_MAX=24G
test -f "$D/wave-report.json"
scripts/limited .venv/bin/python "$D/export_oof.py" > "$D/export-oof.log" 2>&1
scripts/limited "$HOME/venv311/bin/python" "$D/native_replay.py" > "$D/native-replay.log" 2>&1
scripts/limited .venv/bin/python "$D/evaluate_native.py" > "$D/evaluate-native.log" 2>&1
