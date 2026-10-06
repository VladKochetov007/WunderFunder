#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/WunderFunder-creative"
D="$PWD/research/iterations/2026-10-05-wave33"
export WUNDER_RESEARCH_ROOT="$PWD" PYTHONPATH="$D/scripts:$HOME/WunderFunder/src" WNN_CPU_PCT=6 WNN_MEM_MAX=32G
scripts/limited .venv/bin/python "$D/export_train_probes.py" > "$D/train-probes.log" 2>&1
scripts/limited "$HOME/venv311/bin/python" "$D/native_replay.py" --train-only > "$D/native-train-repair.log" 2>&1
scripts/limited .venv/bin/python "$D/evaluate_native.py" > "$D/evaluate-native-repair.log" 2>&1
cp "$D/native-input-parity.json" "$D/schema-repair-complete.json"
