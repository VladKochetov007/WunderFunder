#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/WunderFunder-creative"
D="$PWD/research/iterations/2026-10-05-wave36"
while systemctl --user is-active --quiet wunder-wave33-native-repair; do sleep 10;done
test -f research/iterations/2026-10-05-wave33/schema-repair-complete.json
export WUNDER_RESEARCH_ROOT="$PWD" PYTHONPATH="$D/scripts:$HOME/WunderFunder/src" WNN_CPU_PCT=6 WNN_MEM_MAX=24G
scripts/limited .venv/bin/python "$D/export_train_probes.py" comp > "$D/train-probes-comp.log" 2>&1
scripts/limited "$HOME/venv311/bin/python" "$D/native_replay.py" comp --train-only > "$D/native-replay-comp-repaired.log" 2>&1
scripts/limited .venv/bin/python "$D/evaluate_native_comp.py" > "$D/evaluate-native-comp.log" 2>&1
