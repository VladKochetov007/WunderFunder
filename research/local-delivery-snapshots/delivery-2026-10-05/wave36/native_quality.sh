#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/WunderFunder-creative"
D="$PWD/research/iterations/2026-10-05-wave36";kind=$1
while systemctl --user is-active --quiet "wunder-wave36-evaluate-${kind}"; do sleep 10; done
if [ "$kind" = comp ]; then while systemctl --user is-active --quiet wunder-wave33-native-quality; do sleep 10; done;fi
test -f "$D/${kind}-report.json"
export WUNDER_RESEARCH_ROOT="$PWD" PYTHONPATH="$D/scripts:$HOME/WunderFunder/src" WNN_CPU_PCT=6 WNN_MEM_MAX=24G
scripts/limited .venv/bin/python "$D/export_oof.py" "$kind" > "$D/export-${kind}.log" 2>&1
scripts/limited .venv/bin/python "$D/export_train_probes.py" "$kind" > "$D/train-probes-${kind}.log" 2>&1
scripts/limited "$HOME/venv311/bin/python" "$D/native_replay.py" "$kind" > "$D/native-replay-${kind}.log" 2>&1
scripts/limited .venv/bin/python "$D/evaluate_native_${kind}.py" > "$D/evaluate-native-${kind}.log" 2>&1
