#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/WunderFunder-creative"
D="$PWD/research/iterations/2026-10-05-wave43"
for kind in full comp; do
 while systemctl --user is-active --quiet wunder-wave43-evaluate-$kind;do sleep 10;done
 test -f "$D/$kind-report.json"
done
.venv/bin/python "$D/native_gate.py" > "$D/native-gate.log" 2>&1
# Deadline review: report the preregistered gate, but finish native quality for this already-preparing final candidate regardless.
export WUNDER_RESEARCH_ROOT="$PWD" PYTHONPATH="$D/scripts:$HOME/WunderFunder/src" WNN_CPU_PCT=6 WNN_MEM_MAX=24G
for kind in full comp;do
 scripts/limited .venv/bin/python "$D/export_oof.py" "$kind" > "$D/export-$kind.log" 2>&1
 scripts/limited .venv/bin/python "$D/export_train_probes.py" "$kind" > "$D/train-probes-$kind.log" 2>&1
 scripts/limited "$HOME/venv311/bin/python" "$D/native_replay.py" "$kind" > "$D/native-replay-$kind.log" 2>&1
 scripts/limited .venv/bin/python "$D/evaluate_native_$kind.py" > "$D/evaluate-native-$kind.log" 2>&1
done
