#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/WunderFunder-creative"
D="$PWD/research/iterations/2026-10-05-wave43"
for kind in full comp;do
 while systemctl --user is-active --quiet wunder-wave43-evaluate-$kind;do sleep 5;done
 test -f "$D/$kind-report.json"
done
.venv/bin/python "$D/native_gate.py" > "$D/native-gate.log" 2>&1
