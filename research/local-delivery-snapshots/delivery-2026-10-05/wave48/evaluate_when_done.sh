#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/WunderFunder-creative"
D="$PWD/research/iterations/2026-10-05-wave48"
export WNN_CPU_PCT=6 WNN_MEM_MAX=24G
for kind in comp full;do
 while systemctl --user is-active --quiet wunder-wave43-evaluate-$kind;do sleep 5;done
 scripts/limited .venv/bin/python "$D/evaluate_$kind.py" > "$D/evaluate-$kind.log" 2>&1
 while systemctl --user is-active --quiet wunder-wave43-native-$kind;do sleep 5;done
 test -f "research/iterations/2026-10-05-wave43/native-$kind-report.json"
 scripts/limited .venv/bin/python "$D/evaluate_native_$kind.py" > "$D/evaluate-native-$kind.log" 2>&1
done
