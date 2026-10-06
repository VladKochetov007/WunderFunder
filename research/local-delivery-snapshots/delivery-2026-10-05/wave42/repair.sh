#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/WunderFunder-creative"
D="$PWD/research/iterations/2026-10-05-wave42"
export WUNDER_RESEARCH_ROOT="$PWD" PYTHONPATH="$PWD/research/iterations/2026-10-05-wave13/scripts:$HOME/WunderFunder/src" WNN_CPU_PCT=6 WNN_MEM_MAX=24G
scripts/limited .venv/bin/python "$D/prepare_probe.py" > "$D/probe.log" 2>&1
scripts/limited .venv/bin/python "$D/invalidate.py" > "$D/invalidate.log" 2>&1
scripts/limited "$HOME/venv311/bin/python" "$D/replay_train.py" > "$D/replay.log" 2>&1
scripts/limited .venv/bin/python research/iterations/2026-10-05-wave13/evaluate_precision.py > "$D/evaluate.log" 2>&1
