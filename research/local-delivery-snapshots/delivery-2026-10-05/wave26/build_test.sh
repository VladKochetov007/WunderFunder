#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/WunderFunder-creative"
D="$PWD/research/iterations/2026-10-05-wave26"
export WUNDER_RESEARCH_ROOT="$PWD" PYTHONPATH="$PWD/research/iterations/2026-10-05-wave23/scripts:$HOME/WunderFunder/src" WNN_CPU_PCT=6 WNN_MEM_MAX=16G
scripts/limited .venv/bin/python "$D/export_fixtures.py" > "$D/fixtures.log" 2>&1
scripts/limited /bin/bash "$D/native-class/build.sh" "$HOME/venv311/bin/python" v2 avx2 v3 > "$D/build.log" 2>&1
for level in v2 avx2 v3; do
 scripts/limited "$HOME/venv311/bin/python" "$D/test_native.py" "$level" >> "$D/test.log" 2>&1
done
