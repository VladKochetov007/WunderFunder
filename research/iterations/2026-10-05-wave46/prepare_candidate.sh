#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/WunderFunder-creative"
D="$PWD/research/iterations/2026-10-05-wave46"
for seed in 15 16;do
 while systemctl --user is-active --quiet wunder-wave46-final-$seed;do sleep 10;done
 test -f "artifacts/runs/w46final_distilled_s$seed/model_ep1.pt"
done
export WUNDER_RESEARCH_ROOT="$PWD" PYTHONPATH="$D/scripts:$HOME/WunderFunder/src" WNN_CPU_PCT=6 WNN_MEM_MAX=24G
scripts/limited .venv/bin/python "$D/export_final.py" > "$D/export.log" 2>&1
"$HOME/venv311/bin/python" "$D/build_candidate.py" > "$D/build.log" 2>&1
mkdir "$D/unpacked"
"$HOME/venv311/bin/python" -m zipfile -e "$D/distilled_h256_four_seeds.zip" "$D/unpacked"
for level in v2 avx2 v3;do scripts/limited "$HOME/venv311/bin/python" "$D/parity_final.py" "$level" >> "$D/parity.log" 2>&1;done
scripts/limited "$HOME/venv311/bin/python" "$D/functional_audit.py" "$D/unpacked" "$D/submission-audit.json" > "$D/audit.log" 2>&1

"$HOME/venv311/bin/python" "$D/build_candidate.py" distilled_twostream_seed_four > "$D/build-mixed.log" 2>&1
mkdir "$D/unpacked_mixed"
"$HOME/venv311/bin/python" -m zipfile -e "$D/distilled_twostream_seed_four.zip" "$D/unpacked_mixed"
scripts/limited "$HOME/venv311/bin/python" "$D/functional_audit.py" "$D/unpacked_mixed" "$D/mixed-audit.json" > "$D/audit-mixed.log" 2>&1
