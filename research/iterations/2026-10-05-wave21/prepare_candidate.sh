#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/WunderFunder-creative"
D="$PWD/research/iterations/2026-10-05-wave21"
while systemctl --user is-active --quiet wunder-wave21-final-student-5 || systemctl --user is-active --quiet wunder-wave21-final-student-6; do sleep 10; done
test -f artifacts/runs/w21final_distilled_s5/model_ep1.pt
test -f artifacts/runs/w21final_distilled_s6/model_ep1.pt
export WUNDER_RESEARCH_ROOT="$PWD" PYTHONPATH="$PWD/research/iterations/2026-10-05-wave6/scripts:$HOME/WunderFunder/src" WNN_CPU_PCT=6 WNN_MEM_MAX=16G
scripts/limited .venv/bin/python "$D/export_final.py" > "$D/export.log" 2>&1
"$HOME/venv311/bin/python" "$D/build_candidate.py" > "$D/build.log" 2>&1
mkdir "$D/unpacked"
"$HOME/venv311/bin/python" -m zipfile -e "$D/distilled_twostream_h256_pair.zip" "$D/unpacked"
/usr/bin/time -v "$HOME/venv311/bin/python" research/iterations/2026-10-05-wave2/audit_submission.py "$D/unpacked" "$D/submission-audit.json" > "$D/audit.log" 2>&1
for level in v2 avx2 v3; do
 "$HOME/venv311/bin/python" "$D/parity_final.py" "$level" >> "$D/parity.log" 2>&1
done
