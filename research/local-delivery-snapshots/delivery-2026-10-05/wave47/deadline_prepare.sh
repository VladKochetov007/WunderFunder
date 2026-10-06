#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/WunderFunder-creative"
D="$PWD/research/iterations/2026-10-05-wave47"
G="$PWD/research/iterations/2026-10-05-wave45/native-gate.json"
while [ ! -f "$G" ];do
 if [ "$(date -u +%Y%m%d%H%M)" -ge 202610052332 ];then echo 'Missed latest final-fit start cutoff' > "$D/decision.log";exit 0;fi
 sleep 10
done
if ! .venv/bin/python -c 'import json,sys;sys.exit(not json.load(open(sys.argv[1]))["proceed_native"])' "$G";then echo 'Wave45 failed predeclared quality gate; final fit skipped' > "$D/decision.log";exit 0;fi
for wave in 43 45;do for seed in 3 4 13 14;do
 while systemctl --user is-active --quiet "wunder-wave$wave-comp-$seed";do sleep 10;done
done;done
if [ "$(date -u +%Y%m%d%H%M)" -ge 202610052332 ];then echo 'Missed latest final-fit start cutoff' > "$D/decision.log";exit 0;fi
echo "START $(date -Is) quality gate passed" > "$D/decision.log"
bash "$D/final_student_lane.sh" 5 2 & p5=$!
bash "$D/final_student_lane.sh" 6 5 & p6=$!
wait "$p5";wait "$p6"
export WUNDER_RESEARCH_ROOT="$PWD" PYTHONPATH="$D/scripts:$HOME/WunderFunder/src" WNN_CPU_PCT=6 WNN_MEM_MAX=24G
scripts/limited .venv/bin/python "$D/export_final.py" > "$D/export.log" 2>&1
"$HOME/venv311/bin/python" "$D/build_candidate.py" > "$D/build.log" 2>&1
mkdir "$D/unpacked"
"$HOME/venv311/bin/python" -m zipfile -e "$D/distilled_short_h256_pair.zip" "$D/unpacked"
for level in v2 avx2 v3;do scripts/limited "$HOME/venv311/bin/python" "$D/parity_final.py" "$level" >> "$D/parity.log" 2>&1;done
scripts/limited "$HOME/venv311/bin/python" "$D/functional_audit.py" "$D/unpacked" "$D/pair-audit.json" > "$D/pair-audit.log" 2>&1
mkdir "$D/unpacked_four"
"$HOME/venv311/bin/python" -m zipfile -e "$D/distilled_short_h256_four.zip" "$D/unpacked_four"
scripts/limited "$HOME/venv311/bin/python" "$D/functional_audit.py" "$D/unpacked_four" "$D/four-audit.json" > "$D/four-audit.log" 2>&1
