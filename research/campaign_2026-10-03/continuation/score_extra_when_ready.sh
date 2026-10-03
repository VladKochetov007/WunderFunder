#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
state=research/campaign_2026-10-03/continuation
wait_for() {
while true; do
  [[ ! -f $state/RESOURCE_FAILURE ]] || { sleep 10; continue; }
  complete=true
  for name in "$@"; do
    record=research/campaign_2026-09-29/jobs/$name.json
    if [[ ! -f $record ]]; then complete=false; continue; fi
    status=$(.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$record")
    [[ $status != failed && $status != interrupted ]] || { echo "failed confirmation: $name"; exit 1; }
    [[ $status == finished ]] || complete=false
  done
  [[ $complete == false ]] || break
  sleep 10
done
}
wait_for c20261003_w320_s3_f0 c20261003_w320_s3_f1
"$state/run_cpu.sh" scoring .venv/bin/python scripts/screen_score.py \
  research/campaign_2026-10-03/scores/width320_prereg.json \
  --output research/campaign_2026-10-03/scores/width320_prereg_report.json
printf '%s\n' "$(date -Is) width320 scored" > "$state/WIDTH_SCORES_READY"
wait_for c20261003_combo_s3_f2 c20261003_combo_s3_f3
"$state/run_cpu.sh" scoring .venv/bin/python scripts/screen_score.py \
  research/campaign_2026-10-03/scores/combo_fold23_prereg.json \
  --output research/campaign_2026-10-03/scores/combo_fold23_prereg_report.json
printf '%s\n' "$(date -Is) fixed fold23 confirmation scored" > "$state/FOLD23_SCORES_READY"
