#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
state=research/campaign_2026-10-03/continuation
record=research/campaign_2026-09-29/jobs/c20261003_ctrl_s4_f1.json
while true; do
  if [[ -f $record ]]; then
    status=$(.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$record")
    [[ $status != failed && $status != interrupted ]] || { echo 'matched control failed; retain pending score'; exit 1; }
    [[ $status != finished ]] || break
  fi
  if [[ -f $state/RESOURCE_FAILURE ]]; then
    sleep 10; continue
  fi
  sleep 10
done
while [[ -f $state/RESOURCE_FAILURE ]]; do sleep 10; done
"$state/run_cpu.sh" scoring .venv/bin/python scripts/screen_score.py \
  research/campaign_2026-10-03/scores/combo_matched_2seed.json \
  --output research/campaign_2026-10-03/scores/combo_matched_2seed_report.json
printf '%s matched two-seed combination scoring complete\n' "$(date -Is)" > "$state/SCORES_READY"
