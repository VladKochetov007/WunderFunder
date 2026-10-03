#!/usr/bin/env bash
# Controller-owned integration: invoke only through campaign_job -> run_gpu.sh.
set -euo pipefail
cd /home/vlad/development/wunder/wnn_connectome_starterpack
[[ ${WNN_IN_CAMPAIGN_JOB:-0} == 1 ]] || exit 75
[[ ${WNN_SLICE:-} == wnn-campaign-gpu.slice ]] || exit 75
research_deadline=$(date -d '2026-10-03 20:00:00 UTC' +%s)
research_now=$(date +%s)
(( research_deadline - research_now >= 605 )) || exit 75
receipt=${1:?Hash-bound independent review receipt required}
output=${2:?Fresh output directory required}
scope=research/campaign_2026-10-03/continuation/round4/live_capture_v4
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="src:$scope"
exec timeout --signal=TERM --kill-after=5s 585 .venv/bin/python "$scope/runner.py" \
    --device cuda --mode all --review-receipt "$receipt" --output "$output"
