#!/usr/bin/env bash
# PROPOSAL ONLY. Controller's independent receipt must say queue_finished=true.
set -euo pipefail
receipt=${1:?Independent controller review receipt path required; actor cannot authorize this job}
cd /home/vlad/development/wunder/wnn_connectome_starterpack
scope=research/campaign_2026-10-03/continuation/round4/live_capture
tag="r4_capture_$(date +%Y%m%dT%H%M%S)_${BASHPID}"
# One controlled GPU family; <=590s,8cores,24GiB RAM. Context+allocator observed by runner.
exec env WNN_SLICE=wnn-campaign.slice WNN_RUNTIME_MAX=600 WNN_CPU_QUOTA=800% WNN_THREADS=8 WNN_MEM_MAX=24G WNN_GPU_DUTY=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="src:$scope" scripts/limited timeout --signal=TERM 590 .venv/bin/python "$scope/runner.py" --device cuda --mode all --review-receipt "$receipt" --output "$scope/$tag" > "$scope/logs/$tag.log" 2>&1
