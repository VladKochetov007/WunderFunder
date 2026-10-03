#!/usr/bin/env bash
# One controller-approved counterfactual; campaign admission supplies resource limits.
set -euo pipefail
cd /home/vlad/development/wunder/wnn_connectome_starterpack
[[ ${WNN_IN_CAMPAIGN_JOB:-0} == 1 ]] || exit 75
[[ ${WNN_SLICE:-} == wnn-campaign-gpu.slice ]] || exit 75
research_deadline=$(date -d '2026-10-03 20:00:00 UTC' +%s)
research_now=$(date +%s)
(( research_deadline - research_now >= 325 )) || exit 75
export WNN_RESEARCH_DEADLINE_UTC=1791057600
export WNN_CAMPAIGN_VRAM_BYTES=5500000000
export WNN_STATE_DRIFT_PAYLOAD_SHA256
WNN_STATE_DRIFT_PAYLOAD_SHA256=$(sha256sum research/campaign_2026-10-03/continuation/round6/controller_state_drift_payload.sh | cut -d ' ' -f 1)
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=src:research/campaign_2026-10-03/continuation/round4/live_capture_v4
scope=research/campaign_2026-10-03/continuation/round6/state_drift_v2
exec timeout --signal=TERM --kill-after=5s 300 .venv/bin/python "$scope/run_real.py" "$@"
