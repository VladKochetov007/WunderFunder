#!/usr/bin/env bash
set -euo pipefail
sha256sum --check --status research/campaign_2026-10-06/bandit_continuation/own_baseline_E520/E520_PROFILE_V7_BUILD_SHA256SUMS_20261007_R1.txt
export WNN_SLICE=wnn-campaign.slice
export WNN_CPU_QUOTA=100%
export WNN_THREADS=1
export WNN_MEM_MAX=1500000000
export WNN_RUNTIME_MAX=120s
exec scripts/limited .venv/bin/python \
  research/campaign_2026-10-06/bandit_continuation/throughput_optimization_E522/e520_profile_v7_candidate_v1/prepare_profile_packet.py \
  --build --repo-root . \
  --plan research/campaign_2026-10-06/bandit_continuation/throughput_optimization_E522/e520_profile_v7_candidate_v1/E520_PACKET_PLAN.json \
  --output-dir research/campaign_2026-10-06/bandit_continuation/own_baseline_E520/packet_v7 \
  --review-receipt research/campaign_2026-10-06/bandit_continuation/own_baseline_E520/E520_PROFILE_V7_CONTROLLER_ADMISSION_20261007_R1.json \
  --review-sha256 d59e66115773464be173524d07cf1b3f38c45ee3d61261ab57be5c985feeebf1
