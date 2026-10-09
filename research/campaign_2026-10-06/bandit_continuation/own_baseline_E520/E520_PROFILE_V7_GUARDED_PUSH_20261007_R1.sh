#!/usr/bin/env bash
set -euo pipefail
packet=research/campaign_2026-10-06/bandit_continuation/own_baseline_E520/packet_v7
(
  cd "$packet"
  sha256sum --check --status ../E520_PROFILE_V7_PACKET_SHA256SUMS_20261007_R1.txt
)
export WNN_SLICE=wnn-campaign.slice
export WNN_CPU_QUOTA=100%
export WNN_THREADS=1
export WNN_MEM_MAX=1500000000
export WNN_RUNTIME_MAX=120s
exec scripts/limited timeout --signal=TERM --kill-after=5s 100s kaggle kernels push -p "$packet" -t 1700
