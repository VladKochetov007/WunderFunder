#!/usr/bin/env bash
# calib.sh ZIP [ZIP...]: engine_fp16_check calib on core 15 (5G cap), one JSON line per zip.
cd "$(dirname "$0")/../../../../.."
for z in "$@"; do
  WNN_SLICE=wnn-campaign.slice WNN_CPU_PCT=19 WNN_MEM_MAX=5G scripts/limited taskset -c 15 \
    research/campaign_2026-10-01/engine/.venv/bin/python scripts/engine_fp16_check.py calib "$z" --seqs 3 2>/dev/null | tail -1
done
