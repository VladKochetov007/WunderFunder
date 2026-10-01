#!/usr/bin/env bash
# Calibration zips before and after the profile, all pinned to one core (cycle counts are the
# frequency-independent measure; see scripts/engine_fp16_check.py).
set -euo pipefail
cd /home/vlad/development/wunder/wnn_connectome_starterpack
for z in final_since2_gclp final_mixed_ema6_shared; do taskset -c 15 research/campaign_2026-10-01/engine/.venv/bin/python scripts/engine_fp16_check.py calib submissions/$z/$z.zip --seqs 3; done
taskset -c 15 research/campaign_2026-10-01/engine/.venv/bin/python scripts/engine_fp16_check.py profile --lib $1 --seqs 1 --repeats 2 --out $2 "${@:3}"
for z in final_since2_gclp final_mixed_ema6_shared; do taskset -c 15 research/campaign_2026-10-01/engine/.venv/bin/python scripts/engine_fp16_check.py calib submissions/$z/$z.zip --seqs 3; done
