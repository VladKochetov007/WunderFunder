#!/usr/bin/env bash
set -euo pipefail
repo=/home/vlad/development/wunder/wnn_connectome_starterpack
own="$repo/research/campaign_2026-09-30/integration"
cd "$repo"
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export WNN_SLICE=wnn-campaign.slice WNN_SCOPE_UNIT=wnn-integration-amended-v2-scope
export WNN_THREADS=1 WNN_CPU_QUOTA=100% WNN_MEM_MAX=2G WNN_RUNTIME_MAX=300
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ARROW_NUM_THREADS=1 POLARS_MAX_THREADS=1
date --iso-8601=ns > "$own/start_v2.txt"
status=0
scripts/limited /bin/bash "$own/proof_sequence.sh" > "$own/amended_v2.log" 2>&1 || status=$?
printf '%s\n' "$status" > "$own/exit_v2.txt"
date --iso-8601=ns > "$own/end_v2.txt"
exit "$status"
