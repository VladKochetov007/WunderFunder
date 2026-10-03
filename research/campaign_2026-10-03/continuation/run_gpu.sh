#!/usr/bin/env bash
# Used inside campaign_job's serialized and monitored GPU service.
set -euo pipefail
cd "$(dirname "$0")/../../.."
state=research/campaign_2026-10-03/continuation
[[ -f $state/READY && ! -f $state/RESOURCE_FAILURE ]] || exit 75
export WNN_CPU_QUOTA=800% WNN_THREADS=8 WNN_MEM_MAX=24G
export WNN_IN_CAMPAIGN_JOB=1
export WNN_VRAM_GB=4.8 WNN_GPU_FRACTION=0.70
export WNN_LOADER_WORKERS=3 WNN_LOADER_RAM_GB=3
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export NUMBA_NUM_THREADS=8 POLARS_MAX_THREADS=8 ARROW_NUM_THREADS=1
exec scripts/limited "$@"
