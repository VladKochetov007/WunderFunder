#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
actor_dir="$repo_root/research/campaign_2026-09-30/structured_state"
label=${1:-manual}
if [[ ! "$label" =~ ^[a-zA-Z0-9_-]+$ ]]; then
    echo 'Use an alphanumeric label with underscores or hyphens.' >&2
    exit 2
fi
if [[ -e "$actor_dir/smoke_results_${label}.json" ]]; then
    echo 'This label already has results. Use a fresh label to preserve evidence.' >&2
    exit 2
fi
cd "$repo_root"
export CUDA_VISIBLE_DEVICES=''
export WNN_SLICE=wnn-campaign.slice
export WNN_SCOPE_UNIT="wnn-structured-state-smoke-${label}"
export WNN_THREADS=1 WNN_CPU_QUOTA=100% WNN_MEM_MAX=2G WNN_RUNTIME_MAX=300
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
command=(scripts/limited .venv/bin/python "$actor_dir/smoke.py"
    --output "$actor_dir/smoke_results_${label}.json")
printf '%q ' "${command[@]}" > "$actor_dir/command_${label}.txt"
printf '\n' >> "$actor_dir/command_${label}.txt"
if "${command[@]}" > "$actor_dir/smoke_${label}.log" 2>&1; then
    status=0
else
    status=$?
fi
printf '%s\n' "$status" > "$actor_dir/exit_${label}.txt"
exit "$status"
