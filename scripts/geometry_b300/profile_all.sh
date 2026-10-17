#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
WNN_B300_SCRATCH="${WNN_B300_SCRATCH:?set WNN_B300_SCRATCH to a physical-disk mount}"
WNN_PORTABLE_INPUT_DIR="${WNN_PORTABLE_INPUT_DIR:-/mnt/geometry-inputs}"
PROFILE_DIR="$PROJECT_ROOT/artifacts/geometry_b300/profiles"

cd "$PROJECT_ROOT"
mkdir -p "$PROFILE_DIR"
declare -a pids=()
for gpu in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES="$gpu" WNN_GPU_ID="$gpu" \
    WNN_PROJECT_ROOT="$PROJECT_ROOT" \
    WNN_B300_SCRATCH="$WNN_B300_SCRATCH" \
    WNN_PORTABLE_INPUT_DIR="$WNN_PORTABLE_INPUT_DIR" \
    "$PYTHON_BIN" -u scripts/geometry_b300/run_cell.py \
    --pair-index 0 --profile-only --profile-steps 8 --profile-warmup 2 \
    >"$PROFILE_DIR/profile_gpu${gpu}.stdout.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if (( status != 0 )); then
  echo "At least one card profile failed; preserve logs and diagnose before retrying." >&2
  exit "$status"
fi

args=()
for gpu in 0 1 2 3 4 5 6 7; do
  args+=(--profile "$PROFILE_DIR/b300_profile_${gpu}.json")
done
"$PYTHON_BIN" scripts/geometry_b300/make_profile_receipt.py \
  "${args[@]}" \
  --output "$PROFILE_DIR/profile_receipt.json"
