#!/usr/bin/env bash
set -euo pipefail

if (( $# != 3 )); then
  echo "usage: $0 PAIR_INDEX_A PAIR_INDEX_B PROFILE_RECEIPT" >&2
  exit 2
fi
pair_a="$1"
pair_b="$2"
profile_receipt="$3"
if (( pair_a < 0 || pair_a > 28 || pair_b != pair_a + 1 || pair_a % 2 != 0 )); then
  echo "use an even pair index from 0..28 followed by the next index" >&2
  exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
WNN_B300_SCRATCH="${WNN_B300_SCRATCH:?set WNN_B300_SCRATCH to a physical-disk mount}"
WNN_PORTABLE_INPUT_DIR="${WNN_PORTABLE_INPUT_DIR:-/mnt/geometry-inputs}"
if [[ "$profile_receipt" != /* ]]; then
  profile_receipt="$PROJECT_ROOT/$profile_receipt"
fi
if [[ ! -f "$profile_receipt" ]]; then
  echo "profile receipt not found: $profile_receipt" >&2
  exit 2
fi

cd "$PROJECT_ROOT"
mkdir -p "$PROJECT_ROOT/artifacts/geometry_b300/logs"
declare -a pids=()
arms=(baseline moments dictionary combined)
for gpu in 0 1 2 3 4 5 6 7; do
  if (( gpu < 4 )); then
    pair_index="$pair_a"
  else
    pair_index="$pair_b"
  fi
  arm="${arms[$((gpu % 4))]}"
  CUDA_VISIBLE_DEVICES="$gpu" WNN_GPU_ID="$gpu" \
    WNN_PROJECT_ROOT="$PROJECT_ROOT" \
    WNN_B300_SCRATCH="$WNN_B300_SCRATCH" \
    WNN_PORTABLE_INPUT_DIR="$WNN_PORTABLE_INPUT_DIR" \
    "$PYTHON_BIN" -u scripts/geometry_b300/run_cell.py \
    --pair-index "$pair_index" --arm "$arm" \
    --skip-completed \
    --profile-receipt "$profile_receipt" \
    >"$PROJECT_ROOT/artifacts/geometry_b300/logs/wave_${pair_a}_${pair_b}_gpu${gpu}.stdout.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"
