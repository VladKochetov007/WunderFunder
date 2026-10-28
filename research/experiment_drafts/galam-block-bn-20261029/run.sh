#!/usr/bin/env bash
# Explicit receiver actions only. No queue, live-job discovery or automatic fitting.
set -euo pipefail
ACTION="${1:?Expected check, profile, or bn}"
shift
: "${PYTHON_EXECUTABLE:?Set the absolute path to the provisioned receiver Python}"
: "${CUDA_VISIBLE_DEVICES:?Select one free GPU explicitly}"
if [[ "$CUDA_VISIBLE_DEVICES" == *,* || -z "$CUDA_VISIBLE_DEVICES" ]]; then
  echo 'This packet uses exactly one explicitly selected GPU.' >&2
  exit 2
fi
PACKET_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(git -C "$PACKET_DIR" rev-parse --show-toplevel)"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUBLAS_EMULATE_SINGLE_PRECISION=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
case "$ACTION" in
  check) exec "$PYTHON_EXECUTABLE" "$PACKET_DIR/check_block.py" "$@" ;;
  profile) exec "$PYTHON_EXECUTABLE" "$PACKET_DIR/profile_block.py" "$@" ;;
  bn) exec "$PYTHON_EXECUTABLE" scripts/galam_bn_diagnostic.py "$@" ;;
  *) echo "Unknown action: $ACTION" >&2; exit 2 ;;
esac
