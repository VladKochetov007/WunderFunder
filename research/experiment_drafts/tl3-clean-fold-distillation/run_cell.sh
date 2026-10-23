#!/usr/bin/env bash
set -euo pipefail
# Run explicitly on one assigned B300 after binding assets and remote acceptance checks.
: "${CUDA_VISIBLE_DEVICES:?Assign exactly one B300 before running this cell}"
: "${WNN_TL3_ASSETS:?Absolute path to the sealed receiver assets JSON}"
cell="${1:?Usage: bash run_cell.sh F0 or F2}"
case "$cell" in F0|F2) ;; *) exit 2 ;; esac
packet="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for stage in preflight profile-teachers fit-teachers cache profile-students fit-students predict; do
  python "$packet/receiver.py" "$stage" --assets "$WNN_TL3_ASSETS" --cell "$cell"
done
