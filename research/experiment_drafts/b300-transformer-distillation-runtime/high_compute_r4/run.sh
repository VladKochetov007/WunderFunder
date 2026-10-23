#!/usr/bin/env bash
set -euo pipefail

packet=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd -- "$packet/../../../.."
[[ $(uname -s) == Linux ]] || { echo 'Receiving Linux B300 host only' >&2; exit 2; }
: "${WNN_EXPECTED_COMMIT:?pin the reviewed full PR head SHA}"
: "${CUDA_VISIBLE_DEVICES:?bind the exact eight allocated GPU IDs or UUIDs}"
[[ $(git rev-parse HEAD) == "$WNN_EXPECTED_COMMIT" ]] || { echo 'Source commit mismatch' >&2; exit 2; }
[[ -z $(git status --porcelain -- src scripts tests "$packet/.." pyproject.toml uv.lock) ]] || {
  echo 'Runtime, tests, packet or dependencies are dirty' >&2; exit 2;
}
IFS=, read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
[[ ${#devices[@]} == 8 ]] || { echo 'Exactly eight allocated GPUs required' >&2; exit 2; }
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
uv run --no-sync python "$packet/dispatch.py" "$@"
