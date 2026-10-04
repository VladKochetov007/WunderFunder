#!/usr/bin/env bash
set -u
cd /tmp/wnn-c20261004-native-factors
level=${1:?pass v2 or v3}
if [[ "$level" != v2 && "$level" != v3 ]]; then
  echo "level must be v2 or v3" >&2
  exit 2
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export WNN_FACTOR_TEST_LIBRARY="/tmp/wnn-factorized-e420/$level/wnn_native"
export WNN_Q8_TEST_LIBRARY="/tmp/wnn-factorized-e420/$level"
export WNN_FACTOR_LIBRARY_ROOT=/tmp/wnn-factorized-e420
log="research/campaign_2026-10-04/E420_factorized/${level}_test_output.txt"
set +e
/usr/bin/time -v /home/vlad/development/wunder/wnn_connectome_starterpack/.venv/bin/python \
  -m pytest -q tests/test_probe_factorized_native.py tests/test_native_factorized.py \
  tests/test_native_quantized_export.py > "$log" 2>&1
rc=$?
printf 'exit_code=%s\n' "$rc" >> "$log"
exit "$rc"
