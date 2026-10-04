#!/usr/bin/env bash
set -u
cd /tmp/wnn-c20261004-native-factors
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export WNN_FACTOR_TEST_LIBRARY=/tmp/wnn-factor-build/storage-fix/wnn_native
export WNN_Q8_TEST_LIBRARY=/tmp/wnn-factor-build/storage-fix
log=research/campaign_2026-10-04/native_factor_implementation/test_output.txt
set +e
/usr/bin/time -v /home/vlad/development/wunder/wnn_connectome_starterpack/.venv/bin/python \
  -m pytest -q tests/test_native_factorized.py tests/test_native_quantized_export.py > "$log" 2>&1
rc=$?
printf 'exit_code=%s\n' "$rc" >> "$log"
exit "$rc"
