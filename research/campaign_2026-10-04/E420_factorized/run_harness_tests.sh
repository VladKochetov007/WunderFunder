#!/usr/bin/env bash
set -u
cd /tmp/wnn-c20261004-native-factors
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
log=research/campaign_2026-10-04/E420_factorized/harness_test_output.txt
set +e
/usr/bin/time -v /home/vlad/development/wunder/wnn_connectome_starterpack/.venv/bin/python \
  -m pytest -q tests/test_probe_factorized_native.py > "$log" 2>&1
rc=$?
printf 'exit_code=%s\n' "$rc" >> "$log"
exit "$rc"
