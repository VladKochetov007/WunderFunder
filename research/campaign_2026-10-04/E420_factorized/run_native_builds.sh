#!/usr/bin/env bash
set -u
cd /tmp/wnn-c20261004-native-factors
export CARGO_BUILD_JOBS=1
log=research/campaign_2026-10-04/E420_factorized/native_build_output.txt
set +e
/usr/bin/time -v /bin/bash scripts/build_factorized_native.sh /tmp/wnn-factorized-e420 > "$log" 2>&1
rc=$?
printf 'exit_code=%s\n' "$rc" >> "$log"
exit "$rc"
