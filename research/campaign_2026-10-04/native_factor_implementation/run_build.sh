#!/usr/bin/env bash
set -u
cd /tmp/wnn-c20261004-native-factors
export CARGO_BUILD_JOBS=1 CARGO_TARGET_DIR=/tmp/wnn-factor-cargo-v3
export RUSTFLAGS="-C target-cpu=x86-64-v3"
log=research/campaign_2026-10-04/native_factor_implementation/build_output.txt
set +e
/usr/bin/time -v /home/vlad/development/wunder/wnn_connectome_starterpack/.venv/bin/maturin \
  build --release --features python -m native/Cargo.toml \
  -o /tmp/wnn-factor-wheel/storage-fix > "$log" 2>&1
rc=$?
printf 'exit_code=%s\n' "$rc" >> "$log"
exit "$rc"
