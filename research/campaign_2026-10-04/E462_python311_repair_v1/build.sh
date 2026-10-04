#!/usr/bin/env bash
# Packaging repair only: frozen numerical source, explicit extension linking.
set -euo pipefail
root=$(cd "$(dirname "$0")/../../.." && pwd)
out="$root/research/campaign_2026-10-04/E462_python311_repair_v1"
test ! -d "$out/native_source"
cp -a "$root/research/campaign_2026-10-04/E416_native_q8_build/native" "$out/native_source"
for level in v2 v3; do
  mkdir -p "$out/wheels/$level"
  docker run --rm --cpus 2 --memory 2g --memory-swap 2g \
    -e CARGO_BUILD_JOBS=1 -e CARGO_HOME=/out/cargo \
    -e CARGO_TARGET_DIR="/out/target-$level" \
    -e PYO3_PYTHON=/usr/local/bin/python3.11 \
    -e RUSTFLAGS="-C target-cpu=x86-64-$level" \
    -v "$out/native_source:/src:ro" -v "$out:/out" wnn-native-build \
    bash -c "maturin build --locked --release --features python,pyo3/extension-module -m /src/Cargo.toml -o /out/wheels/$level && chown -R $(id -u):$(id -g) /out" \
    > "$out/build-$level.log" 2>&1
done
docker build --cpuset-cpus 14-15 --memory 2g --memory-swap 2g \
  -t wnn-python311-submission-check \
  -f "$out/Dockerfile.check" "$out" > "$out/build-check-image.log" 2>&1
