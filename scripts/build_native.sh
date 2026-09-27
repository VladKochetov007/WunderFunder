#!/usr/bin/env bash
# Build the wnn_native extension for the scorer: x86-64-v2 and x86-64-v3 builds compiled in the
# scorer base image (submission/Dockerfile.build), extracted to OUT/v2 and OUT/v3.
#   scripts/build_native.sh OUT
# Resource knobs: WNN_BUILD_CPUS (default 2), WNN_BUILD_MEM (default 6g), WNN_BUILD_JOBS (default 2).
set -euo pipefail
out=$(realpath -m "$1")
root=$(cd "$(dirname "$0")/.." && pwd)
rm -rf "$out/v2" "$out/v3" "$out/wheel" && mkdir -p "$out"
docker build -q -t wnn-native-build -f "$root/submission/Dockerfile.build" "$root/submission" >/dev/null
for level in v2 v3; do
  # Build cache stays inside the container so the host native/target tree is never root-owned.
  docker run --rm --cpus "${WNN_BUILD_CPUS:-2}" --memory "${WNN_BUILD_MEM:-6g}" \
    -e RUSTFLAGS="-C target-cpu=x86-64-$level" -e CARGO_TARGET_DIR=/tmp/target -e CARGO_BUILD_JOBS="${WNN_BUILD_JOBS:-2}" \
    -v "$root/native:/src" -v "$out:/out" wnn-native-build \
    bash -c "maturin build --release -m /src/Cargo.toml -o /out/wheel/$level -q && chown -R $(id -u):$(id -g) /out"
  mkdir -p "$out/$level"
  (cd "$out/$level" && python3 -m zipfile -e "$(ls "$out"/wheel/$level/wnn_native-*.whl)" whl >/dev/null \
    && find whl -name 'wnn_native*.so' -exec mv {} . \; && rm -rf whl)
done
rm -rf "$out/wheel"
ls -l "$out"/v2 "$out"/v3
