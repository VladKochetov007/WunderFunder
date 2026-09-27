#!/usr/bin/env bash
# Build native/ at x86-64-v2 and x86-64-v3 in the scorer base image (glibc of Debian bookworm):
# writes OUT/v2/wnn_native.abi3.so and OUT/v3/wnn_native.abi3.so.
#   WNN_BUILD_CPUS=2 WNN_BUILD_MEM=6g scripts/build_native.sh OUT     (defaults: 8 cpus, 8g)
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
out=$(realpath -m "$1")
cpus=${WNN_BUILD_CPUS:-8}
mem=${WNN_BUILD_MEM:-8g}
rm -rf "$out/wheel" "$out/v2" "$out/v3" && mkdir -p "$out"
docker build -q -t wnn-native-build -f "$root/submission/Dockerfile.build" "$root/submission" >/dev/null
for level in v2 v3; do
  # Build cache stays inside the container so the host native/target tree is never root-owned.
  docker run --rm --cpus "$cpus" --memory "$mem" -e CARGO_BUILD_JOBS="$cpus" \
    -e RUSTFLAGS="-C target-cpu=x86-64-$level" -e CARGO_TARGET_DIR=/tmp/target \
    -v "$root/native:/src" -v "$out:/out" wnn-native-build \
    bash -c "maturin build --release -m /src/Cargo.toml -o /out/wheel/$level -q && chown -R $(id -u):$(id -g) /out"
  mkdir -p "$out/$level"
  (cd "$out/$level" && python3 -m zipfile -e "$(ls "$out"/wheel/$level/wnn_native-*.whl)" whl >/dev/null \
    && find whl -name 'wnn_native*.so' -exec mv {} . \; && rm -rf whl)
done
rm -rf "$out/wheel"
