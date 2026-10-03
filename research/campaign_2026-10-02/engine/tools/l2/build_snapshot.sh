#!/usr/bin/env bash
# scripts/build_native.sh from a snapshot of native/ (so edits during the build cannot leak in).
#   build_snapshot.sh OUT_DIR   -> OUT_DIR/{v2,avx2,v3}, OUT_DIR/src (the snapshot built)
set -euo pipefail
root=$(cd "$(dirname "$0")/../../../../.." && pwd)
out=$(realpath -m "$1")
mkdir -p "$out"
rm -rf "$out/src"
mkdir -p "$out/src"
cp -r "$root/native/Cargo.toml" "$root/native/Cargo.lock" "$root/native/src" "$root/native/examples" "$out/src/" 2>/dev/null || true
cp "$root/native/pyproject.toml" "$out/src/" 2>/dev/null || true
docker build -q -t wnn-native-build -f "$root/submission/Dockerfile.build" "$root/submission" >/dev/null
declare -A FLAGS=(
  [v2]="-C target-cpu=x86-64-v2"
  [avx2]="-C target-cpu=x86-64 -C target-feature=+avx2,+fma"
  [v3]="-C target-cpu=x86-64-v3"
)
for level in v2 avx2 v3; do
  rm -rf "$out/wheel/$level" "$out/$level"
  docker run --rm --cpus 3 --memory 5g -e RUSTFLAGS="${FLAGS[$level]}" -e CARGO_TARGET_DIR=/tmp/target \
    -v "$out/src:/src" -v "$out:/out" wnn-native-build \
    bash -c "maturin build --release -m /src/Cargo.toml -o /out/wheel/$level -q && chown -R $(id -u):$(id -g) /out"
  mkdir -p "$out/$level"
  (cd "$out/$level" && python3 -m zipfile -e "$(ls "$out"/wheel/$level/wnn_native-*.whl)" whl >/dev/null \
    && find whl -name 'wnn_native*.so' -exec mv {} . \; && rm -rf whl)
done
rm -rf "$out/wheel"
ls -l "$out"/*/*.so
