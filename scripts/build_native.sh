#!/usr/bin/env bash
# Build the wnn_native extension for three CPU levels inside the scorer base image.
#   scripts/build_native.sh OUT_DIR
# OUT_DIR/v2    x86-64-v2 (SSE dot kernel)
# OUT_DIR/avx2  x86-64 baseline + avx2,fma only (no BMI/MOVBE/F16C/XSAVE assumptions)
# OUT_DIR/v3    x86-64-v3
set -euo pipefail
out=$(realpath -m "$1")
root=$(cd "$(dirname "$0")/.." && pwd)
cpus=${WNN_BUILD_CPUS:-6}
mem=${WNN_BUILD_MEM:-6g}
mkdir -p "$out"
if [[ ${WNN_USE_EXISTING_BUILD_IMAGE:-0} == 1 ]]; then
  docker image inspect wnn-native-build >/dev/null
else
  docker build -q -t wnn-native-build -f "$root/submission/Dockerfile.build" "$root/submission" >/dev/null
fi
declare -A FLAGS=(
  [v2]="-C target-cpu=x86-64-v2"
  [avx2]="-C target-cpu=x86-64 -C target-feature=+avx2,+fma"
  [v3]="-C target-cpu=x86-64-v3"
)
build_levels=${WNN_BUILD_LEVELS:-"v2 avx2 v3"}
read -r -a levels <<< "$build_levels"
for level in "${levels[@]}"; do
  case "$level" in
    v2|avx2|v3) ;;
    *) echo "unsupported build level: $level" >&2; exit 2 ;;
  esac
  rm -rf "$out/wheel/$level" "$out/$level"
  docker run --rm --cpus "$cpus" --memory "$mem" --memory-swap "$mem" \
    -e RUSTFLAGS="${FLAGS[$level]}" -e CARGO_TARGET_DIR=/tmp/target \
    -v "$root/native:/src" -v "$out:/out" wnn-native-build \
    bash -c "maturin build --locked --release --features extension-module -m /src/Cargo.toml -o /out/wheel/$level -q && chown -R $(id -u):$(id -g) /out"
  mkdir -p "$out/$level"
  (cd "$out/$level" && python3 -m zipfile -e "$(ls "$out"/wheel/$level/wnn_native-*.whl)" whl >/dev/null \
    && find whl -name 'wnn_native*.so' -exec mv {} . \; && rm -rf whl)
done
python3 "$root/scripts/check_submission_abi.py" "$out"
rm -rf "$out/wheel"
ls -l "$out"/*/
