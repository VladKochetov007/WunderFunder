#!/usr/bin/env bash
# Build wnn_native for local benchmarking without touching the shared .venv.
#   research/campaign_2026-10-01/engine/build.sh TAG [LEVEL ...]   (levels: v2 avx2 v3; default v3)
# Output: engine/lib/TAG/LEVEL/wnn_native.abi3.so (import with sys.path.insert(0, that dir)).
# Host toolchain build (glibc of this host); packages still come from scripts/build_native.sh.
set -euo pipefail
root=$(cd "$(dirname "$0")/../../.." && pwd)
eng="$root/research/campaign_2026-10-01/engine"
tag=$1; shift
levels=("${@:-v3}")
declare -A FLAGS=(
  [v2]="-C target-cpu=x86-64-v2"
  [avx2]="-C target-cpu=x86-64 -C target-feature=+avx2,+fma"
  [v3]="-C target-cpu=x86-64-v3"
)
for level in "${levels[@]}"; do
  whl="$eng/wheels/$tag/$level"
  rm -rf "$whl" "$eng/lib/$tag/$level"
  WNN_CPU_PCT=13 WNN_MEM_MAX=4G "$root/scripts/limited" env CARGO_TARGET_DIR="$eng/target/$level" CARGO_BUILD_JOBS=${JOBS:-2} \
    RUSTFLAGS="${FLAGS[$level]}" "$root/.venv/bin/maturin" build --release -q -m "$root/native/Cargo.toml" \
    -i "$eng/.venv/bin/python" -o "$whl" >/dev/null
  mkdir -p "$eng/lib/$tag/$level"
  (cd "$eng/lib/$tag/$level" && "$eng/.venv/bin/python" -m zipfile -e "$(ls "$whl"/wnn_native-*.whl)" whl \
    && find whl -name 'wnn_native*.so' -exec mv {} . \; && rm -rf whl)
  ls -l "$eng/lib/$tag/$level"
done
