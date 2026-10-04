#!/usr/bin/env bash
set -euo pipefail

out=${1:?output directory required}
if [[ -e "$out" ]]; then
  echo "refusing to overwrite $out" >&2
  exit 2
fi
mkdir -p "$out"
maturin=/home/vlad/development/wunder/wnn_connectome_starterpack/.venv/bin/maturin
python=/home/vlad/development/wunder/wnn_connectome_starterpack/.venv/bin/python

for level in v2 v3; do
  case "$level" in
    v2) flags="-C target-cpu=x86-64-v2" ;;
    v3) flags="-C target-cpu=x86-64-v3" ;;
  esac
  wheel_dir="$out/wheel-$level"
  mkdir -p "$wheel_dir"
  if [[ "$level" == v2 ]]; then
    target=/tmp/wnn-q8-build/v2
  else
    target=/tmp/wnn-factor-cargo-v3
  fi
  CARGO_BUILD_JOBS=1 CARGO_TARGET_DIR="$target" RUSTFLAGS="$flags" \
    "$maturin" build --release --features extension-module -m native/Cargo.toml -o "$wheel_dir"
  WNN_WHEEL_DIR="$wheel_dir" WNN_EXTRACT_DIR="$out/$level" "$python" - <<'PY'
import os
from pathlib import Path
from zipfile import ZipFile

wheel_dir = Path(os.environ["WNN_WHEEL_DIR"])
extract_dir = Path(os.environ["WNN_EXTRACT_DIR"])
wheels = list(wheel_dir.glob("wnn_native-*.whl"))
if len(wheels) != 1:
    raise SystemExit(f"expected one wheel in {wheel_dir}, found {len(wheels)}")
extract_dir.mkdir()
with ZipFile(wheels[0]) as archive:
    archive.extractall(extract_dir)
PY
done
"$python" scripts/check_submission_abi.py "$out"
