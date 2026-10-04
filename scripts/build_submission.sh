#!/usr/bin/env bash
# Build a submission zip for a trained run: x86-64-v2 and x86-64-v3 builds of the native
# extension (compiled in the scorer base image), weights from the checkpoint, solution.py.
#   scripts/build_submission.sh RUN
set -euo pipefail
run=$1
root=$(cd "$(dirname "$0")/.." && pwd)
out="$root/submissions/$run"
rm -rf "$out/pkg" "$out/wheel" && mkdir -p "$out/pkg"
docker build -q -t wnn-native-build -f "$root/submission/Dockerfile.build" "$root/submission" >/dev/null
for level in v2 v3; do
  # Build cache stays inside the container so the host native/target tree is never root-owned.
  docker run --rm --cpus 8 --memory 8g -e RUSTFLAGS="-C target-cpu=x86-64-$level" -e CARGO_TARGET_DIR=/tmp/target \
    -v "$root/native:/src" -v "$out:/out" wnn-native-build \
    bash -c "maturin build --release --features extension-module -m /src/Cargo.toml -o /out/wheel/$level -q && chown -R $(id -u):$(id -g) /out"
  mkdir -p "$out/pkg/$level"
  (cd "$out/pkg/$level" && python3 -m zipfile -e "$(ls "$out"/wheel/$level/wnn_native-*.whl)" whl >/dev/null \
    && find whl -name 'wnn_native*.so' -exec mv {} . \; && rm -rf whl)
done
"$root/.venv/bin/python" - "$root/artifacts/runs/$run/model.pt" "$out/pkg/weights.npz" <<'PY'
import sys, numpy as np, torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
np.savez(sys.argv[2], **{k: v.float().numpy() for k, v in ck["model"].items()})
PY
cp "$root/submission/solution.py" "$out/pkg/solution.py"
(cd "$out/pkg" && rm -f ../submission.zip && python3 -m zipfile -c ../submission.zip solution.py weights.npz v2 v3)
python3 "$root/scripts/check_submission_abi.py" "$out/submission.zip"
ls -l "$out/submission.zip" && (cd "$out/pkg" && find . -type f | sort)
