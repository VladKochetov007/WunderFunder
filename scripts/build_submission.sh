#!/usr/bin/env bash
# Build a submission zip for a trained run: portable native wheel in the scorer base image,
# weights exported from the checkpoint, solution.py at the zip root.
#   scripts/build_submission.sh RUN [TARGET_CPU]
set -euo pipefail
run=$1
target_cpu=${2:-x86-64-v2}
root=$(cd "$(dirname "$0")/.." && pwd)
out="$root/submissions/$run"
mkdir -p "$out/pkg"
docker build -q -t wnn-native-build -f "$root/submission/Dockerfile.build" "$root/submission" >/dev/null
docker run --rm --cpus 8 --memory 8g -e RUSTFLAGS="-C target-cpu=$target_cpu" \
  -v "$root/native:/src" -v "$out:/out" wnn-native-build \
  bash -c "maturin build --release -m /src/Cargo.toml -o /out/wheel -q && chown -R $(id -u):$(id -g) /out"
rm -rf "$out/pkg"/* && mkdir -p "$out/pkg"
(cd "$out/pkg" && python3 -m zipfile -e "$(ls "$out"/wheel/wnn_native-*.whl | tail -1)" whl >/dev/null && find whl -name 'wnn_native*.so' -exec mv {} . \; && rm -rf whl)
"$root/.venv/bin/python" - "$root/artifacts/runs/$run/model.pt" "$out/pkg/weights.npz" <<'PY'
import sys, numpy as np, torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
np.savez(sys.argv[2], **{k: v.float().numpy() for k, v in ck["model"].items()})
PY
cp "$root/submission/solution.py" "$out/pkg/solution.py"
(cd "$out/pkg" && rm -f ../submission.zip && python3 -m zipfile -c ../submission.zip solution.py weights.npz wnn_native*.so)
ls -l "$out/submission.zip" "$out/pkg"
