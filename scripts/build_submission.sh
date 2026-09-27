#!/usr/bin/env bash
# Build a submission zip for a trained run: x86-64-v2 and x86-64-v3 builds of the native
# extension (scripts/build_native.sh), weights from the checkpoint, solution.py.
#   scripts/build_submission.sh RUN
set -euo pipefail
run=$1
root=$(cd "$(dirname "$0")/.." && pwd)
out="$root/submissions/$run"
rm -rf "$out/pkg" "$out/wheel" && mkdir -p "$out/pkg"
"$root/scripts/build_native.sh" "$out/pkg"
"$root/.venv/bin/python" - "$root/artifacts/runs/$run/model.pt" "$out/pkg/weights.npz" <<'PY'
import sys, numpy as np, torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
if ck["args"].get("online_norm"):
    sys.exit("online-norm runs need members.json: package them with scripts/package_submission.py")
np.savez(sys.argv[2], **{k: v.float().numpy() for k, v in ck["model"].items()})
PY
cp "$root/submission/solution.py" "$out/pkg/solution.py"
(cd "$out/pkg" && rm -f ../submission.zip && python3 -m zipfile -c ../submission.zip solution.py weights.npz v2 v3)
ls -l "$out/submission.zip" && (cd "$out/pkg" && find . -type f | sort)
