"""T4-safe strict-FP32 guard, runtime receipt, and PR49 traced trainer entrypoint."""

from __future__ import annotations

import json
import os
import runpy
import sys
from pathlib import Path

import torch

if not torch.cuda.is_available():
    raise RuntimeError("Strict-FP32 Kaggle fit requires the declared T4 GPU")
device = torch.cuda.get_device_properties(0)
capability = (int(device.major), int(device.minor))
if "T4" not in str(device.name):
    raise RuntimeError(f"Expected a Kaggle T4, got {device.name!r}")

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
rnn_backend = getattr(torch.backends.cudnn, "rnn", None)
rnn_precision = (
    getattr(rnn_backend, "fp32_precision", None) if rnn_backend is not None else None
)
if rnn_precision is not None:
    rnn_backend.fp32_precision = "ieee"
    rnn_mode = "ieee"
elif capability == (7, 5):
    # T4 has no TF32 hardware; legacy flags above are the correct PyTorch 2.6 path.
    rnn_mode = "legacy_flags_t4_no_tf32_hardware"
else:
    raise RuntimeError(
        f"No explicit cuDNN RNN IEEE control for unexpected device capability {capability}"
    )

try:
    autocast_enabled = torch.is_autocast_enabled("cuda")
except TypeError:  # older supported PyTorch signature
    autocast_enabled = torch.is_autocast_enabled()
if autocast_enabled:
    raise RuntimeError("CUDA autocast was enabled before trainer entry")

try:
    run_name = sys.argv[sys.argv.index("--name") + 1]
except (ValueError, IndexError) as error:
    raise RuntimeError("worker argv must include --name") from error
root = Path(os.environ["WNN_PROJECT_ROOT"])
receipt = {
    "run": run_name,
    "torch": str(torch.__version__),
    "device_name": str(device.name),
    "device_capability": list(capability),
    "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
    "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
    "cudnn_rnn_fp32_precision": rnn_mode,
    "autocast_context": "cuda disabled around entire trainer entrypoint",
    "cuda_autocast_enabled_at_entry": bool(autocast_enabled),
    "requested_precision": "fp32",
}
receipt_path = root / "artifacts/runtime" / f"{run_name}.json"
receipt_path.parent.mkdir(parents=True, exist_ok=True)
receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
print("STRICT_FP32_RUNTIME", json.dumps(receipt, sort_keys=True), flush=True)

traced = Path(__file__).with_name("kaggle_traced_train.py")
sys.argv = [str(traced), *sys.argv[1:]]
with torch.autocast(device_type="cuda", enabled=False):
    try:
        active = torch.is_autocast_enabled("cuda")
    except TypeError:
        active = torch.is_autocast_enabled()
    if active:
        raise RuntimeError("Explicit disabled-autocast context did not take effect")
    runpy.run_path(str(traced), run_name="__main__")
