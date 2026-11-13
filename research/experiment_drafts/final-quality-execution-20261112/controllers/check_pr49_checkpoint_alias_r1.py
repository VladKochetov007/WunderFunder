"""Compare every logical field of the two hash-bound PR49 checkpoint containers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def main() -> None:
    """Record full logical equality without running a model or reading labels."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    import numpy as np
    import torch

    torch.set_num_threads(1)
    root = Path("/data2/lc-epanov_s/tabdata/runtime/launch-20261028/tl3-trainonly-20261028-r1")
    matches = [p for p in root.rglob("model.pt") if p.parent.name == "tl3c_fbc1k_s5"]
    if len(matches) != 1:
        raise ValueError(f"Expected one legacy FBC checkpoint, got {len(matches)}")
    legacy = matches[0]
    official = Path(
        "/data/lc-epanov_s/tabdata-final-quality-20261112-r1/intake/official-teacher/model_ep3.pt"
    )
    pins = {
        legacy: "759165147d99b33034b74f4af63a30e8a202dafc78fd5e5699d05b6cfe9bef90",
        official: "840966894144bca5aafbf0a8e21ad0daab07a8503ba80e30d96668ef1bc3a496",
    }
    for path, expected in pins.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Checkpoint byte identity mismatch at {path}")

    def logical(value: object) -> object:
        if isinstance(value, torch.Tensor):
            raw = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
            return {
                "tensor": hashlib.sha256(raw).hexdigest(),
                "dtype": str(value.dtype),
                "shape": list(value.shape),
            }
        if isinstance(value, np.ndarray):
            return {
                "ndarray": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
                "dtype": str(value.dtype),
                "shape": list(value.shape),
            }
        if isinstance(value, dict):
            return {
                str(key): logical(item)
                for key, item in sorted(value.items(), key=lambda item: str(item[0]))
            }
        if isinstance(value, (tuple, list)):
            return [logical(item) for item in value]
        if isinstance(value, argparse.Namespace):
            return {"namespace": logical(vars(value))}
        if isinstance(value, np.generic):
            return logical(value.item())
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        raise TypeError(f"Unsupported logical checkpoint field: {type(value).__name__}")

    values = [logical(torch.load(path, map_location="cpu", weights_only=False)) for path in pins]
    canonical = [
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        for value in values
    ]
    equal = canonical[0] == canonical[1]
    receipt = {
        "schema": "wnn.pr49-full-checkpoint-alias.r1",
        "logical_checkpoint_equal": equal,
        "all_checkpoint_fields_compared": True,
        "model_inference_executed": False,
        "labels_read": False,
        "checkpoints": [
            {"path": str(path), "sha256": pin, "logical_sha256": hashlib.sha256(body).hexdigest()}
            for (path, pin), body in zip(pins.items(), canonical, strict=True)
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps(receipt))
    if not equal:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
