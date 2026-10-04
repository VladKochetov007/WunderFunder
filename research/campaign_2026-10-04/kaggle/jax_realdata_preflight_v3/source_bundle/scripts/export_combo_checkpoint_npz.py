"""Export the exact fold-0 combo model weights to the staged Torch-name NPZ."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from wnn.nn import RNNModel

EXPECTED_CHECKPOINT_SHA256 = "d4a1e50b1892b4d9edd7c255908b68073a760c059ca80223e98a63c7e023371e"


def sha256_file(path: Path) -> str:
    """Hash a file in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def export(checkpoint_path: Path, output_path: Path) -> dict:
    """Validate/export a checkpoint and bind NPZ bytes to its source hash."""
    sidecar = output_path.with_suffix(output_path.suffix + ".json")
    if output_path.exists() or sidecar.exists():
        raise FileExistsError(f"refusing to overwrite existing export: {output_path}")
    source_sha256 = sha256_file(checkpoint_path)
    if source_sha256 != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(f"combo checkpoint hash changed: {source_sha256}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if sha256_file(checkpoint_path) != source_sha256:
        raise RuntimeError("source checkpoint changed during CPU load")
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
        raise ValueError("checkpoint does not contain a model state dict")

    model = RNNModel(n_in=211, hidden=256, layers=2, dropout=0.0, n_out=23)
    state = checkpoint["model"]
    expected_keys = set(model.state_dict())
    if set(state) != expected_keys:
        raise ValueError(
            "checkpoint tensor names differ: "
            f"missing={sorted(expected_keys - set(state))}, "
            f"extra={sorted(set(state) - expected_keys)}"
        )
    for name, tensor in state.items():
        if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32:
            raise ValueError(f"{name} must be an FP32 Torch tensor")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} contains non-finite values")
    model.load_state_dict(state, strict=True)
    if not torch.all(model.std > 0):
        raise ValueError("checkpoint input normalization std must be positive")
    arrays = {
        name: tensor.detach().cpu().numpy().astype(np.float32, copy=True)
        for name, tensor in model.state_dict().items()
    }
    if sha256_file(checkpoint_path) != source_sha256:
        raise RuntimeError("source checkpoint changed during validation")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **arrays)
    report = {
        "format": "wnn.jax-npz-export",
        "version": 1,
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": source_sha256,
        "source_epoch": 3,
        "tensor_layout": "Torch RNNModel names, exact validated shapes, float32",
        "dimensions": {"n_in": 211, "hidden": 256, "layers": 2, "n_out": 23},
        "npz_path": str(output_path),
        "npz_sha256": sha256_file(output_path),
    }
    sidecar.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    """Parse exact source and output paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/runs/c20261003_combo_s3_f0/model_ep3.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/runs/c20261003_combo_s3_f0/model_ep3.jax.npz"),
    )
    args = parser.parse_args()
    report = export(args.checkpoint, args.output)
    print(json.dumps({"npz_path": report["npz_path"], "npz_sha256": report["npz_sha256"]}))


if __name__ == "__main__":
    main()
