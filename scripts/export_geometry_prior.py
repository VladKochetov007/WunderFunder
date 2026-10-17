"""Export the fitting-only geometry prior from one exact checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

_SCHEMA = "causal_geometry_discrete_v1"
_COLUMNS = [0, 3, 52, 55]


def sha256_file(path: Path) -> str:
    """Hash a checkpoint with bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_value(value: Any) -> Any:
    """Convert checkpoint tensors and NumPy values to JSON-compatible values."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


def main() -> None:
    """Validate and write a checkpoint-bound `geometry_prior.json`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-reference", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    checkpoint_path = args.checkpoint.resolve(strict=True)
    output_path = args.output.resolve()
    receipt_path = args.receipt.resolve()
    if output_path.exists() or receipt_path.exists():
        raise FileExistsError("refusing to replace geometry prior or export receipt")

    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "research_feature_prior" not in checkpoint:
        raise ValueError("checkpoint does not contain research_feature_prior")
    prior = json_value(checkpoint["research_feature_prior"])
    if not isinstance(prior, dict) or prior.get("schema") != _SCHEMA:
        raise ValueError("checkpoint prior has the wrong geometry schema")
    if prior.get("selected_raw_columns") != _COLUMNS:
        raise ValueError("checkpoint prior has unexpected selected raw columns")
    if len(prior.get("feature_order", [])) != 60:
        raise ValueError("checkpoint geometry feature order must contain 60 channels")
    arrays = {
        name: np.asarray(prior.get(name), dtype=np.float32)
        for name in ("beta", "intercept", "atoms", "vocabulary")
    }
    if tuple(value.shape for value in arrays.values()) != ((4,), (4,), (4,), (4, 3)):
        raise ValueError("checkpoint geometry prior arrays have unexpected shapes")
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("checkpoint geometry prior contains non-finite values")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(prior, indent=2, allow_nan=False) + "\n"
    with output_path.open("x", encoding="utf-8") as output:
        output.write(payload)
    receipt = {
        "schema": "wnn.geometry-prior-export.v1",
        "run_reference": args.run_reference,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "output_path": str(output_path),
        "output_sha256": sha256_file(output_path),
        "prior_schema": prior["schema"],
        "selected_raw_columns": prior["selected_raw_columns"],
        "feature_count": len(prior["feature_order"]),
        "scope": "feature prior copied from the selected child checkpoint; no refit or data scan",
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with receipt_path.open("x", encoding="utf-8") as output:
        output.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
