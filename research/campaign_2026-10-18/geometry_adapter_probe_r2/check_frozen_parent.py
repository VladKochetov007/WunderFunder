"""Prove the adapter changed only its explicitly appended input columns."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
PACKET = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    """Hash one checkpoint without loading its bytes together."""
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def main() -> None:
    """Fail rather than certify unexpected backbone or normalization drift."""
    parent_path = ROOT / "artifacts/runs/warm_mechanism_f0_s3_c_r3/model_ep1.pt"
    child_path = ROOT / "artifacts/runs/geometry_adapter_f0_s3_r2/model_ep0.pt"
    parent = torch.load(parent_path, map_location="cpu", weights_only=False)
    child = torch.load(child_path, map_location="cpu", weights_only=False)
    old, new = parent["model"], child["model"]
    assert old.keys() == new.keys(), "state key mismatch"
    checked = []
    for key, value in old.items():
        current = new[key]
        if key == "inp.weight":
            assert value.shape[1] == 211 and current.shape[1] == 271
            assert torch.equal(value, current[:, :211]), key
            suffix = current[:, 211:]
            assert bool(torch.isfinite(suffix).all()) and bool(torch.count_nonzero(suffix))
        elif value.shape != current.shape:
            assert value.numel() == 211 and current.numel() == 271, key
            assert torch.equal(value.flatten(), current.flatten()[:211]), key
            suffix = current.flatten()[211:]
            assert "mean" in key or "std" in key, key
            expected = 0 if "mean" in key else 1
            assert bool((suffix == expected).all()), key
        else:
            assert torch.equal(value, current), key
        checked.append(key)
    result = {
        "pass": True,
        "parent_sha256": digest(parent_path),
        "child_sha256": digest(child_path),
        "checked_state_keys": checked,
        "old_input_columns_bitwise_equal": True,
        "all_other_parent_parameters_bitwise_equal": True,
        "new_input_columns": [211, 271],
        "new_weight_max_abs": float(new["inp.weight"][:, 211:].abs().max()),
        "new_weight_nonzero_count": int(torch.count_nonzero(new["inp.weight"][:, 211:])),
        "limitations": "State preservation proof; does not establish causal usefulness or score gain.",
    }
    (PACKET / "frozen_parent_check.json").write_text(json.dumps(result, indent=2) + "\n")
    prior = child["research_feature_prior"]
    (PACKET / "geometry_prior.json").write_text(json.dumps(prior, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
