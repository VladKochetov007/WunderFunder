"""Derive the predeclared within-run epoch2/3 mean, preserving parent artifacts."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path

import torch

RUNS = Path("/home/vlad/development/wunder/research_tmp/chrono_root/artifacts/runs")
OUT = Path(__file__).resolve().parent
PARENTS = {"q14s_local_kd90_s3": "q14s_swa23_kd90_s3", "q14s_local_kd100_s3": "q14s_swa23_kd100_s3"}


def digest(path: Path) -> str:
    """Hash a small checkpoint or provenance artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def derive(parent: str, name: str) -> dict:
    """Build one explicitly derived endpoint without creating a fictional fitted run."""
    original = RUNS / parent
    provenance = json.loads((original / "provenance.json").read_text())
    assert provenance["selected_epoch"] == 3
    paths = [original / f"model_ep{epoch}.pt" for epoch in (2, 3)]
    hashes = [digest(path) for path in paths]
    strict_state = json.loads((OUT.parent / f"{parent}.state_check.json").read_text())
    assert strict_state["status"] == "verified_completed_strict_state"
    assert strict_state["model_ep3_sha256"] == hashes[1]
    assert strict_state["source_sha256"]["training_dynamics.py"] == (
        "47d755817e84b17754576bbf16e34936dd7c9040716d879c7ee71797a921051c"
    )
    assert all(
        provenance["checkpoint_sha256"][path.name] == value
        for path, value in zip(paths, hashes, strict=True)
    )
    checkpoints = [torch.load(path, map_location="cpu", weights_only=False) for path in paths]
    first, last = checkpoints
    assert first["epoch"] == 2 and last["epoch"] == 3 and first["args"] == last["args"]
    assert first["model"].keys() == last["model"].keys()
    assert first["args"]["precision"] == "fp32" and not first["args"].get("eval_ema")
    assert first["args"]["name"] == parent
    assert first["readout"] is None and last["readout"] is None
    assert first["n_in"] == last["n_in"]
    averaged = {}
    for key, value in first["model"].items():
        other = last["model"][key]
        assert value.shape == other.shape and value.dtype == other.dtype == torch.float32
        assert torch.isfinite(value).all() and torch.isfinite(other).all()
        if key in ("mean", "std"):
            assert torch.equal(value, other)
            averaged[key] = value.clone()
        else:
            averaged[key] = value.mul(0.5).add(other, alpha=0.5)
    derivation = {
        "kind": "fixed_within_trajectory_parameter_mean",
        "parent_run": parent,
        "parent_provenance_sha256": digest(original / "provenance.json"),
        "parent_epochs": [2, 3],
        "parent_checkpoint_sha256": hashes,
        "weights": [0.5, 0.5],
        "new_fit_count": 0,
        "quality": "not evaluated",
        "epoch_field_semantics": "terminal parent boundary3; tensor state is mean of epochs2/3",
    }
    checkpoint = copy.deepcopy(last)
    checkpoint["model"] = averaged
    checkpoint["args"]["name"] = name
    inherited = {
        key: value
        for key, value in provenance.items()
        if key
        not in {
            "checkpoint_sha256",
            "optimizer_switch_events",
            "update_diagnostics",
            "training_clocks",
            "cumulative_updates",
            "fitting_seconds",
        }
    }
    inherited["args"] = checkpoint["args"]
    inherited["derivation"] = derivation
    checkpoint["provenance"] = copy.deepcopy(inherited)
    checkpoint["derivation"] = derivation
    folder = RUNS / name
    folder.mkdir(exist_ok=False)
    torch.save(checkpoint, folder / "model_ep3.pt")
    shutil.copyfile(folder / "model_ep3.pt", folder / "model.pt")
    inherited["checkpoint_sha256"] = {
        key: digest(folder / key) for key in ("model_ep3.pt", "model.pt")
    }
    (folder / "provenance.json").write_text(json.dumps(inherited, indent=2) + "\n")
    result = {
        "run": name,
        "parent": parent,
        "derivation": derivation,
        "checkpoint_sha256": inherited["checkpoint_sha256"],
        "provenance_sha256": digest(folder / "provenance.json"),
    }
    (OUT / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    for source, target in PARENTS.items():
        print(json.dumps(derive(source, target)))
