from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

import torch

# Checkpoints contain heterogeneous tensor, argument and provenance dictionaries.
Checkpoint = dict[str, object]
RUNTIME_ARGUMENTS = {"name", "resume_state", "stop_after_epoch"}
FIXED_BUFFERS = {"mean", "std"}


def sha256(path: Path) -> str:
    """Hash an immutable source without retaining its serialized payload."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def mean_state(states: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Average floating parameters while requiring identical normalization buffers."""
    if len(states) != 3:
        raise ValueError("this recipe requires exactly three trajectory checkpoints")
    if any(state.keys() != states[0].keys() for state in states):
        raise ValueError("checkpoint tensor keys differ")
    result = {}
    for name, reference in states[0].items():
        values = [state[name] for state in states]
        if any(v.shape != reference.shape or v.dtype != reference.dtype for v in values):
            raise ValueError(f"tensor shape or dtype differs: {name}")
        if any(not torch.isfinite(v).all() for v in values):
            raise ValueError(f"nonfinite tensor: {name}")
        if name in FIXED_BUFFERS or not reference.is_floating_point():
            if any(not torch.equal(v, reference) for v in values):
                raise ValueError(f"fixed buffer differs: {name}")
            result[name] = reference.clone()
        else:
            total = sum(
                (v.to(torch.float64) for v in values),
                torch.zeros_like(reference, dtype=torch.float64),
            )
            result[name] = (total / len(values)).to(reference.dtype)
    return result


def load_bound(root: Path, item: dict[str, object]) -> Checkpoint:
    """Read one small, hash-bound own checkpoint using restricted unpickling."""
    path = (root / str(item["path"])).resolve(strict=True)
    if not path.is_relative_to(root) or path.stat().st_size > 32_000_000:
        raise ValueError("source is outside the project or exceeds this recipe's bound")
    if sha256(path) != item["sha256"]:
        raise ValueError(f"checkpoint identity differs: {path}")
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("epoch") != item["epoch"]:
        raise ValueError("checkpoint epoch differs from the frozen manifest")
    if any(key in checkpoint for key in ("quantization", "factorization")):
        raise ValueError("this recipe averages ordinary floating checkpoints only")
    if checkpoint.get("readout") is not None or checkpoint.get("readout_tilted"):
        raise ValueError("this recipe requires the ordinary regression readout")
    return checkpoint


def derive(root: Path, manifest_path: Path) -> dict[str, object]:
    """Materialize the fixed epoch-3/4/5 mean, without fitting or evaluating data."""
    root = root.resolve(strict=True)
    manifest = json.loads(manifest_path.read_text())
    inputs = manifest["checkpoints"]
    if [item["epoch"] for item in inputs] != [3, 4, 5]:
        raise ValueError("only the predeclared epoch-3/4/5 equal mean is supported")
    destination = root / "artifacts/runs" / manifest["destination_run"]
    if destination.parent != root / "artifacts/runs" or destination.exists():
        raise ValueError("destination must be a new direct child of artifacts/runs")
    checkpoints = [load_bound(root, item) for item in inputs]
    arguments = [
        {k: v for k, v in checkpoint["args"].items() if k not in RUNTIME_ARGUMENTS}
        for checkpoint in checkpoints
    ]
    if any(args != arguments[0] for args in arguments):
        raise ValueError("scientific training arguments differ")
    if any(checkpoint["n_in"] != checkpoints[0]["n_in"] for checkpoint in checkpoints):
        raise ValueError("input representations differ")
    for field in (
        "split_manifest_sha256",
        "initial_checkpoint_sha256",
        "propensity_checkpoint_sha256",
        "fit_membership_identity",
        "normalization_identity",
        "distillation_cache",
    ):
        values = [checkpoint["provenance"].get(field) for checkpoint in checkpoints]
        if values[0] is None or any(value != values[0] for value in values):
            raise ValueError(f"training ancestry differs or is missing: {field}")
    resume = checkpoints[-1]["provenance"]["resumed_from"]
    state_path = root / resume["path"]
    if (
        resume["next_epoch"] != 4
        or sha256(state_path) != resume["sha256"]
        or state_path.parent != (root / inputs[0]["path"]).parent
        or checkpoints[1]["provenance"]["resumed_from"] != resume
    ):
        raise ValueError("checkpoints do not share the declared epoch-3 continuation")
    averaged = copy.deepcopy(checkpoints[-1])
    averaged["model"] = mean_state([checkpoint["model"] for checkpoint in checkpoints])
    derivation = {
        "method": "equal_parameter_mean_epochs_3_4_5",
        "checkpoints": inputs,
        "manifest_sha256": sha256(manifest_path),
        "normalization": "identical buffers preserved",
        "accumulation": "float64 mean, cast to original parameter dtype",
        "training_run": False,
        "quality_evaluated": False,
    }
    averaged["weight_derivation"] = derivation
    averaged["provenance"]["weight_derivation"] = derivation
    averaged["provenance"].pop("checkpoint_sha256", None)
    destination.mkdir()
    output = destination / "model_ep5.pt"
    torch.save(averaged, output)
    os.link(output, destination / "model.pt")
    digest = sha256(output)
    provenance = {
        **averaged["provenance"],
        "checkpoint_sha256": {"model_ep5.pt": digest, "model.pt": digest},
    }
    (destination / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return {
        "destination": str(destination.relative_to(root)),
        "checkpoint_sha256": digest,
        **derivation,
    }


def main() -> None:
    """Derive an audited trajectory mean from an immutable input manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(derive(args.root, args.manifest), indent=2))


if __name__ == "__main__":
    main()
