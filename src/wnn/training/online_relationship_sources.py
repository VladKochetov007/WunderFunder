"""Load and fingerprint the frozen parents and propensity dependencies."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from wnn.propensity import load_propensity
from wnn.splits import file_sha256
from wnn.training.environment import TrainingEnvironment
from wnn.training.models import make_model
from wnn.training.online_relationship_cache import CACHE_SCHEMA, ROWS, WIDTH, row_refs_sha256


def load_parent(
    root: Path, fold: dict[str, Any], device: str
) -> tuple[torch.nn.Module, TrainingEnvironment, dict[str, object], np.ndarray, np.ndarray]:
    """Load one recipe-pinned parent, its normalization, and inference readout."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    parent = fold["parent"]
    checkpoint_path = resolve(root, Path(parent["checkpoint"]))
    provenance_path = resolve(root, Path(parent["provenance"]))
    check_hash(checkpoint_path, parent["checkpoint_sha256"], "parent checkpoint")
    check_hash(provenance_path, parent["provenance_sha256"], "parent provenance")
    provenance = read_json(provenance_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("epoch") != parent["epoch"]:
        raise ValueError("parent checkpoint epoch differs from the frozen recipe")
    embedded = checkpoint.get("provenance", {})
    if embedded.get("split_manifest_sha256") != fold["split_manifest_sha256"]:
        raise ValueError("parent checkpoint belongs to a different fold")
    if embedded.get("fit_row_groups") != provenance.get("fit_row_groups"):
        raise ValueError("parent checkpoint membership differs from its provenance")
    if int(checkpoint.get("n_in", 0)) != int(parent["input_width"]):
        raise ValueError("parent input width differs from the recipe")
    state, args = checkpoint.get("model"), checkpoint.get("args")
    if not isinstance(state, Mapping) or not isinstance(args, Mapping):
        raise ValueError("parent checkpoint lacks model state or training arguments")
    _require_clean_211_inputs(args)
    model = make_model(dict(args), int(checkpoint["n_in"]), _output_width(state))
    model.load_state_dict(state, strict=True)
    model = model.to(torch.device(device)).eval()
    model._inference_fp32 = True
    model._online_parent_chunk = int(args.get("chunk", 4000))
    mean = model.mean.detach().float().cpu().numpy().astype(np.float32, copy=True)
    std = model.std.detach().float().cpu().numpy().astype(np.float32, copy=True)
    if (
        mean.shape != (211,)
        or std.shape != (211,)
        or not np.isfinite(mean).all()
        or not np.isfinite(std).all()
        or np.any(std <= 0)
    ):
        raise ValueError("parent normalization must be finite, positive and width 211")
    mean_sha, std_sha = tensor_bytes_sha(model.mean), tensor_bytes_sha(model.std)
    if mean_sha != parent["mean_identity"] or std_sha != parent["std_identity"]:
        raise ValueError("parent normalization identity differs from the frozen recipe")
    environment = TrainingEnvironment(root=root)
    readout = checkpoint.get("readout")
    environment.readout.tables = (
        None if readout is None else tuple(value.to(device) for value in readout)
    )
    environment.readout.tilted = bool(checkpoint.get("readout_tilted", False))
    identity = {
        "run": parent["run"],
        "epoch": int(parent["epoch"]),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": parent["checkpoint_sha256"],
        "provenance": str(provenance_path),
        "provenance_sha256": parent["provenance_sha256"],
        "mean_sha256": mean_sha,
        "std_sha256": std_sha,
        "input_width": 211,
        "endpoint_precision": "strict_fp32",
        "endpoint_backend": str(torch.device(device)),
    }
    return model, environment, identity, mean, std


def load_fresh_propensity(
    root: Path, fold: dict[str, Any]
) -> tuple[torch.nn.Module, dict[str, object]]:
    """Load the pinned fold propensity model in FP32 without reusing row caches."""
    source = fold["canonical_propensity"]
    run_root = root / "artifacts" / "runs" / source["run"]
    checkpoint_path = run_root / "model.pt"
    provenance_path = run_root / "provenance.json"
    check_hash(checkpoint_path, source["checkpoint_sha256"], "propensity checkpoint")
    check_hash(provenance_path, source["provenance_sha256"], "propensity provenance")
    metadata = read_json(provenance_path)
    if metadata.get("checkpoint_sha256", {}).get("model.pt") != source["checkpoint_sha256"]:
        raise ValueError("propensity provenance does not bind the pinned checkpoint")
    model = load_propensity(checkpoint_path, torch.device("cpu")).float().eval()
    if model.mean.shape != (211,):
        raise ValueError("fresh propensity model must consume the canonical 211 inputs")
    identity = {
        "run": source["run"],
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": source["checkpoint_sha256"],
        "provenance": str(provenance_path),
        "provenance_sha256": source["provenance_sha256"],
        "precision": "FP32 CPU probabilities; no cached propensity rows reused",
    }
    return model, identity


def dataset_identities(root: Path, split_path: Path, campaign: Any) -> dict[str, object]:
    """Record the campaign's file-size and mtime identities without full Parquet hashing."""
    manifest = read_json(split_path)
    sources = {}
    for split in ("train", "valid"):
        path = root / "datasets" / f"{split}.parquet"
        stat = path.stat()
        sources[split] = {
            "path": str(path),
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "campaign_identity": manifest["sources"][f"datasets/{split}.parquet"],
        }
    return {"campaign_split_sha256": campaign.sha256, "sources": sources}


def cache_metadata(
    data_name: str,
    data_path: Path,
    prediction_path: Path | None,
    refs: list[dict[str, object]],
    scope: str,
    recipe: dict[str, Any],
    recipe_sha: str,
    fold: dict[str, Any],
    subset: dict[str, object],
    split_path: Path,
    data_sources: dict[str, object],
    propensity_identity: dict[str, object],
    parent_identity: dict[str, object],
    parent_mean: np.ndarray,
    parent_std: np.ndarray,
    row_weight_args: dict[str, object],
    sample_mode: bool,
    root: Path,
) -> dict[str, object]:
    """Bind cache bytes, training inputs, fit weights and exact source order."""
    return {
        "schema": CACHE_SCHEMA,
        "scope": scope,
        "prediction_scope": scope,
        "fold": int(fold["fold"]),
        "seed": int(fold["seed"]),
        "recipe_id": recipe["recipe_id"],
        "recipe_sha256": recipe_sha,
        "split_manifest": str(split_path),
        "split_manifest_sha256": file_sha256(split_path),
        "subset_manifest": fold["subset"]["path"],
        "subset_manifest_sha256": fold["subset"]["sha256"],
        "subset_members_sha256": subset["manifest"]["selected_members_sha256"],
        "row_refs": refs,
        "membership_sha256": row_refs_sha256(refs),
        "complete_membership": not sample_mode,
        "scope_label": "profile_sample" if sample_mode else scope,
        "source_datasets": data_sources,
        "parent": parent_identity,
        "propensity": propensity_identity if scope == "in_sample_fit" else None,
        "parent_mean": parent_mean.tolist(),
        "parent_std": parent_std.tolist(),
        "row_weight_contract": row_weight_args if scope == "in_sample_fit" else None,
        "row_weight_batch_semantics": (
            "one complete sequence per main_row_weights call; true-mask mass scale is sequence-local"
            if scope == "in_sample_fit"
            else None
        ),
        "read_contract": {
            "targets": scope == "in_sample_fit",
            "scored_mask": scope == "in_sample_fit",
        },
        "data_file": data_name,
        "shape": [len(refs), ROWS, WIDTH],
        "dtype": "<f4",
        "layout": {
            "raw": [0, 112],
            "target": [112, 114],
            "row_weight": [114, 115],
            "parent_prediction": [115, 117],
        },
        "zero_columns": [112, 113, 114] if scope == "feature_only_eval" else [],
        "parent_prediction_file": "evaluation_predictions.npy" if prediction_path else None,
        "parent_prediction_sha256": file_sha256(prediction_path) if prediction_path else None,
        "source_sha256": module_hashes(root),
        "data_sha256": file_sha256(data_path),
        "sampled_profile": sample_mode,
    }


def module_hashes(root: Path) -> dict[str, str]:
    """Hash implementation modules that construct inputs, weights, parents and caches."""
    paths = (
        root / "src/wnn/training/online_relationship_data.py",
        root / "src/wnn/training/online_relationship_data_writer.py",
        root / "src/wnn/training/online_relationship_parent_batch.py",
        root / "src/wnn/splits.py",
        root / "src/wnn/training/membership.py",
        root / "src/wnn/training/online_relationship_cache.py",
        root / "src/wnn/training/online_relationship_sources.py",
        root / "src/wnn/training/online_relationship.py",
        root / "src/wnn/training/online_relationship_fit.py",
        root / "src/wnn/training/inputs.py",
        root / "src/wnn/training/targets.py",
        root / "src/wnn/training/models.py",
        root / "src/wnn/training/evaluation.py",
        root / "src/wnn/training/environment.py",
        root / "src/wnn/data.py",
        root / "src/wnn/features.py",
        root / "src/wnn/propensity.py",
        root / "src/wnn/online_relationship.py",
        root / "src/wnn/online_relationship_numba.py",
    )
    return {path.relative_to(root).as_posix(): file_sha256(path) for path in paths}


def tensor_bytes_sha(tensor: torch.Tensor) -> str:
    """Hash the exact contiguous CPU tensor bytes used by the frozen recipe."""
    return hashlib.sha256(
        tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def find_fold(recipe: dict[str, Any], fold_index: int) -> dict[str, Any]:
    """Return the recipe's unique fold record."""
    matches = [fold for fold in recipe.get("folds", []) if int(fold.get("fold", -1)) == fold_index]
    if len(matches) != 1:
        raise ValueError(f"recipe must contain exactly one fold {fold_index}")
    return matches[0]


def check_hash(path: Path, expected: str, name: str) -> None:
    """Require a file to match its immutable recipe hash."""
    if file_sha256(path) != expected:
        raise ValueError(f"{name} SHA256 differs from the frozen recipe")


def resolve(root: Path, path: Path) -> Path:
    """Resolve relative recipe paths against the project root."""
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def read_json(path: Path) -> dict[str, Any]:
    """Read bounded JSON metadata as an object."""
    if path.stat().st_size > 5_000_000:
        raise ValueError(f"metadata exceeds the 5 MB bound: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _output_width(state: Mapping[str, object]) -> int:
    """Infer the final linear head width from a canonical checkpoint state."""
    options = []
    for name, value in state.items():
        if name.startswith("head.") and name.endswith("weight") and getattr(value, "ndim", 0) == 2:
            suffix = name.removeprefix("head.").removesuffix(".weight")
            options.append((int(suffix) if suffix.isdecimal() else -1, int(value.shape[0])))
    if not options:
        raise ValueError("could not infer the parent model's output width")
    return max(options)[1]


def _require_clean_211_inputs(args: Mapping[str, object]) -> None:
    """Reject parent preprocessing that cannot be reconstructed as raw+event inputs."""
    expected = {
        "research_features": "none",
        "research_feature_control": "none",
        "online_norm": False,
        "heuristics": False,
        "raw_only": False,
        "privileged_mode": "none",
        "privileged_horizon": 8,
        "sentinel_base_width": 0,
        "sentinel_indicators": "none",
    }
    for name, value in expected.items():
        actual = args.get(name, value)
        if actual != value:
            raise ValueError(
                f"parent input option {name} is incompatible with the clean 211 inputs"
            )
    if args.get("prop_input") not in (None, False, ""):
        raise ValueError("parent propensity input is incompatible with the clean 211 inputs")
