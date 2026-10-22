"""Validate the exact sequence-cache headers, membership and source identities."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any  # JSON metadata is validated at runtime.

import numpy as np
from online_relationship_grade_contract import (
    ROWS,
    WIDTH,
    _resolve,
    _verify_source_hashes,
    read_json,
    row_refs_sha256,
)
from online_relationship_grade_core import npy_coverage, sha256_file


def verify_sequence_cache(
    root: Path,
    split: dict[str, Any],
    data_path: Path,
    metadata_path: Path,
    scope: str,
    refs: list[dict[str, Any]],
    recipe: dict[str, Any],
    recipe_sha: str,
    fold: dict[str, Any],
    subset: dict[str, Any],
) -> dict[str, Any]:
    metadata = read_json(metadata_path)
    fold_id = int(fold["fold"])
    expected = {
        "schema": "wnn.online-relationships.sequence-cache.v1",
        "scope": scope,
        "fold": fold_id,
        "seed": int(fold["seed"]),
        "recipe_id": recipe["recipe_id"],
        "recipe_sha256": recipe_sha,
        "split_manifest_sha256": fold["split_manifest_sha256"],
        "subset_manifest_sha256": fold["subset"]["sha256"],
        "subset_members_sha256": subset["selected_members_sha256"],
        "row_refs": refs,
        "membership_sha256": row_refs_sha256(refs),
        "complete_membership": True,
        "scope_label": scope,
        "data_file": data_path.name,
        "shape": [len(refs), ROWS, WIDTH],
        "dtype": "<f4",
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"F{fold_id} {scope} cache field {key} differs from the recipe")
    if metadata.get("sampled_profile") is not False:
        raise ValueError(f"F{fold_id} {scope} cache is a profile sample, not complete")
    if metadata.get("split_manifest") != str(_resolve(root, fold["split_manifest"])):
        raise ValueError(f"F{fold_id} {scope} cache names a different split manifest")
    if metadata.get("subset_manifest") != fold["subset"]["path"]:
        raise ValueError(f"F{fold_id} {scope} cache names a different fit subset")
    if metadata.get("layout") != {
        "raw": [0, 112],
        "target": [112, 114],
        "row_weight": [114, 115],
        "parent_prediction": [115, 117],
    }:
        raise ValueError(f"F{fold_id} {scope} cache layout changed")
    parent_identity = metadata.get("parent", {})
    parent_recipe = fold["parent"]
    expected_parent = {
        "run": parent_recipe["run"],
        "epoch": parent_recipe["epoch"],
        "checkpoint": str(_resolve(root, parent_recipe["checkpoint"])),
        "checkpoint_sha256": parent_recipe["checkpoint_sha256"],
        "provenance": str(_resolve(root, parent_recipe["provenance"])),
        "provenance_sha256": parent_recipe["provenance_sha256"],
        "mean_sha256": parent_recipe["mean_identity"],
        "std_sha256": parent_recipe["std_identity"],
        "input_width": parent_recipe["input_width"],
    }
    if any(parent_identity.get(key) != value for key, value in expected_parent.items()):
        raise ValueError(f"F{fold_id} {scope} cache parent identity differs from the recipe")
    if metadata.get("propensity") is not None and scope != "in_sample_fit":
        raise ValueError(f"F{fold_id} evaluation cache unexpectedly binds propensity outputs")
    if scope == "in_sample_fit":
        propensity = metadata.get("propensity", {})
        expected_propensity = fold["canonical_propensity"]
        if (
            propensity.get("run") != expected_propensity["run"]
            or propensity.get("checkpoint_sha256") != expected_propensity["checkpoint_sha256"]
            or propensity.get("provenance_sha256") != expected_propensity["provenance_sha256"]
            or propensity.get("precision")
            != "FP32 CPU probabilities; no cached propensity rows reused"
        ):
            raise ValueError(f"F{fold_id} fit cache uses a different propensity ancestor")
    coverage = npy_coverage(data_path)
    if coverage["shape"] != expected["shape"] or coverage["dtype"] != "<f4":
        raise ValueError(f"F{fold_id} {scope} NPY header differs from its frozen schema")
    data_sha = sha256_file(data_path)
    if metadata.get("data_sha256") != data_sha:
        raise ValueError(f"F{fold_id} {scope} payload SHA256 differs from the sidecar")
    _verify_parent_stats(fold, metadata)
    _verify_dataset_identities(root, split, metadata)
    _verify_source_hashes(root, metadata.get("source_sha256", {}))
    required = {
        "src/wnn/splits.py",
        "src/wnn/training/membership.py",
        "src/wnn/training/online_relationship_data.py",
        "src/wnn/training/online_relationship_data_writer.py",
        "src/wnn/training/online_relationship_cache.py",
        "src/wnn/training/online_relationship_sources.py",
        "src/wnn/training/online_relationship.py",
        "src/wnn/training/online_relationship_fit.py",
        "src/wnn/training/inputs.py",
        "src/wnn/training/targets.py",
        "src/wnn/training/models.py",
        "src/wnn/training/evaluation.py",
        "src/wnn/training/environment.py",
        "src/wnn/data.py",
        "src/wnn/features.py",
        "src/wnn/propensity.py",
        "src/wnn/online_relationship.py",
        "src/wnn/online_relationship_numba.py",
    }
    if not required.issubset(metadata["source_sha256"]):
        raise ValueError(f"F{fold_id} {scope} cache source map omits builder dependencies")
    return {**metadata, "data_sha256": data_sha}


def _verify_parent_stats(fold: dict[str, Any], metadata: dict[str, Any]) -> None:
    parent = fold["parent"]
    for name, identity in (("parent_mean", "mean_identity"), ("parent_std", "std_identity")):
        values = np.asarray(metadata.get(name), dtype="<f4")
        if values.shape != (211,) or not np.isfinite(values).all():
            raise ValueError(f"F{fold['fold']} cache {name} must be a finite 211-vector")
        digest = hashlib.sha256(values.tobytes()).hexdigest()
        if digest != parent[identity]:
            raise ValueError(f"F{fold['fold']} cache {name} differs from the frozen parent")
        if name == "parent_std" and np.any(values <= 0):
            raise ValueError(f"F{fold['fold']} cache parent std must be positive")


def _verify_dataset_identities(root: Path, split: dict[str, Any], metadata: dict[str, Any]) -> None:
    """Require the cached train/valid Parquet metadata to match its split manifest."""
    identity = metadata.get("source_datasets", {})
    for name in ("train", "valid"):
        record = identity.get("sources", {}).get(name)
        if not isinstance(record, dict):
            raise ValueError(f"parent cache lacks {name} dataset identity")
        path = root / "datasets" / f"{name}.parquet"
        stat = path.stat()
        if (
            record.get("path") != str(path)
            or record.get("bytes") != stat.st_size
            or record.get("mtime_ns") != stat.st_mtime_ns
            or record.get("campaign_identity")
            != split.get("sources", {}).get(f"datasets/{name}.parquet")
        ):
            raise ValueError(f"parent cache {name} dataset identity differs from the split")
