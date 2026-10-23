"""Cache and parent identities used by portable age-readout fit receipts."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from wnn.splits import file_sha256
from wnn.training.online_relationship_cache import (
    CACHE_SCHEMA,
    SequenceCache,
    row_refs_sha256,
)
from wnn.training.online_relationship_sources import module_hashes

from .age_readout_portable_paths import PortablePaths

JsonObject = dict[str, Any]


def payload_pin(receipt: JsonObject, fold_id: int, kind: str) -> JsonObject:
    """Return one full-cache identity from the six-file local verification receipt."""
    matches = [
        item for item in receipt["files"] if int(item["fold"]) == fold_id and item["kind"] == kind
    ]
    if len(matches) != 1:
        raise ValueError(f"cache-verification receipt has no unique {kind} pin for fold{fold_id}")
    return matches[0]


def require_refs(metadata: JsonObject, expected: list[dict[str, Any]], scope: str) -> None:
    """Require exact source-qualified cache row order and membership digest."""
    if metadata.get("row_refs") != expected or metadata.get("membership_sha256") != row_refs_sha256(
        expected
    ):
        raise ValueError(f"portable {scope} refs differ from frozen source-qualified membership")


def verify_loaded_cache_metadata(
    fit: SequenceCache,
    evaluation: SequenceCache,
    fit_refs: list[dict[str, Any]],
    eval_refs: list[dict[str, Any]],
    source_fold: JsonObject,
    source_recipe: JsonObject,
    repo_root: Path,
) -> None:
    """Recheck cache sidecar contracts after exact full-payload reconstruction."""
    if (
        fit.metadata.get("schema") != CACHE_SCHEMA
        or evaluation.metadata.get("schema") != CACHE_SCHEMA
    ):
        raise ValueError("portable cache sidecar schema differs from the canonical cache format")
    if list(fit.row_refs) != fit_refs or list(evaluation.row_refs) != eval_refs:
        raise ValueError("loaded portable cache order differs from source-qualified membership")
    for cache in (fit, evaluation):
        if cache.metadata.get("complete_membership") is not True or cache.metadata.get(
            "sampled_profile"
        ):
            raise ValueError("portable cache is not the complete frozen membership")
        if cache.metadata.get("source_sha256") != module_hashes(repo_root):
            raise ValueError("portable cache source hashes differ from canonical cache producers")
        if cache.metadata.get("split_manifest_sha256") != source_fold["split_manifest_sha256"]:
            raise ValueError("portable cache belongs to a different frozen split")
        if cache.metadata.get("subset_manifest_sha256") != source_fold["subset"]["sha256"]:
            raise ValueError("portable cache belongs to a different frozen fit subset")
    if evaluation.metadata.get("read_contract") != {"targets": False, "scored_mask": False}:
        raise ValueError("portable evaluation cache does not bind target-free source reads")
    if evaluation.metadata.get("zero_columns") != [112, 113, 114]:
        raise ValueError("portable evaluation target/weight columns are not declared zero")
    if fit.metadata.get("parent") != evaluation.metadata.get("parent"):
        raise ValueError("portable fit/evaluation caches have different parent endpoints")
    if fit.metadata.get("parent_mean") != evaluation.metadata.get(
        "parent_mean"
    ) or fit.metadata.get("parent_std") != evaluation.metadata.get("parent_std"):
        raise ValueError("portable fit/evaluation parent normalization differs")
    readout = source_recipe["readout"]
    if fit.metadata.get("row_weight_contract") != readout["row_weight_args"]:
        raise ValueError("portable fit-cache row weights differ from frozen recipe semantics")
    if fit.metadata.get("row_weight_batch_semantics") != readout["row_weight_batch_semantics"]:
        raise ValueError("portable fit-cache row-weight batching differs from frozen semantics")
    parent = fit.metadata["parent"]
    _verify_parent_identity(parent, source_fold["parent"])
    propensity = fit.metadata.get("propensity", {})
    expected_propensity = source_fold["canonical_propensity"]
    if any(
        propensity.get(key) != expected_propensity.get(key)
        for key in ("run", "checkpoint_sha256", "provenance_sha256")
    ):
        raise ValueError("portable fit-cache propensity ancestry differs from the frozen recipe")
    if parent.get("endpoint_backend") != "cpu" or parent.get("endpoint_precision") != "strict_fp32":
        raise ValueError("portable cached parent must remain the frozen strict-FP32 CPU endpoint")
    for values_key, identity_key in (
        ("parent_mean", "mean_identity"),
        ("parent_std", "std_identity"),
    ):
        digest = hashlib.sha256(
            np.asarray(fit.metadata[values_key], dtype="<f4").tobytes()
        ).hexdigest()
        if digest != parent.get(identity_key):
            raise ValueError("portable cached parent normalization bytes differ from identity")


def cache_receipt(
    recipe: JsonObject,
    source_sha: str,
    fold: JsonObject,
    source_fold: JsonObject,
    fit: SequenceCache,
    evaluation: SequenceCache,
    parent_sidecar: JsonObject,
    parent_milestone: JsonObject,
    cache_dir: Path,
    bundle_fold: JsonObject,
    paths: Any,
    source_identity: JsonObject,
    verification: JsonObject,
    metadata_attestation_sha: str,
) -> dict[str, object]:
    """Record logical source paths, actual Kaggle paths and both cache generations."""
    cells = [cell for cell in recipe["cells"] if int(cell["fold"]) == int(fold["fold"])]
    logical_outputs = {str(cell["arm"]): str(cell["output"]) for cell in cells}
    actual_outputs = {str(cell["arm"]): str(paths.output_root / str(cell["run"])) for cell in cells}
    parent_manifest = parent_milestone["sidecars"]["evaluation_predictions.json"]
    logical_sources = {
        split: {
            "path": value["original_path"],
            "mtime_ns": value["original_mtime_ns"],
            "bytes": value["bytes"],
            "footer_sha256": value["footer_sha256"],
        }
        for split, value in source_identity.items()
    }
    actual_sources = {
        split: {
            "path": value["actual_path"],
            "mtime_ns": value["actual_mtime_ns"],
            "bytes": value["bytes"],
            "footer_sha256": value["footer_sha256"],
        }
        for split, value in source_identity.items()
    }
    return {
        "source_cache_recipe_path": recipe["source_cache_recipe"]["path"],
        "source_cache_recipe_sha256": source_sha,
        "cache_directory": str(cache_dir),
        "path_mapping": {
            "logical": {
                "cache_directory": str(source_fold["parent_cache"]),
                "storage_root": str(recipe["storage"]["new_root"]),
                "output_directories": logical_outputs,
                "raw_sources": logical_sources,
            },
            "actual": {
                "bundle_root": str(paths.bundle_root),
                "cache_directory": str(cache_dir),
                "storage_root": str(paths.workspace_root),
                "output_directories": actual_outputs,
                "raw_sources": actual_sources,
            },
        },
        "live_disk": portable_disk_snapshot(paths),
        "portable_validation": {
            "mode": "attested_full_hash_remote_rebuild",
            "canonical_mtime_loader_called": False,
            "cache_payload_hash_verified": True,
            "transport_sha256": paths.transport_sha256,
            "actual_parent_metadata_check_sha256": metadata_attestation_sha,
            "target_cache_opened": False,
            "evaluation_targets_or_masks_read": False,
        },
        "fit_cache": _cache_artifact(fit, bundle_fold["fit_suffix5"]["path"], verification),
        "evaluation_cache": _cache_artifact(
            evaluation, bundle_fold["evaluation_predictions"]["path"], verification
        ),
        "parent_prediction_path": str(cache_dir / "evaluation_predictions.npy"),
        "parent_prediction_logical_path": parent_sidecar["prediction_path"],
        "parent_prediction_sha256": parent_sidecar["prediction_sha256"],
        "parent_manifest_path": str(cache_dir / "evaluation_predictions.json"),
        "parent_manifest_logical_path": parent_manifest["path"],
        "parent_manifest_sha256": file_sha256(cache_dir / "evaluation_predictions.json"),
        "parent_normalization": {
            "mean": fit.metadata["parent_mean"],
            "std": fit.metadata["parent_std"],
            "mean_identity": fit.metadata["parent"]["mean_sha256"],
            "std_identity": fit.metadata["parent"]["std_sha256"],
        },
        "payload_hash_verified": True,
    }


def _cache_artifact(
    cache: SequenceCache, suffix_path: str, verification: JsonObject
) -> dict[str, object]:
    kinds = {"in_sample_fit": "fit_sequences", "feature_only_eval": "evaluation_sequences"}
    kind = kinds[cache.metadata["scope"]]
    pin = next(
        item
        for item in verification["files"]
        if item["kind"] == kind and int(item["fold"]) == int(cache.metadata["fold"])
    )
    return {
        "path": str(cache.data_path),
        "sidecar_path": str(cache.sidecar_path),
        "sidecar_sha256": file_sha256(cache.sidecar_path),
        "payload_sha256": cache.metadata["data_sha256"],
        "membership_sha256": cache.metadata["membership_sha256"],
        "sequences": len(cache.row_refs),
        "shape": list(cache.array.shape),
        "dtype": "<f4",
        "transport_suffix_path": suffix_path,
        "original_identity": pin["identity"],
        "actual_mtime_ns": cache.data_path.stat().st_mtime_ns,
    }


def _verify_parent_identity(cached: JsonObject, expected: JsonObject) -> None:
    fields = (
        ("run", "run"),
        ("epoch", "epoch"),
        ("checkpoint_sha256", "checkpoint_sha256"),
        ("provenance_sha256", "provenance_sha256"),
        ("mean_sha256", "mean_identity"),
        ("std_sha256", "std_identity"),
    )
    if any(cached.get(key) != expected.get(recipe_key) for key, recipe_key in fields):
        raise ValueError("portable cache parent checkpoint or normalization identity differs")


def portable_disk_snapshot(paths: PortablePaths) -> dict[str, object]:
    """Stat regular files only and report the live bundle plus worktree size."""
    bundle_bytes = _tree_bytes(paths.bundle_root)
    work_bytes = _tree_bytes(paths.workspace_root)
    return {
        "bundle_regular_file_bytes": bundle_bytes,
        "work_regular_file_bytes": work_bytes,
        "live_disk_bytes": bundle_bytes + work_bytes,
        "method": "sum_regular_file_st_size; no file payloads opened",
        "raw_input_mount_excluded": True,
    }


def update_disk_peak(current: dict[str, object] | None, paths: PortablePaths) -> dict[str, object]:
    """Keep the largest observed regular-file footprint across a worker phase."""
    candidate = portable_disk_snapshot(paths)
    if current is None or int(candidate["live_disk_bytes"]) > int(current["live_disk_bytes"]):
        return candidate
    return current


def _tree_bytes(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"portable disk accounting encountered a symlink: {path}")
        if path.is_file():
            total += path.stat().st_size
    return total
