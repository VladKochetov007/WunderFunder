"""Read-only adapters for the frozen online parent caches."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from wnn.splits import file_sha256
from wnn.training.online_relationship_cache import load_sequence_cache, sequence_cache_paths
from wnn.training.online_relationship_cache_validation import load_fold_caches


def load_profile_caches(
    root: Path, recipe: dict[str, Any], fold: dict[str, Any], fold_id: int
) -> tuple[Any, Any, dict[str, object]]:
    """Bind pinned cache sidecars and NPY headers without scanning payload hashes."""
    source_recipe_path, _, source_recipe_sha = _source_recipe(root, recipe)
    cache_dir = Path(fold["parent_cache"])
    fit_data, fit_meta = sequence_cache_paths(cache_dir, "in_sample_fit")
    eval_data, eval_meta = sequence_cache_paths(cache_dir, "feature_only_eval")
    milestone = _json(
        root
        / "research/experiment_drafts/online-relationships-20261022/PARENT_CACHE_MILESTONE_R1.json"
    )
    if milestone["execution"]["recipe_sha256"] != source_recipe_sha:
        raise ValueError("parent-cache milestone does not bind the declared source recipe")
    pinned = milestone["parent_caches"][str(fold_id)]["sidecars"]
    fit_sidecar_sha = file_sha256(fit_meta)
    eval_sidecar_sha = file_sha256(eval_meta)
    if fit_sidecar_sha != pinned["fit_sequences.json"]["sha256"]:
        raise ValueError("fit sidecar differs from the immutable parent-cache milestone")
    if eval_sidecar_sha != pinned["evaluation_sequences.json"]["sha256"]:
        raise ValueError("evaluation sidecar differs from the immutable parent-cache milestone")
    fit_cache = load_sequence_cache(
        fit_data,
        fit_meta,
        expected_scope="in_sample_fit",
        expected_recipe_sha256=source_recipe_sha,
        expected_fold=fold_id,
        verify_payload=False,
    )
    eval_cache = load_sequence_cache(
        eval_data,
        eval_meta,
        expected_scope="feature_only_eval",
        expected_recipe_sha256=source_recipe_sha,
        expected_fold=fold_id,
        verify_payload=False,
    )
    _validate_cache_metadata(fit_cache, eval_cache, fold)
    parent_path = cache_dir / "evaluation_predictions.npy"
    parent_manifest_path = cache_dir / "evaluation_predictions.json"
    parent_manifest_sha = file_sha256(parent_manifest_path)
    expected_parent_sha = pinned["evaluation_predictions.json"]["sha256"]
    if parent_manifest_sha != expected_parent_sha:
        raise ValueError("parent prediction manifest differs from the immutable milestone")
    parent_manifest = _json(parent_manifest_path)
    return (
        fit_cache,
        eval_cache,
        _receipt(
            cache_dir,
            source_recipe_path,
            source_recipe_sha,
            fit_cache,
            eval_cache,
            fit_sidecar_sha,
            eval_sidecar_sha,
            parent_manifest_path,
            parent_path,
            parent_manifest_sha,
            parent_manifest,
            payload_hash_verified=False,
        ),
    )


def load_validated_caches(
    root: Path, recipe: dict[str, Any], fold: dict[str, Any], fold_id: int
) -> tuple[Any, Any, dict[str, object]]:
    """Run the original full cache/ancestry validator once per fold on CPU parent data."""
    source_recipe_path, source_recipe, source_recipe_sha = _source_recipe(root, recipe)
    source_fold = next(item for item in source_recipe["folds"] if int(item["fold"]) == fold_id)
    fit_cache, eval_cache, parent_manifest = load_fold_caches(
        root,
        source_recipe,
        source_recipe_sha,
        source_fold,
        "cpu",
    )
    milestone = _json(
        root
        / "research/experiment_drafts/online-relationships-20261022/PARENT_CACHE_MILESTONE_R1.json"
    )
    pinned = milestone["parent_caches"][str(fold_id)]["sidecars"]
    cache_dir = Path(fold["parent_cache"])
    fit_data, fit_meta = sequence_cache_paths(cache_dir, "in_sample_fit")
    eval_data, eval_meta = sequence_cache_paths(cache_dir, "feature_only_eval")
    fit_sidecar_sha = file_sha256(fit_meta)
    eval_sidecar_sha = file_sha256(eval_meta)
    if (
        fit_sidecar_sha != pinned["fit_sequences.json"]["sha256"]
        or eval_sidecar_sha != pinned["evaluation_sequences.json"]["sha256"]
    ):
        raise ValueError("validated cache sidecars differ from the immutable milestone")
    _validate_cache_metadata(fit_cache, eval_cache, fold)
    parent_manifest_path = Path(str(parent_manifest["manifest_path"]))
    parent_path = Path(str(parent_manifest["prediction_path"]))
    return (
        fit_cache,
        eval_cache,
        _receipt(
            cache_dir,
            source_recipe_path,
            source_recipe_sha,
            fit_cache,
            eval_cache,
            fit_sidecar_sha,
            eval_sidecar_sha,
            parent_manifest_path,
            parent_path,
            str(parent_manifest["manifest_sha256"]),
            parent_manifest,
            payload_hash_verified=True,
        ),
    )


def _source_recipe(root: Path, recipe: dict[str, Any]) -> tuple[Path, dict[str, Any], str]:
    source_path = Path(recipe["source_cache_recipe"]["path"])
    source_path = source_path if source_path.is_absolute() else root / source_path
    source_sha = file_sha256(source_path)
    if source_sha != recipe["source_cache_recipe"]["sha256"]:
        raise ValueError("immutable source-cache recipe SHA differs from new recipe")
    return source_path, _json(source_path), source_sha


def _validate_cache_metadata(fit: Any, evaluation: Any, fold: dict[str, Any]) -> None:
    if fit.metadata.get("complete_membership") is not True or fit.metadata.get("sampled_profile"):
        raise ValueError("fit cache is not the complete frozen 10-percent membership")
    if evaluation.metadata.get("complete_membership") is not True or evaluation.metadata.get(
        "sampled_profile"
    ):
        raise ValueError("evaluation cache is not the complete frozen fold membership")
    if fit.metadata.get("scope") != "in_sample_fit":
        raise ValueError("fit cache scope differs from the declared in-sample parent scope")
    if evaluation.metadata.get("read_contract") != {"targets": False, "scored_mask": False}:
        raise ValueError("evaluation cache did not exclude targets and scored masks")
    if evaluation.metadata.get("zero_columns") != [112, 113, 114]:
        raise ValueError("evaluation cache zero-column contract changed")
    if len(fit.row_refs) != int(fit.metadata["shape"][0]):
        raise ValueError("fit cache references do not match its NPY header")
    if list(evaluation.row_refs) != [
        {"split": "valid", "row_group": int(group)} for group in fold["evaluation_groups"]
    ]:
        raise ValueError("evaluation cache sequence order differs from the frozen fold")
    if fit.metadata.get("parent_mean") != evaluation.metadata.get(
        "parent_mean"
    ) or fit.metadata.get("parent_std") != evaluation.metadata.get("parent_std"):
        raise ValueError("fit and evaluation caches use different parent normalization")


def _receipt(
    cache_dir: Path,
    recipe_path: Path,
    recipe_sha: str,
    fit: Any,
    evaluation: Any,
    fit_sidecar_sha: str,
    eval_sidecar_sha: str,
    parent_manifest_path: Path,
    parent_path: Path,
    parent_manifest_sha: str,
    parent_manifest: dict[str, Any],
    *,
    payload_hash_verified: bool,
) -> dict[str, object]:
    return {
        "source_cache_recipe_path": str(recipe_path),
        "source_cache_recipe_sha256": recipe_sha,
        "cache_directory": str(cache_dir),
        "fit_cache": {
            "path": str(fit.data_path),
            "sidecar_path": str(fit.sidecar_path),
            "sidecar_sha256": fit_sidecar_sha,
            "payload_sha256": fit.metadata["data_sha256"],
            "membership_sha256": fit.metadata["membership_sha256"],
            "sequences": len(fit.row_refs),
            "shape": list(fit.array.shape),
            "dtype": "<f4",
        },
        "evaluation_cache": {
            "path": str(evaluation.data_path),
            "sidecar_path": str(evaluation.sidecar_path),
            "sidecar_sha256": eval_sidecar_sha,
            "payload_sha256": evaluation.metadata["data_sha256"],
            "membership_sha256": evaluation.metadata["membership_sha256"],
            "sequences": len(evaluation.row_refs),
            "shape": list(evaluation.array.shape),
            "dtype": "<f4",
            "read_contract": evaluation.metadata["read_contract"],
        },
        "parent_prediction_path": str(parent_path),
        "parent_prediction_sha256": parent_manifest["prediction_sha256"],
        "parent_manifest_path": str(parent_manifest_path),
        "parent_manifest_sha256": parent_manifest_sha,
        "parent_normalization": {
            "mean": fit.metadata["parent_mean"],
            "std": fit.metadata["parent_std"],
            "mean_identity": fit.metadata["parent"]["mean_sha256"],
            "std_identity": fit.metadata["parent"]["std_sha256"],
        },
        "payload_hash_verified": payload_hash_verified,
    }


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object at {path}")
    return value
