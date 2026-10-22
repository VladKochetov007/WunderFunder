"""Verify parent caches and the complete source-qualified evaluation endpoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any  # JSON metadata is validated at runtime.

import numpy as np
from online_relationship_grade_cache import verify_sequence_cache
from online_relationship_grade_contract import (
    PARENT_END,
    PARENT_START,
    RAW_END,
    ROW_WEIGHT_CONTRACT,
    ROWS,
    ParentCacheArtifacts,
    _require_sha,
    _resolve,
    read_json,
    row_refs_sha256,
)
from online_relationship_grade_core import prediction_memmap, sha256_file

from wnn.splits import load_campaign_split
from wnn.training.membership import load_component_fit_subset


def verify_parent_cache(
    root: Path,
    recipe: dict[str, Any],
    recipe_sha: str,
    fold: dict[str, Any],
    recipe_path: Path,
) -> ParentCacheArtifacts:
    """Verify fit/eval parent caches, source order, zero labels and parent endpoint."""
    fold_id = int(fold["fold"])
    cache_dir = Path(fold["parent_cache"]).resolve()
    scratch = Path(recipe["resource_contract"]["scratch"]).resolve()
    if not cache_dir.is_relative_to(scratch):
        raise ValueError(f"F{fold_id} parent cache escapes physical scratch")
    split_path = _resolve(root, fold["split_manifest"])
    subset_path = _resolve(root, fold["subset"]["path"])
    campaign = load_campaign_split(split_path, root)
    split = read_json(split_path)
    if campaign.fold != fold_id or campaign.sha256 != fold["split_manifest_sha256"]:
        raise ValueError(f"F{fold_id} loaded campaign split identity differs from the recipe")
    subset_result = load_component_fit_subset(
        subset_path, campaign, expected_experiment_id=recipe["experiment_id"]
    )
    subset = subset_result["manifest"]
    eval_groups = [int(group) for group in fold["evaluation_groups"]]
    if len(eval_groups) != {0: 468, 1: 457}[fold_id]:
        raise ValueError(f"F{fold_id} recipe does not use the complete evaluation fold")
    if list(campaign.evaluation_groups) != eval_groups:
        raise ValueError(f"F{fold_id} evaluation order differs from the split manifest")
    if subset.get("evaluation_groups") != sorted(eval_groups):
        raise ValueError(f"F{fold_id} fit subset names different evaluation groups")
    if subset.get("split_manifest_sha256") != fold["split_manifest_sha256"]:
        raise ValueError(f"F{fold_id} subset uses a different campaign split")
    if subset.get("evaluation_membership_sha256") != fold["evaluation_membership_sha256"]:
        raise ValueError(f"F{fold_id} subset evaluation digest differs from the recipe")
    eval_refs = [{"split": "valid", "row_group": group} for group in eval_groups]
    fit_refs = [
        {"split": split_name, "row_group": int(group)}
        for split_name in ("train", "valid")
        for group in subset["selected_groups"][split_name]
    ]

    fit_path, fit_meta_path = cache_dir / "fit_sequences.npy", cache_dir / "fit_sequences.json"
    eval_path, eval_meta_path = (
        cache_dir / "evaluation_sequences.npy",
        cache_dir / "evaluation_sequences.json",
    )
    fit_meta = verify_sequence_cache(
        root,
        split,
        fit_path,
        fit_meta_path,
        "in_sample_fit",
        fit_refs,
        recipe,
        recipe_sha,
        fold,
        subset,
    )
    eval_meta = verify_sequence_cache(
        root,
        split,
        eval_path,
        eval_meta_path,
        "feature_only_eval",
        eval_refs,
        recipe,
        recipe_sha,
        fold,
        subset,
    )
    if fit_meta.get("read_contract") != {"targets": True, "scored_mask": True}:
        raise ValueError(f"F{fold_id} fit cache read contract is missing or changed")
    if eval_meta.get("read_contract") != {"targets": False, "scored_mask": False}:
        raise ValueError(f"F{fold_id} evaluation cache may have read target labels or masks")
    if eval_meta.get("zero_columns") != [112, 113, 114]:
        raise ValueError(f"F{fold_id} evaluation target and weight columns are not declared zero")
    if fit_meta.get("row_weight_contract") != ROW_WEIGHT_CONTRACT:
        raise ValueError(f"F{fold_id} fit cache row-weight arguments differ from the frozen method")
    if fit_meta.get("row_weight_batch_semantics") != (
        "one complete sequence per main_row_weights call; true-mask mass scale is sequence-local"
    ):
        raise ValueError(f"F{fold_id} fit cache row-weight normalization differs")
    if fit_meta.get("parent") != eval_meta.get("parent"):
        raise ValueError(f"F{fold_id} fit and evaluation parent identities differ")
    parent_identity = eval_meta.get("parent", {})
    if parent_identity.get("endpoint_precision") != "strict_fp32" or not parent_identity.get(
        "endpoint_backend"
    ):
        raise ValueError(f"F{fold_id} parent cache lacks strict-FP32 inference identity")

    prediction_path = cache_dir / "evaluation_predictions.npy"
    prediction_meta_path = cache_dir / "evaluation_predictions.json"
    prediction_meta = read_json(prediction_meta_path)
    prediction_sha = verify_parent_prediction_metadata(
        root,
        recipe,
        recipe_sha,
        fold,
        recipe_path,
        eval_refs,
        eval_meta_path,
        eval_meta,
        prediction_path,
        prediction_meta_path,
        prediction_meta,
    )
    array, prediction_identity = prediction_memmap(
        prediction_path,
        (len(eval_refs), ROWS, 2),
        prediction_sha,
    )
    eval_cache = np.load(eval_path, mmap_mode="r", allow_pickle=False)
    for index in range(len(eval_refs)):
        row = eval_cache[index]
        if np.any(row[:, 112:115]):
            raise ValueError(f"F{fold_id} evaluation sequence cache contains labels or weights")
        if (
            not np.isfinite(row[:, :RAW_END]).all()
            or not np.isfinite(row[:, PARENT_START:PARENT_END]).all()
        ):
            raise ValueError(f"F{fold_id} evaluation sequence cache contains nonfinite features")
        if not np.array_equal(row[:, PARENT_START:PARENT_END], array[index]):
            raise ValueError(f"F{fold_id} standalone parent endpoint differs from sequence cache")
    parent = fold["parent"]
    _verify_parent_checkpoint(root, parent)
    identity = {
        "parent_run": parent["run"],
        "parent_epoch": parent["epoch"],
        "parent_checkpoint_sha256": parent["checkpoint_sha256"],
        "parent_provenance_sha256": parent["provenance_sha256"],
        "fit_cache_sha256": fit_meta["data_sha256"],
        "fit_cache_sidecar_sha256": sha256_file(fit_meta_path),
        "evaluation_cache_sha256": eval_meta["data_sha256"],
        "evaluation_cache_sidecar_sha256": sha256_file(eval_meta_path),
        "evaluation_prediction_sha256": prediction_identity["sha256"],
        "evaluation_prediction_sidecar_sha256": sha256_file(prediction_meta_path),
        "evaluation_prediction_path": str(prediction_path),
        "evaluation_prediction_sidecar_path": str(prediction_meta_path),
        "parent_endpoint_precision": parent_identity["endpoint_precision"],
        "parent_endpoint_backend": parent_identity["endpoint_backend"],
        "evaluation_refs_sha256": row_refs_sha256(eval_refs),
        "evaluation_groups": eval_groups,
        "cache_source_sha256": dict(sorted(eval_meta["source_sha256"].items())),
        "cache_directory": str(cache_dir),
    }
    del eval_cache
    return ParentCacheArtifacts(array, identity)


def verify_parent_prediction_metadata(
    root: Path,
    recipe: dict[str, Any],
    recipe_sha: str,
    fold: dict[str, Any],
    recipe_path: Path,
    refs: list[dict[str, Any]],
    eval_meta_path: Path,
    eval_meta: dict[str, Any],
    prediction_path: Path,
    prediction_meta_path: Path,
    metadata: dict[str, Any],
) -> str:
    fold_id = int(fold["fold"])
    if (
        metadata.get("schema") != "wnn.online-relationships.parent-predictions.v1"
        or metadata.get("scope") != "feature_only_eval"
        or metadata.get("recipe_id") != recipe["recipe_id"]
        or metadata.get("recipe_sha256") != recipe_sha
        or metadata.get("fold") != fold_id
        or metadata.get("seed") != fold["seed"]
        or metadata.get("parent") != fold["parent"]
        or metadata.get("parent", {}).get("run") != fold["parent"]["run"]
        or metadata.get("recipe") != str(recipe_path.resolve())
        or metadata.get("split_manifest_sha256") != fold["split_manifest_sha256"]
        or metadata.get("row_refs") != refs
        or metadata.get("membership_sha256") != row_refs_sha256(refs)
        or metadata.get("evaluation_cache_sidecar") != str(eval_meta_path)
        or metadata.get("evaluation_cache_sidecar_sha256") != sha256_file(eval_meta_path)
        or metadata.get("prediction_file") != prediction_path.name
        or metadata.get("prediction_path") != str(prediction_path)
        or metadata.get("shape") != [len(refs), ROWS, 2]
        or metadata.get("dtype") != "<f4"
        or metadata.get("sampled_profile") is not False
        or eval_meta.get("parent_prediction_file") != prediction_path.name
        or metadata.get("endpoint")
        != {
            "precision": "strict_fp32",
            "backend": eval_meta.get("parent", {}).get("endpoint_backend"),
        }
    ):
        raise ValueError(f"F{fold_id} standalone parent prediction metadata is inconsistent")
    prediction_sha = metadata.get("prediction_sha256")
    if (
        not isinstance(prediction_sha, str)
        or eval_meta.get("parent_prediction_sha256") != prediction_sha
    ):
        raise ValueError(f"F{fold_id} standalone parent prediction SHA256 differs")
    return prediction_sha


def _verify_parent_checkpoint(root: Path, parent: dict[str, Any]) -> None:
    checkpoint = _resolve(root, parent["checkpoint"])
    provenance = _resolve(root, parent["provenance"])
    _require_sha(checkpoint, parent["checkpoint_sha256"], "parent checkpoint")
    _require_sha(provenance, parent["provenance_sha256"], "parent provenance")
