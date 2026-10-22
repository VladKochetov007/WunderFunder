"""Validate online readout cache membership and frozen parent ancestry."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from wnn.splits import file_sha256, load_campaign_split
from wnn.training.membership import load_component_fit_subset
from wnn.training.online_relationship_cache import (
    SequenceCache,
    load_sequence_cache,
    sequence_cache_paths,
)
from wnn.training.online_relationship_data_writer import representative_refs
from wnn.training.online_relationship_sources import (
    check_hash,
    module_hashes,
    read_json,
    resolve,
    tensor_bytes_sha,
)


def load_fold_caches(
    root: Path,
    recipe: dict[str, Any],
    recipe_sha: str,
    fold: dict[str, Any],
    device: str,
    *,
    cache_directory: Path | None = None,
    sample_fit: int | None = None,
    sample_eval: int | None = None,
) -> tuple[SequenceCache, SequenceCache, dict[str, Any]]:
    """Validate full fit/evaluation memberships and parent-prediction ancestry."""
    if (sample_fit is None) != (sample_eval is None):
        raise ValueError("profile cache validation needs both sample counts")
    sampled = sample_fit is not None
    directory = cache_directory or resolve(root, Path(fold["parent_cache"]))
    fit_data, fit_sidecar = sequence_cache_paths(directory, "in_sample_fit")
    eval_data, eval_sidecar = sequence_cache_paths(directory, "feature_only_eval")
    fit_cache = load_sequence_cache(
        fit_data,
        fit_sidecar,
        expected_scope="in_sample_fit",
        expected_recipe_sha256=recipe_sha,
        expected_fold=int(fold["fold"]),
    )
    eval_cache = load_sequence_cache(
        eval_data,
        eval_sidecar,
        expected_scope="feature_only_eval",
        expected_recipe_sha256=recipe_sha,
        expected_fold=int(fold["fold"]),
    )
    if fit_cache.metadata.get("complete_membership") != (not sampled) or eval_cache.metadata.get(
        "complete_membership"
    ) != (not sampled):
        raise ValueError("cache completeness differs from the requested profile or full-run scope")
    if (
        fit_cache.metadata.get("sampled_profile") != sampled
        or eval_cache.metadata.get("sampled_profile") != sampled
    ):
        raise ValueError("cache profile flags differ from the requested profile or full-run scope")
    expected_sources = module_hashes(root)
    for cache in (fit_cache, eval_cache):
        if cache.metadata.get("source_sha256") != expected_sources:
            raise ValueError("cache source hashes differ from the current input and cache code")
    split_path = resolve(root, Path(fold["split_manifest"]))
    subset_path = resolve(root, Path(fold["subset"]["path"]))
    check_hash(split_path, fold["split_manifest_sha256"], "campaign split")
    check_hash(subset_path, fold["subset"]["sha256"], "fit subset")
    campaign = load_campaign_split(split_path, root)
    if (
        campaign.sha256 != fold["split_manifest_sha256"]
        or campaign.fold != int(fold["fold"])
        or list(campaign.evaluation_groups) != list(fold["evaluation_groups"])
    ):
        raise ValueError("campaign split identity or evaluation order differs from the recipe")
    subset = load_component_fit_subset(
        subset_path, campaign, expected_experiment_id=recipe["experiment_id"]
    )
    for cache in (fit_cache, eval_cache):
        if (
            cache.metadata.get("split_manifest_sha256") != fold["split_manifest_sha256"]
            or cache.metadata.get("subset_manifest_sha256") != fold["subset"]["sha256"]
            or cache.metadata.get("subset_members_sha256")
            != subset["manifest"]["selected_members_sha256"]
        ):
            raise ValueError("cache sidecar split or subset identity differs from the recipe")
    fit_refs = [
        {"split": source, "row_group": int(group)}
        for source, groups in (("train", subset["train_groups"]), ("valid", subset["valid_groups"]))
        for group in groups
    ]
    eval_refs = [
        {"split": "valid", "row_group": int(group)} for group in campaign.evaluation_groups
    ]
    if sampled:
        fit_refs = representative_refs(fit_refs, int(sample_fit))
        eval_refs = representative_refs(eval_refs, int(sample_eval))
    if list(fit_cache.row_refs) != fit_refs or list(eval_cache.row_refs) != eval_refs:
        raise ValueError("cache row order differs from frozen fit/evaluation memberships")
    if fit_cache.metadata.get("row_weight_contract") != recipe["readout"]["row_weight_args"]:
        raise ValueError("fit cache row weights differ from the frozen recipe")
    if (
        fit_cache.metadata.get("row_weight_batch_semantics")
        != recipe["readout"]["row_weight_batch_semantics"]
    ):
        raise ValueError("fit cache batch weighting semantics differ from the frozen recipe")
    parent_prediction_path = directory / "evaluation_predictions.npy"
    parent_manifest_path = directory / "evaluation_predictions.json"
    parent_manifest = read_json(parent_manifest_path)
    parent_prediction_sha = file_sha256(parent_prediction_path)
    if (
        parent_manifest.get("scope") != "feature_only_eval"
        or parent_manifest.get("prediction_file") != parent_prediction_path.name
        or resolve(root, Path(parent_manifest["prediction_path"]))
        != parent_prediction_path.resolve()
    ):
        raise ValueError("parent prediction file or scope differs from the frozen cache contract")
    if parent_manifest.get("prediction_sha256") != parent_prediction_sha:
        raise ValueError("parent evaluation prediction hash differs from its sidecar")
    if parent_manifest.get("recipe_sha256") != recipe_sha:
        raise ValueError("parent evaluation predictions belong to a different recipe")
    if parent_manifest.get("sampled_profile") != sampled:
        raise ValueError("parent endpoint profile flag differs from its sequence caches")
    if parent_manifest.get("row_refs") != eval_refs:
        raise ValueError("parent evaluation prediction order differs from its cache")
    if (
        parent_manifest.get("shape") != [len(eval_refs), 20_000, 2]
        or parent_manifest.get("dtype") != "<f4"
    ):
        raise ValueError("parent evaluation sidecar has a different prediction header")
    if parent_manifest.get("evaluation_cache_sidecar_sha256") != file_sha256(eval_sidecar):
        raise ValueError("parent endpoint does not bind this evaluation cache sidecar")
    if resolve(root, Path(parent_manifest["evaluation_cache_sidecar"])) != eval_sidecar.resolve():
        raise ValueError("parent endpoint references a different evaluation cache sidecar")
    prediction = np.load(parent_prediction_path, mmap_mode="r", allow_pickle=False)
    if prediction.shape != (len(eval_refs), 20_000, 2) or prediction.dtype != np.dtype("<f4"):
        raise ValueError("parent evaluation endpoint has an unexpected NPY header")
    for index, sequence in enumerate(eval_cache.iter_sequences()):
        if not np.array_equal(sequence.parent_prediction, prediction[index]):
            raise ValueError("parent endpoint differs from evaluation cache predictions")
    del prediction
    if fit_cache.metadata.get("scope") != "in_sample_fit":
        raise ValueError("parent fit predictions must be identified as in-sample")
    if parent_manifest.get("parent") != fold["parent"]:
        raise ValueError("parent endpoint identity differs from the frozen parent recipe")
    expected_endpoint = {"precision": "strict_fp32", "backend": str(torch.device(device))}
    if parent_manifest.get("endpoint") != expected_endpoint:
        raise ValueError("parent evaluation endpoint precision or backend differs from the fit")
    parent_identity = fit_cache.metadata.get("parent")
    expected_backend = str(torch.device(device))
    if not isinstance(parent_identity, dict):
        raise ValueError("fit cache lacks parent endpoint identity")
    expected_parent = fold["parent"]
    if any(
        parent_identity.get(cache_key) != expected_parent.get(recipe_key)
        for cache_key, recipe_key in (
            ("run", "run"),
            ("epoch", "epoch"),
            ("checkpoint_sha256", "checkpoint_sha256"),
            ("provenance_sha256", "provenance_sha256"),
            ("mean_sha256", "mean_identity"),
            ("std_sha256", "std_identity"),
        )
    ):
        raise ValueError(
            "fit-cache parent checkpoint or normalization identity differs from recipe"
        )
    if parent_identity.get("checkpoint") != str(
        resolve(root, Path(expected_parent["checkpoint"]))
    ) or parent_identity.get("provenance") != str(
        resolve(root, Path(expected_parent["provenance"]))
    ):
        raise ValueError("fit-cache parent file paths differ from the frozen recipe")
    parent_mean = np.asarray(fit_cache.metadata.get("parent_mean"), dtype=np.float32)
    parent_std = np.asarray(fit_cache.metadata.get("parent_std"), dtype=np.float32)
    if (
        parent_mean.shape != (211,)
        or parent_std.shape != (211,)
        or tensor_bytes_sha(torch.from_numpy(parent_mean)) != expected_parent["mean_identity"]
        or tensor_bytes_sha(torch.from_numpy(parent_std)) != expected_parent["std_identity"]
    ):
        raise ValueError("fit-cache normalization bytes differ from the frozen parent identity")
    if (
        eval_cache.metadata.get("parent_mean") != parent_mean.tolist()
        or eval_cache.metadata.get("parent_std") != parent_std.tolist()
    ):
        raise ValueError("fit and evaluation caches use different parent normalization arrays")
    if parent_identity.get("endpoint_precision") != "strict_fp32":
        raise ValueError("parent predictions were not generated by strict FP32 inference")
    if parent_identity.get("endpoint_backend") != expected_backend:
        raise ValueError("parent-cache and readout inference backends must match exactly")
    if eval_cache.metadata.get("parent") != parent_identity:
        raise ValueError("fit and evaluation parent predictions have different inference ancestry")
    propensity = fit_cache.metadata.get("propensity")
    expected_propensity = fold["canonical_propensity"]
    if not isinstance(propensity, dict) or any(
        propensity.get(key) != expected_propensity.get(key)
        for key in ("run", "checkpoint_sha256", "provenance_sha256")
    ):
        raise ValueError("fit cache propensity ancestry differs from the frozen recipe")
    parent_manifest["manifest_path"] = str(parent_manifest_path)
    parent_manifest["manifest_sha256"] = file_sha256(parent_manifest_path)
    parent_manifest["prediction_path"] = str(parent_prediction_path)
    return fit_cache, eval_cache, parent_manifest
