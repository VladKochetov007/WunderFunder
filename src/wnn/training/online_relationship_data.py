"""Build parent-prediction and target-safe sequence caches for online readouts."""

from __future__ import annotations

import gc
import logging
import time
from pathlib import Path

from wnn.splits import file_sha256, load_campaign_split
from wnn.training.membership import load_component_fit_subset
from wnn.training.online_relationship_cache import sequence_cache_paths
from wnn.training.online_relationship_data_writer import (
    parent_prediction_metadata,
    publish_create_only,
    representative_refs,
    temporary_path,
    write_json_to_temp,
    write_scope,
)
from wnn.training.online_relationship_sources import (
    check_hash,
    dataset_identities,
    find_fold,
    load_fresh_propensity,
    load_parent,
    read_json,
    resolve,
)

ROW_WEIGHT_ARGS = {
    "mask_weight": False,
    "prop_power": 2.0,
    "prop_eps": 0.03,
    "prop_cache": False,
    "class_weight": None,
    "valid_true_mask": 0.03,
    "sync_free": False,
    "since_weight": 0.0,
    "since_floor": 0.1,
    "episode_weight": 0.0,
    "episode_floor": 0.1,
}
_LOG = logging.getLogger(__name__)


def build_parent_caches(
    recipe_path: Path,
    fold_index: int,
    *,
    root: Path,
    device: str = "cpu",
    cache_dir: Path | None = None,
    sample_fit: int | None = None,
    sample_eval: int | None = None,
) -> dict[str, object]:
    """Create immutable per-sequence caches with one source sequence resident at a time.

    Evaluation row groups are opened with targets and score masks disabled. The
    two unused cache columns remain exactly zero, and the separate parent
    endpoint contains only the ordered predictions and source references.
    """
    root = root.resolve()
    recipe_path = resolve(root, recipe_path)
    recipe = read_json(recipe_path)
    if (
        recipe.get("schema_version") != 1
        or recipe.get("resource_contract", {}).get("total_new_fits") != 12
    ):
        raise ValueError("recipe is not the frozen 12-fit online relationship panel")
    readout = recipe.get("readout", {})
    if readout.get("row_weight_args") != ROW_WEIGHT_ARGS:
        raise ValueError("recipe row_weight_args differ from the approved common contract")
    if readout.get("row_weight_batch_semantics") != (
        "one complete sequence per main_row_weights call; true-mask mass scale is sequence-local"
    ):
        raise ValueError("recipe does not declare complete-sequence row-weight semantics")
    fold = find_fold(recipe, fold_index)
    if (sample_fit is None) != (sample_eval is None):
        raise ValueError("sample_fit and sample_eval must be supplied together")
    sample_mode = sample_fit is not None
    destination = (cache_dir or Path(fold["parent_cache"])).resolve()
    if not destination.is_relative_to(Path(recipe["resource_contract"]["scratch"]).resolve()):
        raise ValueError("online relationship caches must be under physical scratch")

    phase_seconds: dict[str, float] = {}
    _LOG.info("online_relationship_cache fold=%d phase=validate_membership", fold_index)
    phase_start = time.perf_counter()
    split_path = resolve(root, Path(fold["split_manifest"]))
    subset_path = resolve(root, Path(fold["subset"]["path"]))
    check_hash(split_path, fold["split_manifest_sha256"], "campaign split")
    check_hash(subset_path, fold["subset"]["sha256"], "fit subset")
    campaign = load_campaign_split(split_path, root)
    subset = load_component_fit_subset(
        subset_path, campaign, expected_experiment_id=recipe["experiment_id"]
    )
    if campaign.fold != fold_index or campaign.sha256 != fold["split_manifest_sha256"]:
        raise ValueError("cache fold or split identity differs from the recipe")
    if list(campaign.evaluation_groups) != list(fold["evaluation_groups"]):
        raise ValueError("evaluation row-group order differs from the frozen recipe")
    fit_refs = [
        {"split": split, "row_group": int(group)}
        for split, groups in (("train", subset["train_groups"]), ("valid", subset["valid_groups"]))
        for group in groups
    ]
    eval_refs = [
        {"split": "valid", "row_group": int(group)} for group in campaign.evaluation_groups
    ]
    if sample_mode:
        fit_refs = representative_refs(fit_refs, int(sample_fit))
        eval_refs = representative_refs(eval_refs, int(sample_eval))
        destination = destination / f"profile_fit{sample_fit}_eval{sample_eval}"
    phase_seconds["membership_validation"] = time.perf_counter() - phase_start

    recipe_sha = file_sha256(recipe_path)
    _LOG.info("online_relationship_cache fold=%d phase=load_dependencies", fold_index)
    phase_start = time.perf_counter()
    propensity_model, propensity_identity = load_fresh_propensity(root, fold)
    parent_model, parent_environment, parent_identity, parent_mean, parent_std = load_parent(
        root, fold, device
    )
    phase_seconds["dependency_load"] = time.perf_counter() - phase_start
    fold_for_cache = {**fold, "_parent_endpoint_backend": parent_identity["endpoint_backend"]}
    source_identities = dataset_identities(root, split_path, campaign)
    destination.mkdir(parents=True, exist_ok=True)
    fit_data, fit_sidecar = sequence_cache_paths(destination, "in_sample_fit")
    eval_data, eval_sidecar = sequence_cache_paths(destination, "feature_only_eval")
    parent_data = destination / "evaluation_predictions.npy"
    parent_sidecar = destination / "evaluation_predictions.json"
    final_paths = (fit_data, fit_sidecar, eval_data, eval_sidecar, parent_data, parent_sidecar)
    if any(path.exists() for path in final_paths):
        raise FileExistsError("online parent cache outputs already exist; refusing to overwrite")

    temporary: list[Path] = []
    try:
        _LOG.info("online_relationship_cache fold=%d phase=write_fit_scope", fold_index)
        phase_start = time.perf_counter()
        fit_data_temp, _, fit_meta_temp, fit_open_seconds = write_scope(
            destination,
            fit_refs,
            "in_sample_fit",
            recipe,
            recipe_sha,
            fold_for_cache,
            subset,
            split_path,
            source_identities,
            propensity_identity,
            parent_identity,
            parent_mean,
            parent_std,
            parent_model,
            parent_environment,
            propensity_model,
            root,
            device,
            ROW_WEIGHT_ARGS,
            sample_mode,
        )
        phase_seconds["fit_cache"] = time.perf_counter() - phase_start
        phase_seconds["fit_cache_source_open"] = fit_open_seconds
        temporary.extend((fit_data_temp, fit_meta_temp))
        _LOG.info("online_relationship_cache fold=%d phase=write_evaluation_scope", fold_index)
        phase_start = time.perf_counter()
        eval_data_temp, parent_prediction_temp, eval_meta_temp, eval_open_seconds = write_scope(
            destination,
            eval_refs,
            "feature_only_eval",
            recipe,
            recipe_sha,
            fold_for_cache,
            subset,
            split_path,
            source_identities,
            propensity_identity,
            parent_identity,
            parent_mean,
            parent_std,
            parent_model,
            parent_environment,
            propensity_model,
            root,
            device,
            ROW_WEIGHT_ARGS,
            sample_mode,
            parent_prediction_path=True,
        )
        if parent_prediction_temp is None:
            raise RuntimeError("evaluation cache build omitted the parent prediction payload")
        phase_seconds["evaluation_cache"] = time.perf_counter() - phase_start
        phase_seconds["evaluation_cache_source_open"] = eval_open_seconds
        temporary.extend((eval_data_temp, parent_prediction_temp, eval_meta_temp))
        _LOG.info("online_relationship_cache fold=%d phase=publish_cache", fold_index)
        phase_start = time.perf_counter()
        parent_meta_temp = temporary_path(destination, "evaluation_predictions.json")
        parent_meta = parent_prediction_metadata(
            parent_prediction_temp,
            parent_data,
            recipe,
            recipe_path,
            recipe_sha,
            fold_for_cache,
            eval_refs,
            eval_sidecar,
            file_sha256(eval_meta_temp),
            sample_mode,
        )
        write_json_to_temp(parent_meta_temp, parent_meta)
        temporary.append(parent_meta_temp)
        published = publish_create_only(
            (
                (fit_data_temp, fit_data),
                (fit_meta_temp, fit_sidecar),
                (eval_data_temp, eval_data),
                (eval_meta_temp, eval_sidecar),
                (parent_prediction_temp, parent_data),
                (parent_meta_temp, parent_sidecar),
            )
        )
        phase_seconds["publish_and_parent_sidecar"] = time.perf_counter() - phase_start
        temporary.clear()
        return {
            "schema": "wnn.online-relationships.parent-cache-build.v1",
            "fold": fold_index,
            "recipe_sha256": recipe_sha,
            "cache_directory": str(destination),
            "files": [str(path) for path in (*published, parent_data, parent_sidecar)],
            "fit_sequences": len(fit_refs),
            "evaluation_sequences": len(eval_refs),
            "scope": "profile_sample" if sample_mode else "full",
            "phase_seconds": phase_seconds,
        }
    finally:
        del parent_model, propensity_model
        gc.collect()
        for path in temporary:
            path.unlink(missing_ok=True)
