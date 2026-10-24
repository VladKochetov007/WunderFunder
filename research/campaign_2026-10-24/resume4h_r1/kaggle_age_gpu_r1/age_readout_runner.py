"""Profile and fit one fold of the six-cell matched age-readout screen."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .age_readout_artifacts import PredictionSink, publish_head_checkpoints, publish_json
from .age_readout_cache import load_profile_caches, load_validated_caches
from .age_readout_fit import ARMS, fit_panel_fold
from .age_readout_portable import (
    PortablePaths,
    close_portable_caches,
    load_portable_caches,
    remove_portable_cache,
)
from .age_readout_portable_builder import release_sequence_pages
from .age_readout_portable_identity import update_disk_peak
from .age_readout_provenance import (
    build_identity,
    build_prediction_sidecar,
    build_provenance,
    runtime_identity,
)
from .age_readout_recipe import find_cell


def run_profile(
    repo_root: Path,
    recipe: dict[str, Any],
    recipe_sha: str,
    fold: dict[str, Any],
    device: str,
    profile_root: Path,
    portable_paths: PortablePaths | None = None,
) -> dict[str, object]:
    """Run the admitted 16-fit/8-evaluation sequence CPU/CUDA timing sample."""
    fold_id = int(fold["fold"])
    seed = int(fold["seed"])
    started = time.perf_counter()
    if portable_paths is None:
        fit_cache, eval_cache, cache_receipt = load_profile_caches(repo_root, recipe, fold, fold_id)
    else:
        fit_cache, eval_cache, cache_receipt = load_portable_caches(
            repo_root, recipe, recipe_sha, fold, fold_id, portable_paths
        )
    load_seconds = time.perf_counter() - started
    try:
        disk_peak = update_disk_peak(None, portable_paths) if portable_paths is not None else None
        identity = build_identity(
            repo_root, recipe, recipe_sha, fold, fit_cache, cache_receipt, device
        )
        result = fit_panel_fold(
            fit_cache.array,
            eval_cache.array,
            np.asarray(fit_cache.metadata["parent_mean"], dtype="<f4"),
            np.asarray(fit_cache.metadata["parent_std"], dtype="<f4"),
            fold=fold_id,
            seed=seed,
            device_name=device,
            identity=identity,
            checkpoint_dir=None,
            max_fit_sequences=16,
            max_eval_sequences=8,
            release_fit_sequence=(
                None
                if portable_paths is None
                else lambda index: release_sequence_pages(fit_cache.array, index)
            ),
            release_eval_sequence=(
                None
                if portable_paths is None
                else lambda index: release_sequence_pages(eval_cache.array, index)
            ),
        )
        receipt: dict[str, object] = {
            "schema": "wnn.age-readout-profile.v1",
            "recipe_sha256": recipe_sha,
            "fold": fold_id,
            "seed": seed,
            "readout_device": device,
            "parent_endpoint_backend": "cpu",
            "fit_sequences": 16,
            "evaluation_sequences": 8,
            "evaluation_targets_read": False,
            "evaluation_scoring_masks_read": False,
            "payload_hash_verified": bool(cache_receipt["payload_hash_verified"]),
            "cache_open_seconds": load_seconds,
            "head_fit_seconds": result["fit_seconds"],
            "candidate_prediction_seconds": result["eval_seconds"],
            "total_profile_seconds": time.perf_counter() - started,
            "phase_scope": "full fold rebuild; 16 fit and 8 target-free evaluation sequences; all heads; 3 epochs",
            "source_cache": cache_receipt,
            "live_disk_peak": disk_peak,
            "runtime": runtime_identity(torch, device),
        }
        profile_root.mkdir(parents=True, exist_ok=True)
        publish_json(receipt, profile_root / f"profile_f{fold_id}_{device.replace(':', '_')}.json")
        return receipt
    finally:
        if portable_paths is not None:
            close_portable_caches(fit_cache, eval_cache)
            remove_portable_cache(portable_paths.cache_dir(fold_id))


def run_fit_fold(
    repo_root: Path,
    recipe: dict[str, Any],
    recipe_sha: str,
    fold: dict[str, Any],
    device: str,
    portable_paths: PortablePaths | None = None,
) -> dict[str, object]:
    """Validate once, fit all three heads in one cache pass per epoch, publish cells."""
    fold_id = int(fold["fold"])
    started = time.perf_counter()
    if portable_paths is None:
        fit_cache, eval_cache, cache_receipt = load_validated_caches(
            repo_root, recipe, fold, fold_id
        )
        release_fit = None
        release_eval = None
    else:
        if device != "cpu":
            raise ValueError("portable Kaggle reconstruction is CPU-only")
        fit_cache, eval_cache, cache_receipt = load_portable_caches(
            repo_root, recipe, recipe_sha, fold, fold_id, portable_paths
        )

        def _release_fit(index: int) -> None:
            release_sequence_pages(fit_cache.array, index)

        def _release_eval(index: int) -> None:
            release_sequence_pages(eval_cache.array, index)

        release_fit, release_eval = _release_fit, _release_eval

    validation_seconds = time.perf_counter() - started
    try:
        return _fit_loaded_fold(
            repo_root,
            recipe,
            recipe_sha,
            fold,
            device,
            fit_cache,
            eval_cache,
            cache_receipt,
            validation_seconds,
            portable_paths,
            release_fit,
            release_eval,
        )
    finally:
        if portable_paths is not None:
            close_portable_caches(fit_cache, eval_cache)
            if portable_paths.cleanup_after_success:
                remove_portable_cache(portable_paths.cache_dir(fold_id))


def _fit_loaded_fold(
    repo_root: Path,
    recipe: dict[str, Any],
    recipe_sha: str,
    fold: dict[str, Any],
    device: str,
    fit_cache: Any,
    eval_cache: Any,
    cache_receipt: dict[str, object],
    validation_seconds: float,
    portable_paths: PortablePaths | None,
    release_fit: Any,
    release_eval: Any,
) -> dict[str, object]:
    """Fit and publish one complete fold from already validated cache mappings."""
    fold_id = int(fold["fold"])
    seed = int(fold["seed"])
    identity = build_identity(repo_root, recipe, recipe_sha, fold, fit_cache, cache_receipt, device)
    if portable_paths is None:
        output_dirs = {arm: Path(find_cell(recipe, fold_id, arm)["output"]) for arm in ARMS}
        resume_root = (
            Path(recipe["storage"]["new_root"]) / "b300_scout" / "resume" / f"fold{fold_id}"
        )
    else:
        output_dirs = {
            arm: portable_paths.output_root / find_cell(recipe, fold_id, arm)["run"] for arm in ARMS
        }
        resume_root = portable_paths.resume_root / f"fold{fold_id}"
    checkpoint_dir = resume_root / "checkpoints"
    checkpoints = sorted(
        checkpoint_dir.glob("epoch*.pt"),
        key=lambda path: int(path.stem.removeprefix("epoch")),
    )
    resume_from = checkpoints[-1] if checkpoints else None
    progress_dir = resume_root.parent / "progress"
    progress_dir.mkdir(parents=True, exist_ok=True)
    progress_path = progress_dir / f"fold{fold_id}.jsonl"
    progress_mode = "a" if progress_path.exists() else "x"
    sink = PredictionSink(output_dirs, len(eval_cache.row_refs))
    disk_peak = {
        "value": (
            update_disk_peak(cache_receipt.get("live_disk"), portable_paths)
            if portable_paths is not None
            else None
        )
    }
    mean = np.asarray(fit_cache.metadata["parent_mean"], dtype="<f4")
    std = np.asarray(fit_cache.metadata["parent_std"], dtype="<f4")
    with progress_path.open(progress_mode, encoding="utf-8") as stream:

        def _progress(record: dict[str, object]) -> None:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            if portable_paths is not None and isinstance(record.get("arms"), dict):
                disk_peak["value"] = update_disk_peak(disk_peak["value"], portable_paths)

        result = fit_panel_fold(
            fit_cache.array,
            eval_cache.array,
            mean,
            std,
            fold=fold_id,
            seed=seed,
            device_name=device,
            identity=identity,
            checkpoint_dir=checkpoint_dir,
            resume_from=resume_from,
            progress=_progress,
            prediction=sink.write,
            release_fit_sequence=release_fit,
            release_eval_sequence=release_eval,
        )
    prediction_artifacts = sink.publish()
    checkpoint_artifacts = publish_head_checkpoints(
        result["models"], output_dirs, identity=identity, parent_mean=mean, parent_std=std
    )
    for arm in ARMS:
        cell = find_cell(recipe, fold_id, arm)
        provenance = build_provenance(
            recipe=recipe,
            cell=cell,
            fold=fold,
            arm=arm,
            fit_cache=fit_cache,
            eval_cache=eval_cache,
            cache_receipt=cache_receipt,
            checkpoint=checkpoint_artifacts[arm],
            prediction=prediction_artifacts[arm],
            identity=identity,
            runtime=runtime_identity(torch, device),
            fit_records=result["records"],
        )
        provenance_path = output_dirs[arm] / "provenance.json"
        provenance_sha = publish_json(provenance, provenance_path)
        prediction_sidecar = build_prediction_sidecar(
            recipe=recipe,
            cell=cell,
            fold=fold,
            arm=arm,
            recipe_sha=recipe_sha,
            cache_receipt=cache_receipt,
            eval_cache=eval_cache,
            prediction=prediction_artifacts[arm],
            checkpoint=checkpoint_artifacts[arm],
            provenance_path=provenance_path,
            provenance_sha=provenance_sha,
        )
        publish_json(prediction_sidecar, output_dirs[arm] / "prediction.json")
    if portable_paths is not None:
        disk_peak["value"] = update_disk_peak(disk_peak["value"], portable_paths)
    return {
        "schema": "wnn.age-readout-fold-run.v1",
        "recipe_sha256": recipe_sha,
        "fold": fold_id,
        "seed": seed,
        "readout_device": device,
        "parent_endpoint_backend": "cpu",
        "cache_validation_seconds": validation_seconds,
        "fit_seconds": result["fit_seconds"],
        "evaluation_seconds": result["eval_seconds"],
        "fit_membership": len(fit_cache.row_refs),
        "evaluation_membership": len(eval_cache.row_refs),
        "cell_outputs": {arm: str(output_dirs[arm]) for arm in ARMS},
        "source_cache": cache_receipt,
        "identity": identity,
        "runtime": runtime_identity(torch, device),
        "live_disk_peak": disk_peak["value"],
    }
