"""Profile the complete online readout path on bounded source samples."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from wnn.online_relationship import fit_online_relationship_normalizer
from wnn.splits import file_sha256, load_campaign_split
from wnn.training.membership import load_component_fit_subset
from wnn.training.online_relationship_cache_validation import load_fold_caches
from wnn.training.online_relationship_data import build_parent_caches
from wnn.training.online_relationship_exports import canonical_json_sha256, source_hashes
from wnn.training.online_relationship_fit import fit_online_relationship_head
from wnn.training.online_relationship_panel_profile_artifacts import (
    cache_receipts,
    project_full_panel,
    time_profile_exports,
    write_profile_report,
)
from wnn.training.online_relationship_sources import find_fold, read_json, resolve

_LOG = logging.getLogger(__name__)
_LAGS = (1, 16)
_ARMS = ("A", "B", "C")
_SCRATCH = Path("/home/vlad/development/wunder/research_tmp").resolve()


def profile_online_relationship_panel(
    recipe_path: Path,
    output_dir: Path,
    *,
    root: Path,
    sample_fit: int = 4,
    sample_eval: int = 2,
) -> dict[str, object]:
    """Measure all twelve tiny-head fits and project full-panel phase costs on CPU.

    This profile uses real source groups, labels, weights, parents and readout
    code. It writes sampled caches under a separate profile directory, never
    emits candidate artifacts, scores no predictions, and does not change recipe
    admission status.
    """
    if sample_fit < 2 or sample_eval < 1:
        raise ValueError("profile requires at least two fit sequences and one evaluation sequence")
    root = root.resolve()
    recipe_path = resolve(root, recipe_path)
    output_dir = output_dir.resolve()
    if not output_dir.is_relative_to(_SCRATCH):
        raise ValueError(f"profile artifacts must be under physical scratch {_SCRATCH}")
    if output_dir.exists():
        raise FileExistsError(f"profile output is immutable and already exists: {output_dir}")
    recipe = read_json(recipe_path)
    if (
        recipe.get("schema_version") != 1
        or recipe.get("resource_contract", {}).get("total_new_fits") != 12
    ):
        raise ValueError("recipe is not the frozen 12-fit online relationship panel")
    output_dir.mkdir(parents=True)
    started = time.perf_counter()
    _LOG.info("online_relationship_profile phase=prepare_real_sample_caches")
    fold_data = {}
    for fold_index in (0, 1):
        fold = find_fold(recipe, fold_index)
        cache_build = build_parent_caches(
            recipe_path,
            fold_index,
            root=root,
            device="cpu",
            cache_dir=output_dir / f"parent_cache_f{fold_index}",
            sample_fit=sample_fit,
            sample_eval=sample_eval,
        )
        cache_dir = Path(str(cache_build["cache_directory"]))
        fit_cache, eval_cache, parent_manifest = load_fold_caches(
            root,
            recipe,
            file_sha256(recipe_path),
            fold,
            "cpu",
            cache_directory=cache_dir,
            sample_fit=sample_fit,
            sample_eval=sample_eval,
        )
        campaign = load_campaign_split(resolve(root, Path(fold["split_manifest"])), root)
        subset = load_component_fit_subset(
            resolve(root, Path(fold["subset"]["path"])),
            campaign,
            expected_experiment_id=recipe["experiment_id"],
        )
        fold_data[fold_index] = {
            "fold": fold,
            "cache_build": cache_build,
            "fit_cache": fit_cache,
            "eval_cache": eval_cache,
            "parent_manifest": parent_manifest,
            "fit_sequences": len(subset["train_groups"]) + len(subset["valid_groups"]),
            "evaluation_sequences": len(campaign.evaluation_groups),
            "cache_receipts": cache_receipts(fit_cache, eval_cache, parent_manifest),
        }

    normalizer_records, cell_records, export_seconds = _profile_heads(
        fold_data, recipe, output_dir, sample_fit, sample_eval
    )

    projection = project_full_panel(fold_data, normalizer_records, cell_records, export_seconds)
    source_manifest = source_hashes(root)
    profile_seconds = time.perf_counter() - started
    report = {
        "schema": "wnn.online-relationships.panel-cost-profile.v1",
        "profile_only": True,
        "recipe_id": recipe["recipe_id"],
        "recipe_path": str(recipe_path),
        "recipe_sha256": file_sha256(recipe_path),
        "profile_directory": str(output_dir),
        "device": "cpu",
        "precision": "strict_fp32",
        "sample": {
            "fit_sequences_per_fold": sample_fit,
            "evaluation_sequences_per_fold": sample_eval,
        },
        "source_sha256": source_manifest,
        "fit_membership_scope": "sampled real fitting groups, including train and fit-valid sources",
        "evaluation_scope": "sampled complete outer-fold groups; labels and masks were not read",
        "sample_fit_targets_used_for_loss_and_weights": True,
        "evaluation_targets_read": False,
        "target_scoring_or_quality_selection": "not run",
        "recipe_admission_mutated": False,
        "cache_builds": [data["cache_build"] for data in fold_data.values()],
        "full_scope_counts": [
            {
                "fold": fold_index,
                "fit_sequences": data["fit_sequences"],
                "evaluation_sequences": data["evaluation_sequences"],
            }
            for fold_index, data in fold_data.items()
        ],
        "cache_receipts": [data["cache_receipts"] for data in fold_data.values()],
        "normalizers": normalizer_records,
        "head_fits": cell_records,
        "measured_profile_wall_seconds": profile_seconds,
        "projection": projection,
        "unmeasured_delivery_costs": [
            "full target grading and grader runtime",
            "production provenance and sidecar serialization beyond the sampled export analogue",
            "thermal scheduling overhead after this CPU-only process exits",
        ],
        "admission": "profile evidence only; root must combine with remaining delivery costs",
    }
    write_profile_report(output_dir / "profile.json", report)
    return report


def _profile_heads(
    fold_data: dict[int, dict[str, Any]],
    recipe: dict[str, Any],
    output_dir: Path,
    sample_fit: int,
    sample_eval: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[float]]]:
    """Fit sample-only scalers and all twelve heads while timing every epoch."""
    normalizers = {}
    normalizer_records = []
    for fold_index, data in fold_data.items():
        for lag in _LAGS:
            _LOG.info(
                "online_relationship_profile phase=fit_sample_normalizer fold=%s lag=%s",
                fold_index,
                lag,
            )
            phase_start = time.perf_counter()
            normalizer = fit_online_relationship_normalizer(
                (item.raw for item in data["fit_cache"].iter_sequences()), lag
            )
            normalizer_records.append(
                {
                    "fold": fold_index,
                    "lag": lag,
                    "normalizer_sha256": canonical_json_sha256(normalizer.to_dict()),
                    "sample_seconds": time.perf_counter() - phase_start,
                    "sample_fit_sequences": sample_fit,
                    "full_fit_sequences": data["fit_sequences"],
                }
            )
            normalizers[(fold_index, lag)] = normalizer

    cell_records = []
    export_seconds = {"checkpoint": [], "prediction": []}
    for fold_index, data in fold_data.items():
        for lag in _LAGS:
            for arm in _ARMS:
                record, exports = _profile_one_cell(
                    fold_index,
                    lag,
                    arm,
                    data,
                    recipe,
                    output_dir,
                    sample_fit,
                    sample_eval,
                    normalizers[(fold_index, lag)],
                )
                cell_records.append(record)
                export_seconds["checkpoint"].append(exports["checkpoint_seconds"])
                export_seconds["prediction"].append(exports["prediction_seconds"])
    return normalizer_records, cell_records, export_seconds


def _profile_one_cell(
    fold_index: int,
    lag: int,
    arm: str,
    data: dict[str, Any],
    recipe: dict[str, Any],
    output_dir: Path,
    sample_fit: int,
    sample_eval: int,
    normalizer: Any,
) -> tuple[dict[str, Any], dict[str, object]]:
    """Fit and disposable-export one sample head, returning phase timings."""
    fold = data["fold"]
    run = _cell(recipe, fold_index, lag, arm)["run"]
    _LOG.info("online_relationship_profile phase=fit_sample_head run=%s", run)
    timer = _EpochTimer(run)
    fit_start = time.perf_counter()
    timer.last = fit_start
    result = fit_online_relationship_head(
        data["fit_cache"].iter_sequences,
        data["eval_cache"].iter_sequences,
        arm=arm,
        lag=lag,
        seed=int(fold["seed"]),
        parent_mean=np.asarray(data["fit_cache"].metadata["parent_mean"], dtype=np.float32),
        parent_std=np.asarray(data["fit_cache"].metadata["parent_std"], dtype=np.float32),
        relationship_normalizer=normalizer,
        device="cpu",
        progress=timer,
    )
    fit_and_eval_seconds = time.perf_counter() - fit_start
    exports = time_profile_exports(output_dir, run, result)
    record = {
        "run": run,
        "fold": fold_index,
        "lag": lag,
        "arm": arm,
        "fit_sequences": sample_fit,
        "evaluation_sequences": sample_eval,
        "fit_rows_per_epoch": result.epoch_records[0]["fit_rows"],
        "updates_per_epoch": result.epoch_records[0]["updates"],
        "epoch_seconds": timer.durations,
        "evaluation_and_setup_seconds": max(0.0, fit_and_eval_seconds - sum(timer.durations)),
        "fit_and_eval_seconds": fit_and_eval_seconds,
        "sample_export": exports,
    }
    del result
    return record, exports


class _EpochTimer:
    """Capture elapsed wall time at each completed sample-fit epoch."""

    def __init__(self, run: str) -> None:
        self.run = run
        self.last = time.perf_counter()
        self.durations: list[float] = []

    def __call__(self, record: dict[str, object]) -> None:
        """Record an epoch boundary and flush the run's progress event."""
        now = time.perf_counter()
        self.durations.append(now - self.last)
        self.last = now
        _LOG.info(
            "online_relationship_profile_epoch %s",
            json.dumps({"run": self.run, **record}, sort_keys=True),
        )


def _cell(recipe: dict[str, Any], fold: int, lag: int, arm: str) -> dict[str, Any]:
    matches = [
        cell
        for cell in recipe["cells"]
        if int(cell["fold"]) == fold and int(cell["lag"]) == lag and cell["arm"] == arm
    ]
    if len(matches) != 1:
        raise ValueError(f"recipe must define one fold={fold}, lag={lag}, arm={arm} profile cell")
    return matches[0]
