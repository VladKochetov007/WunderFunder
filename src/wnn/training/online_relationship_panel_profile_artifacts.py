"""Measure disposable profile exports and project full-panel phase costs."""

from __future__ import annotations

import json
import math
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from wnn.data import SEQUENCE_LENGTH
from wnn.splits import file_sha256
from wnn.training.online_relationship_cache import SequenceCache

_BATCH_ROWS = 8192


def time_profile_exports(output_dir: Path, run: str, result: Any) -> dict[str, object]:
    """Time disposable sample exports without creating candidate-role artifacts."""
    with tempfile.TemporaryDirectory(prefix="sample-export-", dir=output_dir) as directory:
        checkpoint = Path(directory) / "profile_checkpoint.pt"
        started = time.perf_counter()
        torch.save({"model": result.state_dict, "metadata": result.metadata}, checkpoint)
        checkpoint_sha = file_sha256(checkpoint)
        checkpoint_seconds = time.perf_counter() - started
        prediction = Path(directory) / "profile_predictions.npy"
        started = time.perf_counter()
        values = np.asarray(result.evaluation_predictions, dtype="<f4")
        if values.ndim != 3 or values.shape[1:] != (SEQUENCE_LENGTH, 2):
            raise ValueError("sample fitter returned malformed evaluation predictions")
        np.save(prediction, values, allow_pickle=False)
        prediction_sha = file_sha256(prediction)
        prediction_seconds = time.perf_counter() - started
        return {
            "run": run,
            "checkpoint_seconds": checkpoint_seconds,
            "checkpoint_bytes": checkpoint.stat().st_size,
            "checkpoint_sha256": checkpoint_sha,
            "prediction_seconds": prediction_seconds,
            "prediction_bytes": prediction.stat().st_size,
            "prediction_sha256": prediction_sha,
            "prediction_sequences": len(values),
        }


def cache_receipts(
    fit_cache: SequenceCache, eval_cache: SequenceCache, parent: dict[str, Any]
) -> dict[str, object]:
    """Record sampled cache payload and sidecar hashes for source review."""
    return {
        "fit": {
            "payload": str(fit_cache.data_path),
            "payload_sha256": fit_cache.metadata["data_sha256"],
            "sidecar": str(fit_cache.sidecar_path),
            "sidecar_sha256": file_sha256(fit_cache.sidecar_path),
            "membership_sha256": fit_cache.metadata["membership_sha256"],
        },
        "evaluation": {
            "payload": str(eval_cache.data_path),
            "payload_sha256": eval_cache.metadata["data_sha256"],
            "sidecar": str(eval_cache.sidecar_path),
            "sidecar_sha256": file_sha256(eval_cache.sidecar_path),
            "membership_sha256": eval_cache.metadata["membership_sha256"],
        },
        "parent_evaluation": {
            "payload": parent["prediction_path"],
            "payload_sha256": parent["prediction_sha256"],
            "sidecar": parent["manifest_path"],
            "sidecar_sha256": parent["manifest_sha256"],
        },
    }


def write_profile_report(path: Path, value: dict[str, object]) -> None:
    """Create and flush one immutable JSON profile receipt."""
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()


def project_full_panel(
    fold_data: dict[int, dict[str, Any]],
    normalizer_records: list[dict[str, Any]],
    cell_records: list[dict[str, Any]],
    export_seconds: dict[str, list[float]],
) -> dict[str, object]:
    """Scale measured phase times by complete recipe group and row-batch counts."""
    cache_seconds = 0.0
    normalizer_seconds = 0.0
    for data in fold_data.values():
        phases = data["cache_build"]["phase_seconds"]
        cache_seconds += sum(
            phases[key]
            for key in ("membership_validation", "dependency_load", "publish_and_parent_sidecar")
        )
        cache_seconds += (
            phases["fit_cache"] / data["cache_build"]["fit_sequences"] * data["fit_sequences"]
        )
        cache_seconds += (
            phases["evaluation_cache"]
            / data["cache_build"]["evaluation_sequences"]
            * data["evaluation_sequences"]
        )
    for record in normalizer_records:
        normalizer_seconds += (
            record["sample_seconds"] / record["sample_fit_sequences"] * record["full_fit_sequences"]
        )

    training_seconds = evaluation_seconds = 0.0
    cell_projections = []
    for record in cell_records:
        data = fold_data[record["fold"]]
        sample_batches = int(record["updates_per_epoch"])
        full_rows = data["fit_sequences"] * SEQUENCE_LENGTH
        full_batches = math.ceil(full_rows / _BATCH_ROWS)
        projected_fit = sum(record["epoch_seconds"]) * full_batches / sample_batches
        projected_eval = (
            record["evaluation_and_setup_seconds"]
            / record["evaluation_sequences"]
            * data["evaluation_sequences"]
        )
        training_seconds += projected_fit
        evaluation_seconds += projected_eval
        cell_projections.append(
            {
                "run": record["run"],
                "full_fit_rows_per_epoch": full_rows,
                "full_updates_per_epoch": full_batches,
                "sample_updates_per_epoch": sample_batches,
                "projected_three_epoch_fit_seconds": projected_fit,
                "projected_eval_and_setup_seconds": projected_eval,
            }
        )
    checkpoint_exports = sum(export_seconds["checkpoint"])
    prediction_exports = 0.0
    output_prediction_bytes = 0
    for record, sample_seconds in zip(cell_records, export_seconds["prediction"], strict=True):
        eval_count = fold_data[record["fold"]]["evaluation_sequences"]
        prediction_exports += sample_seconds / record["evaluation_sequences"] * eval_count
        output_prediction_bytes += eval_count * SEQUENCE_LENGTH * 2 * np.dtype("<f4").itemsize
    known_cost = (
        cache_seconds
        + normalizer_seconds
        + training_seconds
        + evaluation_seconds
        + checkpoint_exports
        + prediction_exports
    )
    return {
        "basis": "sample wall time scaled by full cache sequence counts, fit batches, and evaluation sequence counts",
        "cache_generation_seconds": cache_seconds,
        "fit_only_normalizer_seconds": normalizer_seconds,
        "three_epoch_readout_fit_seconds": training_seconds,
        "evaluation_prediction_seconds": evaluation_seconds,
        "sampled_checkpoint_export_seconds_for_all_cells": checkpoint_exports,
        "full_prediction_export_seconds": prediction_exports,
        "full_prediction_payload_bytes": output_prediction_bytes,
        "known_cost_sum_seconds": known_cost,
        "cell_projections": cell_projections,
    }
