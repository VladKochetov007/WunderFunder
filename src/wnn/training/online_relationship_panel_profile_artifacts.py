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
from torch.nn.utils import clip_grad_norm_

from wnn.conditional_readout import primary_pair_wmse
from wnn.data import SEQUENCE_LENGTH
from wnn.online_relationship import MODEL_INPUT_WIDTH, OnlineRelationshipState
from wnn.splits import file_sha256
from wnn.training.online_relationship import OnlineRelationshipResidualHead
from wnn.training.online_relationship_cache import SequenceCache
from wnn.training.online_relationship_fit import (
    BATCH_ROWS,
    GRADIENT_CLIP,
    LEARNING_RATE,
    WEIGHT_DECAY,
)

_BATCH_ROWS = 8192


def profile_cell(recipe: dict[str, Any], fold: int, lag: int, arm: str) -> dict[str, Any]:
    """Resolve exactly one frozen profile cell by its source axes."""
    matches = [
        cell
        for cell in recipe["cells"]
        if int(cell["fold"]) == fold and int(cell["lag"]) == lag and cell["arm"] == arm
    ]
    if len(matches) != 1:
        raise ValueError(f"recipe must define one fold={fold}, lag={lag}, arm={arm} profile cell")
    return matches[0]


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
    warmup_seconds: float,
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
        fit_open = phases["fit_cache_source_open"]
        eval_open = phases["evaluation_cache_source_open"]
        cache_seconds += fit_open + eval_open
        cache_seconds += (
            max(0.0, phases["fit_cache"] - fit_open)
            / data["cache_build"]["fit_sequences"]
            * data["fit_sequences"]
        )
        cache_seconds += (
            max(0.0, phases["evaluation_cache"] - eval_open)
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
        + warmup_seconds
    )
    return {
        "basis": "sample wall time scaled by full cache sequence counts, fit batches, and evaluation sequence counts",
        "profile_warmup_seconds_once": warmup_seconds,
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


def warm_profile_kernels(
    raw_strided: np.ndarray, normalizers: dict[tuple[int, int], Any]
) -> dict[str, object]:
    """Warm the actual compiled feature prefix and one disposable fixed-size update."""
    if raw_strided.ndim != 2 or raw_strided.shape[1] != 112 or not len(raw_strided):
        raise ValueError("profile warmup needs one actual raw sequence from the fit cache")
    started = time.perf_counter()
    prefix = raw_strided[: min(len(raw_strided), BATCH_ROWS)]
    warmed_lags = {}
    for (_, lag), normalizer in normalizers.items():
        if lag in warmed_lags:
            continue
        warmed_lags[lag] = normalizer
        OnlineRelationshipState("C", lag, normalizer).transform_compiled(prefix)
    kernel_seconds = time.perf_counter() - started

    step_started = time.perf_counter()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        head = OnlineRelationshipResidualHead().cpu()
        optimizer = torch.optim.AdamW(
            head.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        features = torch.zeros((BATCH_ROWS, MODEL_INPUT_WIDTH), dtype=torch.float32)
        parent = torch.zeros((BATCH_ROWS, 2), dtype=torch.float32)
        target = torch.zeros((BATCH_ROWS, 2), dtype=torch.float32)
        row_weight = torch.ones((BATCH_ROWS,), dtype=torch.float32)
        prediction = head(features, parent)
        loss = primary_pair_wmse(prediction[None], target[None], row_weight[None], target[None])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm_(head.parameters(), GRADIENT_CLIP, error_if_nonfinite=True)
        optimizer.step()
    step_seconds = time.perf_counter() - step_started
    return {
        "relationship_kernel_seconds": kernel_seconds,
        "optimizer_step_seconds": step_seconds,
        "seconds": time.perf_counter() - started,
        "relationship_lags": sorted(warmed_lags),
        "prefix_rows": len(prefix),
        "raw_strides_bytes": list(raw_strided.strides),
        "raw_was_strided": not raw_strided.flags.c_contiguous,
        "disposable_optimizer_rows": BATCH_ROWS,
        "optimizer_step_count": 1,
    }
