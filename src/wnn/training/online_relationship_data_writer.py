"""Write one-sequence-at-a-time parent and fit-data cache files."""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from wnn.data import SequenceFile
from wnn.splits import file_sha256
from wnn.training.environment import TrainingEnvironment
from wnn.training.evaluation import predict
from wnn.training.inputs import build_inputs
from wnn.training.online_relationship_cache import (
    PARENT,
    RAW,
    ROWS,
    TARGET,
    WEIGHT,
    WIDTH,
    row_refs_sha256,
    sequence_cache_paths,
)
from wnn.training.online_relationship_parent_batch import (
    PARENT_BATCH_SEQUENCES,
    fit_sequence_targets_and_mask,
    read_parent_input_batch,
)
from wnn.training.online_relationship_sources import (
    cache_metadata,
)
from wnn.training.targets import main_row_weights

_LOG = logging.getLogger(__name__)


def write_scope(
    destination: Path,
    refs: list[dict[str, object]],
    scope: str,
    recipe: dict[str, Any],
    recipe_sha: str,
    fold: dict[str, Any],
    subset: dict[str, object],
    split_path: Path,
    dataset_identities: dict[str, object],
    propensity_identity: dict[str, object],
    parent_identity: dict[str, object],
    parent_mean: np.ndarray,
    parent_std: np.ndarray,
    parent_model: torch.nn.Module,
    parent_environment: TrainingEnvironment,
    propensity_model: torch.nn.Module,
    root: Path,
    device: str,
    row_weight_args: dict[str, object],
    sample_mode: bool,
    *,
    parent_prediction_path: bool = False,
) -> tuple[Path, Path | None, Path, float]:
    """Stream one cache scope into private temporary NPY files and a sidecar."""
    data_path, _ = sequence_cache_paths(destination, scope)
    data_temp = temporary_path(destination, data_path.name)
    meta_temp = temporary_path(destination, f"{data_path.stem}.json")
    prediction_temp = (
        temporary_path(destination, "evaluation_predictions.npy")
        if parent_prediction_path
        else None
    )
    array = np.lib.format.open_memmap(
        data_temp, mode="w+", dtype="<f4", shape=(len(refs), ROWS, WIDTH)
    )
    predictions = (
        np.lib.format.open_memmap(
            prediction_temp, mode="w+", dtype="<f4", shape=(len(refs), ROWS, 2)
        )
        if prediction_temp is not None
        else None
    )
    files: dict[str, SequenceFile] = {}
    source_open_seconds = 0.0
    weights_environment = TrainingEnvironment(root=root)
    args = argparse.Namespace(**row_weight_args)
    fitting = scope == "in_sample_fit"
    try:
        for start in range(0, len(refs), PARENT_BATCH_SEQUENCES):
            for ref in refs[start : start + PARENT_BATCH_SEQUENCES]:
                split = str(ref["split"])
                if split not in files:
                    _LOG.info("online_relationship_cache phase=open_parquet split=%s", split)
                    open_started = time.perf_counter()
                    files[split] = SequenceFile(root / "datasets" / f"{split}.parquet")
                    source_open_seconds += time.perf_counter() - open_started
            sequences, raw_batch = read_parent_input_batch(files, refs, start, fitting=fitting)
            parent_predictions = predict(
                parent_model,
                torch.from_numpy(raw_batch),
                int(parent_model._online_parent_chunk),
                torch.device(device),
                environment=parent_environment,
            )
            if (
                parent_predictions.shape != (len(sequences), ROWS, 2)
                or not np.isfinite(parent_predictions).all()
            ):
                raise ValueError("parent returned invalid predictions for one source batch")
            for offset, (sequence, parent_prediction) in enumerate(
                zip(sequences, parent_predictions, strict=True)
            ):
                index = start + offset
                raw = np.asarray(sequence.features, dtype=np.float32)
                record = np.zeros((ROWS, WIDTH), dtype="<f4")
                record[:, RAW] = raw
                record[:, PARENT] = parent_prediction
                if fitting:
                    target, score_mask = fit_sequence_targets_and_mask(sequence)
                    raw_tensor = torch.from_numpy(raw[None])
                    with torch.inference_mode():
                        inputs = build_inputs(raw_tensor, environment=weights_environment)
                        y = torch.from_numpy(target[None])
                        scored = torch.from_numpy(score_mask[None])
                        valid = torch.arange(ROWS)[None] >= weights_environment.warmup
                        row_weight = main_row_weights(
                            inputs,
                            y,
                            valid,
                            scored,
                            propensity_model,
                            None,
                            args,
                            raw_tensor,
                            environment=weights_environment,
                        )[0]
                    if (
                        target.shape != (ROWS, 2)
                        or not np.isfinite(target).all()
                        or not torch.isfinite(row_weight).all()
                    ):
                        raise ValueError("canonical labels or row weights are invalid")
                    record[:, TARGET] = target
                    record[:, WEIGHT] = row_weight.numpy().astype(np.float32, copy=False)
                array[index] = record
                if predictions is not None:
                    predictions[index] = parent_prediction
                del raw, record, parent_prediction
            del sequences, raw_batch, parent_predictions
        array.flush()
        if predictions is not None:
            predictions.flush()
        array._mmap.close()
        if predictions is not None:
            predictions._mmap.close()
        array = None
        predictions = None
        for source in files.values():
            _close_sequence_file(source)
        files.clear()
        metadata = cache_metadata(
            data_path.name,
            data_temp,
            prediction_temp,
            refs,
            scope,
            recipe,
            recipe_sha,
            fold,
            subset,
            split_path,
            dataset_identities,
            propensity_identity,
            parent_identity,
            parent_mean,
            parent_std,
            row_weight_args,
            sample_mode,
            root,
        )
        write_json_to_temp(meta_temp, metadata)
        return data_temp, prediction_temp, meta_temp, source_open_seconds
    except BaseException:
        if array is not None:
            array._mmap.close()
        if predictions is not None:
            predictions._mmap.close()
        for source in files.values():
            _close_sequence_file(source)
        for path in (data_temp, meta_temp, prediction_temp):
            if path is not None:
                path.unlink(missing_ok=True)
        raise


def parent_prediction_metadata(
    prediction_temp: Path,
    prediction_path: Path,
    recipe: dict[str, Any],
    recipe_path: Path,
    recipe_sha: str,
    fold: dict[str, Any],
    refs: list[dict[str, object]],
    cache_sidecar: Path,
    cache_sidecar_sha256: str,
    sample_mode: bool,
) -> dict[str, object]:
    """Bind the standalone parent endpoint to evaluation-cache source order."""
    return {
        "schema": "wnn.online-relationships.parent-predictions.v1",
        "scope": "feature_only_eval",
        "recipe_id": recipe["recipe_id"],
        "recipe": str(recipe_path),
        "recipe_sha256": recipe_sha,
        "fold": int(fold["fold"]),
        "seed": int(fold["seed"]),
        "parent": fold["parent"],
        "split_manifest_sha256": fold["split_manifest_sha256"],
        "evaluation_cache_sidecar": str(cache_sidecar),
        "evaluation_cache_sidecar_sha256": cache_sidecar_sha256,
        "row_refs": refs,
        "membership_sha256": row_refs_sha256(refs),
        "prediction_file": prediction_path.name,
        "prediction_path": str(prediction_path),
        "prediction_sha256": file_sha256(prediction_temp),
        "shape": [len(refs), ROWS, 2],
        "dtype": "<f4",
        "endpoint": {
            "precision": "strict_fp32",
            "backend": fold.get("_parent_endpoint_backend", "recorded-in-cache-source"),
        },
        "sampled_profile": sample_mode,
        "prediction_ancestry": "frozen parent checkpoint evaluated on exact source-qualified outer-fold groups",
    }


def representative_refs(refs: list[dict[str, object]], count: int) -> list[dict[str, object]]:
    """Select evenly spaced whole sequences without changing source order."""
    if count <= 0 or count > len(refs):
        raise ValueError("profile sample size must be positive and fit frozen membership")
    indexes = np.linspace(0, len(refs) - 1, count, dtype=np.int64)
    return [refs[int(index)] for index in indexes]


def publish_create_only(items: tuple[tuple[Path, Path], ...]) -> tuple[Path, ...]:
    """Atomically expose completed files while refusing to overwrite any endpoint."""
    destinations = [destination for _, destination in items]
    if len(set(destinations)) != len(destinations) or any(path.exists() for path in destinations):
        raise FileExistsError("online cache destination exists; refusing overwrite")
    created = []
    try:
        for source, destination in items:
            os.link(source, destination)
            created.append(destination)
        return tuple(created)
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    finally:
        for source, _ in items:
            source.unlink(missing_ok=True)


def temporary_path(directory: Path, stem: str) -> Path:
    """Reserve a unique same-filesystem name for an atomic cache publish."""
    fd, name = tempfile.mkstemp(prefix=f".{stem}.", suffix=".partial", dir=directory)
    os.close(fd)
    Path(name).unlink()
    return Path(name)


def write_json_to_temp(path: Path, value: dict[str, object]) -> None:
    """Write deterministic JSON into a temporary file before publication."""
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def _close_sequence_file(source: SequenceFile) -> None:
    """Release Parquet metadata before opening the next source split."""
    close = getattr(source._pf, "close", None)
    if close is not None:
        close()
