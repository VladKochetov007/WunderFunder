"""Create-only R12 output, private model archive, and fit-only diagnostics."""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

import numpy as np
import torch
from idea_screens_r7.model_math import predict_rows
from idea_screens_r7.runtime_support import append_json_line, physical_output, write_json


def create_output(
    output: Path,
    scratch: Path,
    sample_path: Path,
    identities: dict[str, str],
    preparation_sha256: str,
) -> tuple[Path, float, float]:
    """Create a fresh physical output directory and initial durable records."""
    physical_output(output, scratch)
    output.mkdir()
    started_wall, started_cpu = time.perf_counter(), time.process_time()
    write_json(
        output / "START.json",
        {
            "schema": "nf19.r12_objective_pair.start.v1",
            "created_epoch_s": time.time(),
            **identities,
            "preparation_manifest_sha256": preparation_sha256,
            "sample_path": str(sample_path),
            "output_path": str(output),
        },
    )
    progress = output / "PROGRESS.jsonl"
    with progress.open("xb"):
        pass
    append_json_line(
        progress, {"event": "initialized", "elapsed_wall_s": 0.0, "elapsed_cpu_s": 0.0}
    )
    return progress, started_wall, started_cpu


def write_model_archive(
    path: Path,
    wmse: torch.nn.Module,
    pooled_corr: torch.nn.Module,
    mean: np.ndarray,
    scale: np.ndarray,
) -> dict[str, object]:
    """Create a private NPZ of model parameters and shared fit normalizer only."""
    arrays: dict[str, np.ndarray] = {"normalizer_mean": mean, "normalizer_scale": scale}
    for arm, model in (("wmse", wmse), ("pooled_corr", pooled_corr)):
        for name, tensor in model.state_dict().items():
            arrays[f"{arm}__{name.replace('.', '__')}"] = tensor.detach().cpu().numpy()
    with path.open("xb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path_name": path.name,
        "sha256": digest.hexdigest(),
        "bytes": path.stat().st_size,
        "array_names": sorted(arrays),
        "predictions_in_archive": False,
    }


def fit_prediction_diagnostics(
    model: torch.nn.Module, fit_features: np.ndarray
) -> list[dict[str, float | int]]:
    """Summarize fit-only raw prediction clipping frequency by target."""
    prediction, _ = predict_rows(model, fit_features)
    if not np.isfinite(prediction).all():
        raise FloatingPointError("nonfinite R12 fit prediction diagnostic")
    return [
        {
            "rows": int(len(prediction)),
            "outside_clip_rows": int(np.count_nonzero(np.abs(prediction[:, index]) > 2.0)),
            "outside_clip_fraction": float(np.mean(np.abs(prediction[:, index]) > 2.0)),
        }
        for index in range(2)
    ]
