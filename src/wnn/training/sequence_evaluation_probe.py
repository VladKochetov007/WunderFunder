"""Bounded validation-throughput probe for fresh canonical sequence models."""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from wnn.data import SEQUENCE_LENGTH
from wnn.training.sequence_config import load_config
from wnn.training.sequence_data import SequenceRecords
from wnn.training.sequence_distributed import DistributedContext, configure_strict_fp32
from wnn.training.sequence_evaluation_probe_source import (
    MAX_PROBE_SEQUENCES,
    _read_valid_source,
    _select_valid_members,
)
from wnn.training.sequence_features import sequence_input_width
from wnn.training.sequence_models import build_sequence_model
from wnn.training.sequence_monitor_evaluation import (
    _assert_batch_norm_unchanged,
    _batch_norm_buffers,
    _restore_rng,
    _rng_state,
    evaluate_dataset,
)


def _evaluate_preserving_state(
    model: nn.Module,
    dataset: SequenceRecords,
    chunk_rows: int,
    batch_size: int,
    context: DistributedContext,
) -> dict[str, int | float]:
    """Run canonical valid evaluation and restore modes, BatchNorm buffers, and RNG."""
    modes = [(module, module.training) for module in model.modules()]
    buffers = _batch_norm_buffers(model)
    rng = _rng_state(context)
    try:
        model.eval()
        return evaluate_dataset(
            model,
            dataset,
            "valid",
            chunk_rows,
            "fp32",
            context,
            batch_size,
        )
    finally:
        for module, training in modes:
            module.training = training
        _restore_rng(rng, context)
        if any(module.training != training for module, training in modes):
            raise RuntimeError("validation evaluation did not restore model training modes")
        _assert_rng_restored(rng, context)
        _assert_batch_norm_unchanged(buffers)


def _assert_rng_restored(expected: dict[str, object], context: DistributedContext) -> None:
    """Fail if report-only evaluation changes any captured process RNG stream."""
    actual = _rng_state(context)
    if expected["python"] != actual["python"]:
        raise RuntimeError("validation evaluation did not restore Python RNG state")
    expected_numpy, actual_numpy = expected["numpy"], actual["numpy"]
    if not isinstance(expected_numpy, tuple) or not isinstance(actual_numpy, tuple):
        raise RuntimeError("validation evaluation has an invalid NumPy RNG snapshot")
    if (
        expected_numpy[0] != actual_numpy[0]
        or not np.array_equal(expected_numpy[1], actual_numpy[1])
        or expected_numpy[2:] != actual_numpy[2:]
    ):
        raise RuntimeError("validation evaluation did not restore NumPy RNG state")
    expected_torch, actual_torch = expected["torch"], actual["torch"]
    if not isinstance(expected_torch, torch.Tensor) or not isinstance(actual_torch, torch.Tensor):
        raise RuntimeError("validation evaluation has an invalid Torch RNG snapshot")
    if not torch.equal(expected_torch, actual_torch):
        raise RuntimeError("validation evaluation did not restore Torch RNG state")
    if "cuda" in expected:
        expected_cuda, actual_cuda = expected["cuda"], actual.get("cuda")
        if not isinstance(expected_cuda, torch.Tensor) or not isinstance(actual_cuda, torch.Tensor):
            raise RuntimeError("validation evaluation has an invalid CUDA RNG snapshot")
        if not torch.equal(expected_cuda, actual_cuda):
            raise RuntimeError("validation evaluation did not restore CUDA RNG state")


def _strict_fp32_receipt() -> dict[str, object]:
    """Return the active precision controls after explicit strict-FP32 setup."""
    result = {
        "mode": "fp32",
        "matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
    }
    if (
        result["matmul_precision"] != "highest"
        or result["cuda_matmul_allow_tf32"]
        or result["cudnn_allow_tf32"]
    ):
        raise RuntimeError("strict FP32 controls were not active")
    return result


def run_evaluation_probe(
    config_path: Path,
    data_root: Path,
    source_audit: Path,
    *,
    device_name: str,
    batch_size: int,
    chunk_rows: int,
    max_sequences: int = MAX_PROBE_SEQUENCES,
) -> dict[str, object]:
    """Time one bounded canonical evaluation with fresh random, untrained weights."""
    if batch_size not in (128, 384) or chunk_rows not in (64, 256, 1000):
        raise ValueError("probe batch size must be128/384 and chunk rows64/256/1000")
    if device_name != "cpu" and not device_name.startswith("cuda"):
        raise ValueError("device must be cpu or a CUDA device such as cuda:0")
    config_path = config_path.resolve()
    config_bytes = config_path.read_bytes()
    config = load_config(config_path)
    if config["precision_mode"] != "fp32":
        raise ValueError("validation probe accepts only strict FP32 launch configs")
    source, valid_path = _read_valid_source(data_root, source_audit)
    members = _select_valid_members(source["identity"]["row_groups"], max_sequences)
    membership_bytes = json.dumps(members, separators=(",", ":")).encode()

    device = torch.device(device_name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA evaluation requested but CUDA is unavailable")
        torch.cuda.set_device(device)
    configure_strict_fp32()
    precision = _strict_fp32_receipt()
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - fixed engineering-probe initialization
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model_config = config["model"]
    input_mode = str(model_config.get("input_feature_mode", "raw112"))
    width = sequence_input_width(input_mode)
    model = build_sequence_model(model_config).to(device)
    mean = getattr(model, "mean", None)
    std = getattr(model, "std", None)
    if not isinstance(mean, torch.Tensor) or not isinstance(std, torch.Tensor):
        raise TypeError("canonical sequence model must expose external input moments")
    if mean.numel() != width or std.numel() != width:
        raise ValueError("model normalization buffers do not match configured input width")
    with torch.no_grad():
        mean.zero_()
        std.fill_(1.0)
    records = SequenceRecords(
        data_root.resolve(),
        members,
        int(model_config["hint_dim"]),
        training_mask="is_scored",
        input_feature_mode=input_mode,
    )
    records.validate_source_schema()
    context = DistributedContext(0, 0, 1, device, False)
    baseline_allocated = torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        start_event, end_event = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start_event.record()
    started = time.perf_counter()
    metrics = _evaluate_preserving_state(model, records, chunk_rows, batch_size, context)
    if device.type == "cuda":
        end_event.record()
        torch.cuda.synchronize(device)
        device_seconds: float | None = start_event.elapsed_time(end_event) / 1000.0
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
    else:
        device_seconds = None
        peak_allocated = peak_reserved = 0
    elapsed = time.perf_counter() - started
    if not elapsed > 0 or any(not math.isfinite(float(value)) for value in metrics.values()):
        raise RuntimeError("canonical evaluation returned nonfinite metrics or timing")
    selected_rows = int(metrics["valid/selected_rows"])
    forwarded_rows = len(records) * SEQUENCE_LENGTH
    return {
        "schema": "tabdata.untrained-validation-evaluation-probe.v1",
        "purpose": "actual_valid_subset_engineering_only",
        "quality_claim": "none_untrained_weights",
        "no_fit_or_checkpoint": True,
        "config": {
            "path": str(config_path),
            "sha256": hashlib.sha256(config_bytes).hexdigest(),
            "architecture": model_config["architecture"],
            "input_feature_mode": input_mode,
            "input_width": width,
            "seed": seed,
        },
        "source": source,
        "membership": {
            "official_split": "complete later valid source",
            "selected_sequences": len(members),
            "selection": "first physical row groups; deterministic subset, not chronological order",
            "row_groups_sha256": hashlib.sha256(membership_bytes).hexdigest(),
        },
        "normalization": {
            "mode": "identity defaults; no moments fitted or loaded",
            "mean": 0.0,
            "std": 1.0,
        },
        "weights": "fresh canonical random initialization",
        "precision": precision,
        "evaluation": {
            "batch_size": batch_size,
            "chunk_rows": chunk_rows,
            "elapsed_seconds": elapsed,
            "device_event_seconds": device_seconds,
            "forwarded_rows": forwarded_rows,
            "selected_rows_per_second": selected_rows / elapsed,
            "forwarded_rows_per_second": forwarded_rows / elapsed,
            "peak_allocated_bytes": peak_allocated or None,
            "peak_reserved_bytes": peak_reserved or None,
            "peak_allocated_delta_bytes": max(0, peak_allocated - baseline_allocated)
            if device.type == "cuda"
            else None,
            "metrics": metrics,
        },
        "normalization_history": "untrained/no moments",
        "row_source": valid_path,
    }
