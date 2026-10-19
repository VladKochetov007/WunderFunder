"""Small, pure semantic checks for fit-only prediction profiles."""

from __future__ import annotations

import json
import os
import platform
import resource
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from wnn.run_identity import normalization_identity


def first_profile_groups(groups: list[int] | tuple[int, ...], batch_size: int = 16) -> list[int]:
    """Select the fixed leading fit batch without changing manifest order."""
    selected = list(map(int, groups))
    if batch_size <= 0 or len(selected) < batch_size:
        raise ValueError("profile membership must contain the full fixed batch")
    if any(group < 0 for group in selected) or len(set(selected)) != len(selected):
        raise ValueError("profile membership must be unique nonnegative row groups")
    return selected[:batch_size]


def profile_record(
    predictions: np.ndarray,
    groups: list[int],
    *,
    sequence_length: int,
    checkpoint_sha256: str,
    source_sha256: dict[str, str],
    split_manifest_sha256: str,
    normalization_identity: dict[str, Any],
    elapsed_seconds: float,
    rss_peak_bytes: int,
    cuda: dict[str, Any],
    profile_contract: dict[str, Any],
) -> dict[str, Any]:
    """Validate finite raw main predictions and return scalar profile metadata."""
    values = np.asarray(predictions)
    if (
        values.dtype != np.float32
        or values.ndim != 3
        or values.shape != (len(groups), sequence_length, 2)
        or not np.isfinite(values).all()
    ):
        raise ValueError("profile predictions must be finite FP32 main outputs with exact geometry")
    if not np.isfinite(elapsed_seconds) or elapsed_seconds <= 0 or rss_peak_bytes < 0:
        raise ValueError("profile timing and RSS observations must be finite and nonnegative")
    if not groups or len(set(groups)) != len(groups):
        raise ValueError("profile groups must be nonempty and unique")
    return {
        "record_type": "PROFILE",
        "status": "PASS",
        "sequences": len(groups),
        "rows": int(len(groups) * sequence_length),
        "sequence_length": sequence_length,
        "output_width": 2,
        "prediction_shape": list(values.shape),
        "prediction_dtype": str(values.dtype),
        "finite_output_count": int(values.size),
        "row_groups": groups,
        "split_manifest_sha256": split_manifest_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "source_sha256": source_sha256,
        "normalization_identity": normalization_identity,
        "prediction_finite": True,
        "prediction_min": float(values.min()),
        "prediction_max": float(values.max()),
        "prediction_mean": float(values.mean(dtype=np.float64)),
        "elapsed_seconds": float(elapsed_seconds),
        "rss_peak_bytes": int(rss_peak_bytes),
        "cuda": cuda,
        "profile_contract": profile_contract,
    }


def write_json_create_only(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON result without replacing an existing file or symlink."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def peak_rss_bytes() -> int:
    """Return this process's high-water resident set size in bytes."""
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def run_prediction_profile(
    model: torch.nn.Module,
    loader: Iterable[torch.Tensor],
    predict: Callable[[torch.nn.Module, torch.Tensor, int, torch.device], np.ndarray],
    device: torch.device,
    groups: list[int],
    output_path: Path,
    *,
    checkpoint_sha256: str,
    source_sha256: dict[str, str],
    split_manifest_sha256: str,
    model_input_width: int,
    sequence_length: int,
) -> dict[str, Any]:
    """Run exactly one fixed full-length inference batch and write scalar telemetry."""
    if (
        len(groups) != 16
        or sequence_length != 20_000
        or model_input_width != 211
        or torch.get_num_threads() != 1
        or not getattr(model, "_inference_fp32", False)
        or torch.backends.cuda.matmul.allow_tf32
        or torch.backends.cudnn.allow_tf32
    ):
        raise RuntimeError("profile execution differs from its fixed FP32 inference contract")

    normalization = normalization_identity(model)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    free_before, total_bytes = torch.cuda.mem_get_info(device)
    cuda = {
        "device": torch.cuda.get_device_name(device),
        "total_bytes": int(total_bytes),
        "free_before_bytes": int(free_before),
    }
    started = time.perf_counter()
    predictions: np.ndarray | None = None
    completed = 0
    for batch in loader:
        if predictions is not None:
            raise RuntimeError("profile-only yielded more than its one fixed inference batch")
        predictions = predict(model, batch, 4000, device)
        completed += len(predictions)
        print(
            json.dumps(
                {
                    "event": "prediction_progress",
                    "completed_sequences": completed,
                    "total_sequences": 16,
                }
            ),
            flush=True,
        )
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    if predictions is None or completed != 16:
        raise RuntimeError(f"profile wrote {completed} of 16 sequences")
    cuda.update(
        {
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        }
    )
    profile_contract = {
        "batch_size": 16,
        "chunk_size": 4000,
        "membership": "first_16_fit_train_manifest_order",
        "sequence_length": sequence_length,
        "raw_input_width": 112,
        "model_input_width": model_input_width,
        "hidden_width": 384,
        "layers": 2,
        "output_width": 2,
        "raw_main_outputs": True,
        "strict_fp32": True,
        "matmul_tf32": False,
        "cudnn_tf32": False,
        "autocast_enabled": False,
        "torch_threads": torch.get_num_threads(),
    }
    record = profile_record(
        predictions,
        groups,
        sequence_length=sequence_length,
        checkpoint_sha256=checkpoint_sha256,
        source_sha256=source_sha256,
        split_manifest_sha256=split_manifest_sha256,
        normalization_identity=normalization,
        elapsed_seconds=elapsed,
        rss_peak_bytes=peak_rss_bytes(),
        cuda=cuda,
        profile_contract=profile_contract,
    )
    write_json_create_only(output_path, record)
    print("PROFILE " + json.dumps(record, sort_keys=True, allow_nan=False), flush=True)
    return record
