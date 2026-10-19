"""Strict-FP32 normalization and update core, imported after durable START."""

from __future__ import annotations

import argparse
import datetime as dt
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from propensity_profile import (
    PROFILE_BATCH_SIZE,
    PROFILE_CHUNK_ROWS,
    PROFILE_GPU_MEMORY_BYTES,
    PROFILE_GPU_UUID,
    PROFILE_REPEATS,
    PROFILE_SEED,
    PROFILE_SEQUENCE_LENGTH,
    ProfilePlan,
    ProfileRows,
)

if TYPE_CHECKING:
    import numpy as np


class _Journal(Protocol):
    def append(self, attempt_id: str, event: dict[str, object], *, durable: bool = False) -> int:
        """Append a durable profile progress event."""


@dataclass(frozen=True)
class ProfileRowsData:
    """Small data container for the exact admitted fit-valid profile rows."""

    plan: ProfilePlan
    sequence_ids: tuple[int, ...]
    feature_batches: tuple[np.ndarray, ...]
    masks: tuple[np.ndarray, ...]
    split_sha256: str
    valid_path: Path
    valid_stat: dict[str, int]
    valid_registered_identity: dict[str, object]
    load_seconds: float


@dataclass(frozen=True)
class ProfileOutcome:
    """Completed profile evidence passed to the create-only receipt writer."""

    attempt_id: str
    started_utc: str
    runtime: dict[str, object]
    rows: ProfileRows
    source_hashes: dict[str, str]
    normalization: dict[str, object]
    repeat_records: list[dict[str, object]]
    optimizer_progress: dict[str, object]
    journal: _Journal


def _rss_bytes() -> int:
    """Return process peak resident memory in bytes on the worker."""
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value * 1024 if sys.platform.startswith("linux") else value


def _phase(journal: _Journal, attempt_id: str, phase: str) -> None:
    import json

    journal.append(attempt_id, {"event": "profile_phase", "phase": phase}, durable=True)
    print(json.dumps({"event": "PROFILE_PHASE", "phase": phase}), flush=True)


def _normalization(
    rows: ProfileRows, model: object, journal: _Journal, attempt_id: str
) -> dict[str, object]:
    """Fit profile-only Welford statistics on CPU from the selected fit groups."""
    import torch

    from wnn.data import N_FEATURES
    from wnn.features import N_EVENT_FEATURES
    from wnn.propensity import fit_normalization
    from wnn.training.inputs import build_inputs

    _phase(journal, attempt_id, "building_cpu_inputs_and_profile_normalization")
    batches: list[np.ndarray] = []
    for features in rows.feature_batches:
        inputs = build_inputs(torch.from_numpy(features).unsqueeze(0))
        expected = (1, PROFILE_SEQUENCE_LENGTH, N_FEATURES + N_EVENT_FEATURES)
        if inputs.dtype != torch.float32 or tuple(inputs.shape) != expected:
            raise ValueError("canonical input builder changed profile shape or dtype")
        if not torch.isfinite(inputs).all().item():
            raise FloatingPointError("fit-valid profile inputs are non-finite")
        batches.append(inputs.squeeze(0).contiguous().numpy())
    result = fit_normalization(model, (torch.from_numpy(batch).unsqueeze(0) for batch in batches))
    return {**result, "groups": list(rows.plan.batch_groups), "scope": "profile-only; discard"}


def _updates(
    rows: ProfileRows, model: object, journal: _Journal, attempt_id: str, device: object
) -> tuple[list[dict[str, object]], dict[str, object], float, float]:
    """Run two true GPU B16 batches through the canonical strict update helper."""
    import numpy as np
    import torch
    from wnn.propensity_fit_update import _fit_update
    from wnn.propensity_progress import UpdateProgress

    from wnn.data import WARMUP
    from wnn.propensity import ordered_groups_sha256
    from wnn.training.inputs import build_inputs

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-2)
    total = UpdateProgress()
    records: list[dict[str, object]] = []
    _phase(journal, attempt_id, "strict_fp32_gpu_batches")
    gpu_input_seconds = 0.0
    optimizer_seconds = 0.0
    model.train()
    for repeat_index, groups in enumerate(rows.plan.repeat_batch_groups):
        if groups != rows.plan.batch_groups:
            raise RuntimeError("profile repeat changed real row-group membership")
        hidden = None
        torch.cuda.synchronize(device)
        input_started = time.monotonic()
        raw = torch.from_numpy(np.stack(rows.feature_batches)).to(device)
        inputs = build_inputs(raw)
        target = torch.from_numpy(np.stack(rows.masks)).to(device=device, dtype=torch.float32)
        expected_shape = (PROFILE_BATCH_SIZE, PROFILE_SEQUENCE_LENGTH, 211)
        if inputs.dtype != torch.float32 or tuple(inputs.shape) != expected_shape:
            raise ValueError("canonical GPU input builder changed full B16 shape or dtype")
        if not torch.isfinite(inputs).all().item() or not torch.isfinite(target).all().item():
            raise FloatingPointError("full profile batch is non-finite")
        torch.cuda.synchronize(device)
        input_elapsed = time.monotonic() - input_started
        gpu_input_seconds += input_elapsed
        del raw
        progress, hidden = UpdateProgress(), None
        optimizer_started = time.monotonic()
        for chunk_start in rows.plan.chunk_starts:
            stop = chunk_start + PROFILE_CHUNK_ROWS
            hidden = _fit_update(
                model=model,
                optimizer=optimizer,
                inputs=inputs[:, chunk_start:stop],
                target=target[:, chunk_start:stop],
                hidden=hidden,
                chunk_start=chunk_start,
                warmup=WARMUP,
                epoch=repeat_index,
                batch=0,
                attempt_id=attempt_id,
                journal=journal,
                cumulative=total,
                epoch_progress=progress,
            )
        torch.cuda.synchronize(device)
        optimizer_elapsed = time.monotonic() - optimizer_started
        optimizer_seconds += optimizer_elapsed
        del inputs, target
        steps = len(rows.plan.chunk_starts)
        if (
            progress.optimizer_attempts,
            progress.optimizer_step_calls,
            progress.optimizer_completed_updates,
            progress.nonfinite_updates,
        ) != (steps, steps, steps, 0):
            raise RuntimeError("profile repeat did not complete exactly ten finite updates")
        records.append(
            {
                "repeat_index": repeat_index,
                "row_groups": list(groups),
                "row_groups_sha256": ordered_groups_sha256(groups),
                "sequence_ids": list(rows.sequence_ids),
                "full_gpu_input_shape": list(expected_shape),
                "gpu_input_build_seconds": input_elapsed,
                "optimizer_update_seconds": optimizer_elapsed,
                "optimizer_progress": progress.snapshot(),
            }
        )
        _phase(journal, attempt_id, f"repeat_{repeat_index}_complete")
    summary = total.snapshot()
    expected_total = PROFILE_REPEATS * len(rows.plan.chunk_starts)
    if (
        summary["optimizer_attempts"],
        summary["optimizer_step_calls"],
        summary["optimizer_completed_updates"],
        summary["nonfinite_updates"],
    ) != (expected_total, expected_total, expected_total, 0):
        raise RuntimeError("profile did not complete exactly twenty finite updates")
    return records, summary, gpu_input_seconds, optimizer_seconds


def run_profile(
    args: argparse.Namespace,
    rows: ProfileRows,
    journal: _Journal,
    attempt_id: str,
    started_utc: str,
    started: float,
) -> ProfileOutcome:
    """Run the bounded strict-FP32 profile on already admitted fit-valid rows."""
    import numpy as np
    import pyarrow
    import torch

    from wnn.data import N_FEATURES, SEQUENCE_LENGTH
    from wnn.features import N_EVENT_FEATURES
    from wnn.propensity import PROPENSITY_FIT_SOURCE_FILES, STRICT_FP32_PROFILE, PropensityNet
    from wnn.resources import configure_strict_fp32, configure_torch
    from wnn.splits import file_sha256

    source_root = args.source_root.resolve(strict=True)
    if SEQUENCE_LENGTH != PROFILE_SEQUENCE_LENGTH or not torch.cuda.is_available():
        raise RuntimeError("canonical sequence length or admitted CUDA device is unavailable")
    torch.set_num_interop_threads(1)
    torch.manual_seed(PROFILE_SEED)
    torch.cuda.manual_seed_all(PROFILE_SEED)
    configure_torch()
    configure_strict_fp32()
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")
    properties = torch.cuda.get_device_properties(device)
    if properties.total_memory < PROFILE_GPU_MEMORY_BYTES:
        raise RuntimeError("selected CUDA device is smaller than the allocator cap")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    model = PropensityNet(N_FEATURES + N_EVENT_FEATURES, hidden=64).to(device)
    model._propensity_profile = STRICT_FP32_PROFILE
    norm_start = time.monotonic()
    normalization = _normalization(rows, model, journal, attempt_id)
    normalization_seconds = time.monotonic() - norm_start
    repeats, updates, gpu_input_seconds, update_seconds = _updates(
        rows, model, journal, attempt_id, device
    )
    torch.cuda.synchronize()
    after = rows.valid_path.stat()
    identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    expected = tuple(rows.valid_stat[key] for key in ("device", "inode", "bytes", "mtime_ns"))
    if identity != expected:
        raise RuntimeError("valid parquet changed during profile training")
    _phase(journal, attempt_id, "writing_create_only_result")
    hash_started = time.monotonic()
    source_hashes = {name: file_sha256(source_root / name) for name in PROPENSITY_FIT_SOURCE_FILES}
    source_hash_seconds = time.monotonic() - hash_started
    runtime: dict[str, object] = {
        "started_utc": started_utc,
        "ended_utc": dt.datetime.now(dt.UTC).isoformat(),
        "elapsed_seconds": time.monotonic() - started,
        "data_validation_and_read_seconds": rows.load_seconds,
        "input_and_normalization_seconds": normalization_seconds,
        "full_gpu_batch_input_build_seconds": gpu_input_seconds,
        "update_seconds": update_seconds,
        "source_hash_seconds": source_hash_seconds,
        "python": sys.version,
        "numpy": np.__version__,
        "pyarrow": pyarrow.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device_name": properties.name,
        "device_total_memory_bytes": properties.total_memory,
        "cuda_visible_devices": PROFILE_GPU_UUID,
        "allocator_cap_bytes": PROFILE_GPU_MEMORY_BYTES,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "process_peak_rss_bytes": _rss_bytes(),
        "cpu_threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "matmul_precision": torch.get_float32_matmul_precision(),
        "tf32": False,
        "autocast": False,
    }
    if runtime["cpu_threads"] != 1 or runtime["interop_threads"] != 1:
        raise RuntimeError("profile runtime did not honor the one-thread CPU admission")
    return ProfileOutcome(
        attempt_id,
        started_utc,
        runtime,
        rows,
        source_hashes,
        normalization,
        repeats,
        updates,
        journal,
    )
