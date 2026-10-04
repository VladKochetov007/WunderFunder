"""Endpoint-event teacher EMA clock for the isolated E426 trainer patch."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import torch

ENDPOINT = np.float32(-5.199337482452393)
ClockMode = Literal["constant", "aligned", "shuffled"]


@dataclass(frozen=True)
class EventEMASchedule:
    """Per-optimizer-update event counts and teacher decays for one epoch."""

    observed_counts: tuple[int, ...]
    applied_counts: tuple[int, ...]
    decays: tuple[float, ...]
    updates_per_epoch: int
    total_events: int
    mode: ClockMode
    epoch: int
    count_sha256: str


def endpoint_events_by_update(
    parquet_path: Path,
    row_groups: Sequence[int],
    update_rows: int,
    *,
    sequence_length: int = 20_000,
) -> np.ndarray:
    """Count raw a6 endpoint markers per sequence and optimizer update window.

    Only the input-only `a6` column is read. Each row group is one complete
    sequence, so no padding is introduced and no target or scoring column is
    consulted.
    """
    if update_rows <= 0 or sequence_length <= 0 or sequence_length % update_rows:
        raise ValueError("update_rows must positively divide sequence_length")
    groups = np.asarray(row_groups, dtype=np.int64)
    if groups.ndim != 1 or np.any(groups < 0):
        raise ValueError("row_groups must be a one-dimensional nonnegative list")
    if len(np.unique(groups)) != len(groups):
        raise ValueError("E426 event clock does not accept duplicated fit sequences")

    import pyarrow.parquet as parquet

    source = parquet.ParquetFile(parquet_path)
    if np.any(groups >= source.num_row_groups):
        raise ValueError("fit row group is outside the parquet file")
    windows = sequence_length // update_rows
    counts = np.zeros((len(groups), windows), dtype=np.int64)
    for output_index, row_group in enumerate(groups.tolist()):
        table = source.read_row_group(int(row_group), columns=["a6"], use_threads=False)
        values = table.column("a6").combine_chunks().to_numpy(zero_copy_only=False)
        if len(values) != sequence_length:
            raise ValueError(f"row group {row_group} has unexpected sequence length")
        markers = np.asarray(values, dtype=np.float32) == ENDPOINT
        counts[output_index] = markers.reshape(windows, update_rows).sum(axis=1)
    return counts


def per_update_event_counts(
    sequence_counts: np.ndarray,
    epoch_order: Sequence[int],
    batch_size: int,
    *,
    drop_last: bool = True,
) -> tuple[int, ...]:
    """Aggregate event counts in the exact batch/update order used by training."""
    counts = np.asarray(sequence_counts, dtype=np.int64)
    order = np.asarray(epoch_order, dtype=np.int64)
    if counts.ndim != 2 or batch_size < 1 or order.ndim != 1:
        raise ValueError("expected [sequence, update-window] counts and one-dimensional order")
    if np.any(counts < 0) or np.any(order < 0) or np.any(order >= len(counts)):
        raise ValueError("event counts/order contain invalid values")
    retained = len(order) - len(order) % batch_size if drop_last else len(order)
    if not drop_last and retained % batch_size:
        raise ValueError("non-drop-last event batches must have an explicit padded-row policy")
    if retained == 0:
        raise ValueError("event-clock epoch has no complete optimizer batches")
    ordered = counts[order[:retained]]
    batch_windows = ordered.reshape(-1, batch_size, counts.shape[1]).sum(axis=1)
    return tuple(int(value) for value in batch_windows.reshape(-1))


def make_event_ema_schedule(
    observed_counts: Sequence[int],
    *,
    base_decay: float,
    updates_per_epoch: int,
    mode: ClockMode,
    seed: int,
    epoch: int,
) -> EventEMASchedule:
    """Build constant, aligned, or deterministically shuffled update decays.

    Aligned decay follows the preregistered formula
    ``base_decay ** (events_t * updates_per_epoch / total_events_epoch)``.
    This normalizes the product of all update retentions to the constant arm.
    """
    observed = tuple(int(value) for value in observed_counts)
    if not observed or any(value < 0 for value in observed):
        raise ValueError("a nonempty nonnegative event count is required per optimizer update")
    if not math.isfinite(base_decay) or not 0.0 < base_decay <= 1.0:
        raise ValueError("base_decay must be finite and in (0, 1]")
    if updates_per_epoch != len(observed) or updates_per_epoch < 1:
        raise ValueError("updates_per_epoch must equal the event schedule length")
    if mode not in ("constant", "aligned", "shuffled"):
        raise ValueError(f"unknown event EMA clock: {mode}")
    total_events = sum(observed)
    if total_events <= 0:
        raise ValueError("fit epoch contains no endpoint events")

    if mode == "shuffled":
        permutation_seed = (seed + 1_000_003 * epoch) % (2**32)
        permutation = np.random.default_rng(permutation_seed).permutation(len(observed))
        applied = tuple(observed[int(index)] for index in permutation)
    else:
        applied = observed

    if mode == "constant":
        decays = (base_decay,) * updates_per_epoch
    elif mode == "aligned":
        decays = tuple(
            base_decay ** (events * updates_per_epoch / total_events) for events in applied
        )
    else:
        decays = tuple(
            base_decay ** (events * updates_per_epoch / total_events) for events in applied
        )

    digest = hashlib.sha256(np.asarray(applied, dtype=np.int64).tobytes()).hexdigest()
    return EventEMASchedule(
        observed_counts=observed,
        applied_counts=applied,
        decays=decays,
        updates_per_epoch=updates_per_epoch,
        total_events=total_events,
        mode=mode,
        epoch=epoch,
        count_sha256=digest,
    )


@torch.no_grad()
def update_ema_with_decay(averaged: Any, model: torch.nn.Module, decay: float) -> None:
    """Update FastEMA or AveragedModel with one temporary per-update decay."""
    if not math.isfinite(decay) or not 0.0 <= decay <= 1.0:
        raise ValueError("effective EMA decay must be finite and in [0, 1]")
    if hasattr(averaged, "decay"):
        original = averaged.decay
        try:
            averaged.decay = decay
            averaged.update_parameters(model)
        finally:
            averaged.decay = original
    elif hasattr(averaged, "multi_avg_fn"):
        original = averaged.multi_avg_fn
        try:
            averaged.multi_avg_fn = torch.optim.swa_utils.get_ema_multi_avg_fn(decay)
            averaged.update_parameters(model)
        finally:
            averaged.multi_avg_fn = original
    else:
        raise TypeError("expected FastEMA or torch AveragedModel")


def inference_ema_module(evaluation_ema: Any) -> torch.nn.Module:
    """Return the evaluation EMA module; never select the consistency teacher."""
    module = getattr(evaluation_ema, "module", None)
    if not isinstance(module, torch.nn.Module):
        raise ValueError("evaluation EMA has no model module")
    return module
