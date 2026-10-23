"""Single-pass, complete-sequence batching shared by the three matched arms."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from .age_readout_features import normalized_age, normalized_base

ROWS_PER_SEQUENCE = 20_000
RAW = slice(0, 112)
TARGET = slice(112, 114)
WEIGHT = 114
PARENT = slice(115, 117)
WIDTH = 117
BATCH_ROWS = 8192


@dataclass(frozen=True)
class FitBatch:
    """A shared canonical base, two age blocks and fitting-only labels."""

    base: Tensor
    group_age: Tensor
    field_age: Tensor
    target: Tensor
    row_weight: Tensor
    parent: Tensor


@dataclass(frozen=True)
class EvalSequence:
    """Target-free raw features and cached parent output for one held-out group."""

    raw: Tensor
    parent: Tensor


def iter_fit_batches(
    cache: np.ndarray,
    sequence_count: int,
    parent_mean: Tensor,
    parent_std: Tensor,
    device: torch.device,
    *,
    batch_rows: int = BATCH_ROWS,
) -> Iterator[FitBatch]:
    """Read each selected sequence once, then yield common arm batches."""
    if sequence_count < 1 or sequence_count > len(cache) or batch_rows < 1:
        raise ValueError("fit sequence or batch count is invalid")
    buffers = _buffers(batch_rows, device)
    filled = 0
    for sequence_index in range(sequence_count):
        row = cache[sequence_index]
        _validate_row_header(row)
        raw = torch.tensor(row[:, RAW], dtype=torch.float32, device=device)
        base = normalized_base(raw, parent_mean, parent_std)
        group_age = normalized_age(raw, "group_age")
        field_age = normalized_age(raw, "field_age")
        target = torch.tensor(row[:, TARGET], dtype=torch.float32, device=device)
        weight = torch.tensor(row[:, WEIGHT], dtype=torch.float32, device=device)
        parent = torch.tensor(row[:, PARENT], dtype=torch.float32, device=device)
        if not bool(torch.isfinite(target).all() and torch.isfinite(weight).all()):
            raise ValueError("fit cache contains nonfinite labels or row weights")
        if bool(torch.any(weight < 0)):
            raise ValueError("fit cache row weights must be nonnegative")
        position = 0
        while position < ROWS_PER_SEQUENCE:
            take = min(batch_rows - filled, ROWS_PER_SEQUENCE - position)
            end = filled + take
            source_end = position + take
            buffers["base"][filled:end].copy_(base[position:source_end])
            buffers["group_age"][filled:end].copy_(group_age[position:source_end])
            buffers["field_age"][filled:end].copy_(field_age[position:source_end])
            buffers["target"][filled:end].copy_(target[position:source_end])
            buffers["row_weight"][filled:end].copy_(weight[position:source_end])
            buffers["parent"][filled:end].copy_(parent[position:source_end])
            filled = end
            position = source_end
            if filled == batch_rows:
                yield FitBatch(**buffers)
                filled = 0
    if filled:
        yield FitBatch(**{name: values[:filled] for name, values in buffers.items()})


def iter_eval_sequences(
    cache: np.ndarray, sequence_count: int, device: torch.device
) -> Iterator[EvalSequence]:
    """Read only raw and parent columns; never index evaluation target or mask columns."""
    if sequence_count < 1 or sequence_count > len(cache):
        raise ValueError("evaluation sequence count is invalid")
    for sequence_index in range(sequence_count):
        row = cache[sequence_index]
        _validate_row_header(row)
        yield EvalSequence(
            raw=torch.tensor(row[:, RAW], dtype=torch.float32, device=device),
            parent=torch.tensor(row[:, PARENT], dtype=torch.float32, device=device),
        )


def _buffers(batch_rows: int, device: torch.device) -> dict[str, Tensor]:
    return {
        "base": torch.empty((batch_rows, 211), dtype=torch.float32, device=device),
        "group_age": torch.empty((batch_rows, 112), dtype=torch.float32, device=device),
        "field_age": torch.empty((batch_rows, 112), dtype=torch.float32, device=device),
        "target": torch.empty((batch_rows, 2), dtype=torch.float32, device=device),
        "row_weight": torch.empty(batch_rows, dtype=torch.float32, device=device),
        "parent": torch.empty((batch_rows, 2), dtype=torch.float32, device=device),
    }


def _validate_row_header(row: np.ndarray) -> None:
    if row.shape != (ROWS_PER_SEQUENCE, WIDTH) or row.dtype != np.dtype("<f4"):
        raise ValueError("cache sequence shape or dtype differs from the frozen source")
