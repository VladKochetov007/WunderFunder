"""Bounded causal sliding-window masks shared by the attention backends."""

from __future__ import annotations

import torch
from torch import Tensor

MASK_STORAGE_ALIGNMENT = 8


def _trim_window_keys(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    query_start: int,
    key_start: int,
    context_rows: int,
) -> tuple[Tensor, Tensor, int]:
    """Trim keys older than the first query's rolling context window."""
    past_rows = query_start - key_start
    if past_rows < 0:
        raise ValueError("key_start cannot be newer than query_start")
    first_key = max(0, past_rows - (context_rows - 1))
    end_key = min(key.shape[-2], past_rows + query.shape[-2])
    if first_key >= end_key:
        raise ValueError("causal attention requires at least one key for every query")
    return key[..., first_key:end_key, :], value[..., first_key:end_key, :], past_rows - first_key


def _mask_storage_rows(key_rows: int) -> int:
    """Round mask backing width for fused kernels while retaining exact visible K length."""
    return (
        (key_rows + MASK_STORAGE_ALIGNMENT - 1) // MASK_STORAGE_ALIGNMENT * MASK_STORAGE_ALIGNMENT
    )


def _window_mask(
    query_rows: int,
    key_rows: int,
    past_rows: int,
    context_rows: int,
    device: torch.device,
) -> Tensor:
    """Build the exact rolling mask with aligned backing stride and visible Q by K shape."""
    storage_rows = _mask_storage_rows(key_rows)
    query_offsets = torch.arange(query_rows, device=device)
    key_offsets = torch.arange(storage_rows, device=device)
    distance = past_rows + query_offsets[:, None] - key_offsets[None, :]
    allowed = (distance >= 0) & (distance < context_rows) & (key_offsets[None, :] < key_rows)
    return allowed[None, None, :, :][..., :key_rows]


def _additive_window_mask(
    query_rows: int,
    key_rows: int,
    past_rows: int,
    context_rows: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Represent the same exact visible window as a zero-or-negative-infinity bias."""
    allowed = _window_mask(query_rows, key_rows, past_rows, context_rows, device)
    storage_rows = _mask_storage_rows(key_rows)
    mask = torch.zeros((1, 1, query_rows, storage_rows), dtype=dtype, device=device)
    mask[..., :key_rows].masked_fill_(~allowed, float("-inf"))
    return mask[..., :key_rows]


def _relative_window_mask(
    query_index: Tensor, key_index: Tensor, past_rows: int, context_rows: int
) -> Tensor:
    """Apply the exact sliding-window rule using only bounded cache-relative offsets."""
    distance = past_rows + query_index - key_index
    return (distance >= 0) & (distance < context_rows)
