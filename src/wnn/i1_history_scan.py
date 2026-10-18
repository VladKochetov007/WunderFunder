from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from wnn.data import N_FEATURES
from wnn.i1_history import (
    I1_CHANNELS,
    I1_HISTORY_BLOCK_ROWS,
    I1_HISTORY_FAST_HALF_LIFE,
    I1_HISTORY_SLOW_HALF_LIFE,
    I1_HISTORY_WIDTH,
    I1_RAW_START,
    I1_RAW_STOP,
    _ema_weights,
    validate_i1_history_prior,
)

if TYPE_CHECKING:
    import torch


class I1HistoryTorchState(NamedTuple):
    """Fixed-size causal state with EMA means relative to the previous raw row."""

    mean32: torch.Tensor
    mean512: torch.Tensor
    previous: torch.Tensor
    energy512: torch.Tensor
    scan_rows: int


def _ema_block_numpy(
    values: np.ndarray,
    initial: np.ndarray,
    decay: np.float32,
    update: np.float32,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute every pre-update mean and the block's post-update means."""
    positions = np.arange(values.shape[1], dtype=np.float32)
    powers = np.power(decay, positions)
    cumulative = np.cumsum(values * (1.0 / powers)[None, :, None], axis=1, dtype=np.float32)
    post = (decay * powers)[None, :, None] * initial[:, None, :]
    post += (update * powers)[None, :, None] * cumulative
    previous = np.empty_like(post)
    previous[:, :1] = initial[:, None, :]
    if values.shape[1] > 1:
        previous[:, 1:] = post[:, :-1]
    return previous, post


def i1_history_features_numpy(raw: np.ndarray, prior: Mapping[str, object]) -> np.ndarray:
    """Build full-sequence causal residual and change-energy channels in NumPy."""
    checked = validate_i1_history_prior(prior)
    values = np.asarray(raw, dtype=np.float32)
    squeeze = values.ndim == 2
    if squeeze:
        values = values[None]
    if values.ndim != 3 or values.shape[-1] != N_FEATURES or values.shape[1] < 1:
        raise ValueError("Expected nonempty (T,112) or (B,T,112) raw features")
    if not np.isfinite(values).all():
        raise ValueError("Raw features must be finite")
    scale = np.asarray(checked["scale"], dtype=np.float32)
    fast_decay, fast_update = _ema_weights(I1_HISTORY_FAST_HALF_LIFE)
    slow_decay, slow_update = _ema_weights(I1_HISTORY_SLOW_HALF_LIFE)
    output = np.zeros((*values.shape[:2], I1_HISTORY_WIDTH), dtype=np.float32)
    if values.shape[1] > 1:
        raw_i1 = values[..., I1_RAW_START:I1_RAW_STOP]
        mean32 = np.zeros_like(raw_i1[:, 0])
        mean512 = np.zeros_like(raw_i1[:, 0])
        energy512 = np.zeros_like(mean32)
        previous_raw = raw_i1[:, 0]
        for start in range(1, values.shape[1], I1_HISTORY_BLOCK_ROWS):
            end = min(start + I1_HISTORY_BLOCK_ROWS, values.shape[1])
            current = raw_i1[:, start:end]
            anchor = previous_raw
            current_offset = current - anchor[:, None, :]
            delta = current - raw_i1[:, start - 1 : end - 1]
            previous32, mean32_values = _ema_block_numpy(
                current_offset, mean32, fast_decay, fast_update
            )
            previous512, mean512_values = _ema_block_numpy(
                current_offset, mean512, slow_decay, slow_update
            )
            _, energy_values = _ema_block_numpy(
                np.square(delta), energy512, slow_decay, slow_update
            )
            energy_features = np.log1p(np.sqrt(energy_values) / scale)
            output[:, start:end] = np.concatenate(
                (current_offset - previous32, current_offset - previous512, energy_features),
                axis=-1,
            )
            mean32 = mean32_values[:, -1] - current_offset[:, -1]
            mean512 = mean512_values[:, -1] - current_offset[:, -1]
            energy512 = energy_values[:, -1]
            previous_raw = current[:, -1]
    return output[0] if squeeze else output


def _ema_block_torch(
    values: torch.Tensor,
    initial: torch.Tensor,
    decay: float,
    update: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate one bounded float32 EMA block without a per-row launch loop."""
    import torch

    positions = torch.arange(values.shape[1], device=values.device, dtype=values.dtype)
    powers = torch.pow(values.new_tensor(decay), positions)
    weighted = values * powers.reciprocal()[None, :, None]
    cumulative = torch.cumsum(weighted, dim=1)
    post = (decay * powers)[None, :, None] * initial[:, None, :]
    post = post + (update * powers)[None, :, None] * cumulative
    previous = torch.cat((initial[:, None], post[:, :-1]), dim=1)
    return previous, post


def i1_history_features_torch_chunk(
    raw: torch.Tensor,
    prior: Mapping[str, object],
    state: I1HistoryTorchState | None = None,
    output: torch.Tensor | None = None,
) -> tuple[torch.Tensor, I1HistoryTorchState]:
    """Build one causal chunk and return state for the next chunk or sequence."""
    import torch

    if raw.ndim != 3 or raw.shape[-1] != N_FEATURES or raw.shape[1] < 1:
        raise ValueError("Expected nonempty raw input with shape (B,T,112)")
    if raw.dtype != torch.float32:
        raise ValueError("i1 history requires float32 raw inputs before autocast")
    checked = validate_i1_history_prior(prior)
    expected = (*raw.shape[:2], I1_HISTORY_WIDTH)
    if output is None:
        output = raw.new_empty(expected)
    elif output.shape != expected or output.dtype != raw.dtype or output.device != raw.device:
        raise ValueError("i1 history output buffer must match the raw batch, dtype and device")
    raw_i1 = raw[..., I1_RAW_START:I1_RAW_STOP]
    first_chunk = state is None
    if first_chunk:
        initial = raw_i1[:, 0]
        zero = torch.zeros_like(initial)
        state = I1HistoryTorchState(zero, zero, initial, torch.zeros_like(initial), 0)
        if raw.shape[1] == 1:
            output.zero_()
            return output, state
        values = raw_i1[:, 1:]
        previous_raw = raw_i1[:, :-1]
        deltas = values - previous_raw
        destination_offset = 1
        output[:, 0].zero_()
    else:
        expected_state_shape = (raw.shape[0], I1_CHANNELS)
        state_values = (state.mean32, state.mean512, state.previous, state.energy512)
        if any(
            value.shape != expected_state_shape
            or value.dtype != raw.dtype
            or value.device != raw.device
            for value in state_values
        ):
            raise ValueError("i1 history state must match the raw batch, dtype and device")
        values = raw_i1
        previous_raw = torch.cat((state.previous[:, None], raw_i1[:, :-1]), dim=1)
        deltas = values - previous_raw
        destination_offset = 0
    scale = torch.as_tensor(checked["scale"], device=raw.device, dtype=raw.dtype)
    fast_decay, fast_update = _ema_weights(I1_HISTORY_FAST_HALF_LIFE)
    slow_decay, slow_update = _ema_weights(I1_HISTORY_SLOW_HALF_LIFE)
    mean32, mean512, energy512 = state.mean32, state.mean512, state.energy512
    scan_rows = state.scan_rows
    start = 0
    while start < values.shape[1]:
        block_remaining = I1_HISTORY_BLOCK_ROWS - (scan_rows % I1_HISTORY_BLOCK_ROWS)
        end = min(start + block_remaining, values.shape[1])
        current, current_delta = values[:, start:end], deltas[:, start:end]
        anchor = previous_raw[:, start]
        current_offset = current - anchor[:, None, :]
        previous32, mean32_values = _ema_block_torch(
            current_offset, mean32, float(fast_decay), float(fast_update)
        )
        previous512, mean512_values = _ema_block_torch(
            current_offset, mean512, float(slow_decay), float(slow_update)
        )
        _, energy_values = _ema_block_torch(
            current_delta.square(), energy512, float(slow_decay), float(slow_update)
        )
        features = torch.cat(
            (
                current_offset - previous32,
                current_offset - previous512,
                torch.log1p(torch.sqrt(energy_values) / scale),
            ),
            dim=-1,
        )
        output[:, destination_offset + start : destination_offset + end].copy_(features)
        mean32, mean512, energy512 = (
            mean32_values[:, -1] - current_offset[:, -1],
            mean512_values[:, -1] - current_offset[:, -1],
            energy_values[:, -1],
        )
        scan_rows += end - start
        start = end
    next_state = I1HistoryTorchState(mean32, mean512, raw_i1[:, -1], energy512, scan_rows)
    return output, next_state
