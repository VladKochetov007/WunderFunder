"""Causal, groupwise fast/slow states from diagonal Kalman innovations."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch

MODE = "kalman_group_fast_slow"
FEATURE_GROUPS = (
    ("i0_bid_price_like", 0, 11),
    ("i0_ask_price_like", 11, 22),
    ("i0_bid_volume_like", 22, 33),
    ("i0_ask_volume_like", 33, 44),
    ("i0_trade_price", 44, 48),
    ("i0_trade_volume", 48, 52),
    ("i1_bid_price_like", 52, 63),
    ("i1_ask_price_like", 63, 74),
    ("i1_bid_volume_like", 74, 85),
    ("i1_ask_volume_like", 85, 96),
    ("i1_trade_price", 96, 100),
    ("i1_trade_volume", 100, 104),
    ("additional", 104, 112),
)
WARMUP_ROWS = 99
PROCESS_VARIANCE = 0.001
OBSERVATION_VARIANCE = 1.0
FAST_ALPHA = np.float32(2.0 / 21.0)
SLOW_ALPHA = np.float32(2.0 / 201.0)
LOG_EPSILON = np.float32(1e-8)
MAX_SEQUENCE_ROWS = 20_000
BLOCK_ROWS = 512
GROUP_COUNT = len(FEATURE_GROUPS)
FEATURE_WIDTH = 3 * GROUP_COUNT
STREAM_STATE_WIDTH = 3 * 112 + 2 * GROUP_COUNT + 2
FEATURE_NAMES = (
    tuple(f"fast_log_energy_{name}" for name, _, _ in FEATURE_GROUPS)
    + tuple(f"slow_log_energy_{name}" for name, _, _ in FEATURE_GROUPS)
    + tuple(f"fast_slow_log_ratio_{name}" for name, _, _ in FEATURE_GROUPS)
)


def _kalman_coefficients() -> tuple[np.ndarray, np.ndarray]:
    gains = np.empty(MAX_SEQUENCE_ROWS - WARMUP_ROWS, dtype=np.float64)
    innovation_scales = np.empty_like(gains)
    covariance = 1.0
    for index in range(len(gains)):
        predicted = covariance + PROCESS_VARIANCE
        gains[index] = predicted / (predicted + OBSERVATION_VARIANCE)
        innovation_scales[index] = math.sqrt(predicted + OBSERVATION_VARIANCE)
        covariance = (1.0 - gains[index]) * predicted
    return gains, innovation_scales


KALMAN_GAIN, INNOVATION_SCALE = _kalman_coefficients()


def _group_energy_numpy(innovation: np.ndarray) -> np.ndarray:
    clipped = np.clip(innovation, -8.0, 8.0) ** 2
    group_rms = np.asarray(
        [np.sqrt(clipped[start:end].mean()) for _, start, end in FEATURE_GROUPS],
        dtype=np.float32,
    )
    return np.square(group_rms.astype(np.float64))


def _encode_energy_numpy(energy: np.ndarray) -> np.ndarray:
    fast = energy[0].copy()
    slow = energy[0].copy()
    output = np.empty((len(energy), FEATURE_WIDTH), dtype=np.float32)
    for index, value in enumerate(energy):
        if index:
            fast += float(FAST_ALPHA) * (value - fast)
            slow += float(SLOW_ALPHA) * (value - slow)
        output[index] = np.concatenate(
            (
                np.log1p(fast),
                np.log1p(slow),
                np.log(fast + float(LOG_EPSILON)) - np.log(slow + float(LOG_EPSILON)),
            )
        )
    return output


def features_numpy(sequence: np.ndarray, control: str = "none") -> np.ndarray:
    """Return reset-local Kalman features, zero on the 99-row warmup prefix."""
    raw = np.asarray(sequence, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] != 112 or not WARMUP_ROWS < len(raw) <= MAX_SEQUENCE_ROWS:
        raise ValueError("expected 100 to 20,000 rows with 112 inputs")
    if not np.isfinite(raw).all():
        raise ValueError("raw inputs must be finite")
    if control not in ("none", "all"):
        raise ValueError("Kalman group-state control must be none or all")
    output = np.zeros((len(raw), FEATURE_WIDTH), dtype=np.float32)
    if control == "all":
        return output

    warmup = raw[:WARMUP_ROWS].astype(np.float64)
    center = warmup.mean(axis=0)
    scale = np.maximum(warmup.std(axis=0, ddof=1), 1e-6)
    level = np.zeros(112, dtype=np.float64)
    covariance = 1.0
    energies = np.empty((len(raw) - WARMUP_ROWS, GROUP_COUNT), dtype=np.float64)
    for index, row in enumerate(raw[WARMUP_ROWS:].astype(np.float64)):
        observation = (row - center) / scale
        predicted_covariance = covariance + PROCESS_VARIANCE
        innovation = (observation - level) / math.sqrt(predicted_covariance + OBSERVATION_VARIANCE)
        energies[index] = _group_energy_numpy(innovation)
        gain = predicted_covariance / (predicted_covariance + OBSERVATION_VARIANCE)
        level += gain * (observation - level)
        covariance = (1.0 - gain) * predicted_covariance
    output[WARMUP_ROWS:] = _encode_energy_numpy(energies)
    return output


class StreamingKalmanGroupFeatures:
    """Keep reset-local sufficient statistics for one-row causal serving."""

    def __init__(self, control: str = "none") -> None:
        if control not in ("none", "all"):
            raise ValueError("Kalman group-state control must be none or all")
        self.control = control
        self.reset()

    def reset(self) -> None:
        """Clear warmup, Kalman, and fast/slow state at a sequence boundary."""
        self.rows = 0
        self.warmup_sum = np.zeros(112, dtype=np.float64)
        self.warmup_square_sum = np.zeros(112, dtype=np.float64)
        self.level = np.zeros(112, dtype=np.float64)
        self.covariance = 1.0
        self.fast: np.ndarray | None = None
        self.slow: np.ndarray | None = None

    def update(self, row: np.ndarray) -> np.ndarray:
        """Return features for one row using only its prefix and current value."""
        raw = np.asarray(row, dtype=np.float32)
        if raw.shape != (112,) or not np.isfinite(raw).all():
            raise ValueError("expected one finite 112-wide float32 row")
        output = np.zeros(FEATURE_WIDTH, dtype=np.float32)
        if self.control == "all":
            self.rows = min(self.rows + 1, MAX_SEQUENCE_ROWS)
            if self.rows <= WARMUP_ROWS:
                self.warmup_sum += raw
                self.warmup_square_sum += raw.astype(np.float64) ** 2
            return output
        if self.rows < WARMUP_ROWS:
            value = raw.astype(np.float64)
            self.warmup_sum += value
            self.warmup_square_sum += value * value
            self.rows += 1
            return output

        center = self.warmup_sum / WARMUP_ROWS
        variance = (self.warmup_square_sum - self.warmup_sum * self.warmup_sum / WARMUP_ROWS) / (
            WARMUP_ROWS - 1
        )
        scale = np.maximum(np.sqrt(np.maximum(variance, 0.0)), 1e-6)
        observation = (raw.astype(np.float64) - center) / scale
        predicted_covariance = self.covariance + PROCESS_VARIANCE
        innovation = (observation - self.level) / math.sqrt(
            predicted_covariance + OBSERVATION_VARIANCE
        )
        energy = _group_energy_numpy(innovation)
        gain = predicted_covariance / (predicted_covariance + OBSERVATION_VARIANCE)
        self.level += gain * (observation - self.level)
        self.covariance = (1.0 - gain) * predicted_covariance
        if self.fast is None or self.slow is None:
            self.fast = energy.copy()
            self.slow = energy.copy()
        else:
            self.fast += float(FAST_ALPHA) * (energy - self.fast)
            self.slow += float(SLOW_ALPHA) * (energy - self.slow)
        output[:] = np.concatenate(
            (
                np.log1p(self.fast),
                np.log1p(self.slow),
                np.log(self.fast + float(LOG_EPSILON)) - np.log(self.slow + float(LOG_EPSILON)),
            )
        )
        self.rows = min(self.rows + 1, MAX_SEQUENCE_ROWS)
        return output


def _ew_block_torch(
    values: torch.Tensor, start: torch.Tensor, alpha: float
) -> tuple[torch.Tensor, torch.Tensor]:
    import torch

    base = values.new_tensor(alpha)
    row = torch.arange(values.shape[1], device=values.device, dtype=values.dtype)
    forward = torch.pow(base, row)
    inverse = torch.pow(base, -row)
    next_forward = forward * base
    encoded = next_forward[None, :, None] * start[:, None, :]
    encoded = encoded + (1.0 - alpha) * forward[None, :, None] * torch.cumsum(
        values * inverse[None, :, None], dim=1
    )
    return encoded, encoded[:, -1]


def _level_block_torch(
    observations: torch.Tensor, start: torch.Tensor, gains: torch.Tensor
) -> torch.Tensor:
    import torch

    alpha = 1.0 - gains
    product = torch.cumprod(alpha, dim=0)
    safe_product = product.clamp_min(1e-30)
    weighted = (1.0 - alpha)[None, :, None] * observations / safe_product[None, :, None]
    return product[None, :, None] * (start[:, None, :] + torch.cumsum(weighted, dim=1))


def features_torch(
    raw: torch.Tensor,
    output: torch.Tensor | None = None,
    control: str = "none",
) -> torch.Tensor:
    """Vectorized batched counterpart of :func:`features_numpy` in bounded scans."""
    import torch

    if raw.ndim != 3 or raw.shape[-1] != 112 or not WARMUP_ROWS < raw.shape[1] <= MAX_SEQUENCE_ROWS:
        raise ValueError("expected (batch, 100..20000, 112) raw inputs")
    if raw.dtype != torch.float32 or not torch.isfinite(raw).all():
        raise ValueError("raw inputs must be finite float32")
    if control not in ("none", "all"):
        raise ValueError("Kalman group-state control must be none or all")
    expected = (*raw.shape[:2], FEATURE_WIDTH)
    if output is None:
        output = raw.new_zeros(expected)
    elif output.shape != expected or output.dtype != raw.dtype or output.device != raw.device:
        raise ValueError("Kalman feature output must match raw batch, dtype, and device")
    else:
        output.zero_()
    if control == "all":
        return output

    work = raw.to(torch.float64)
    warmup = work[:, :WARMUP_ROWS]
    center = warmup.mean(dim=1)
    scale = warmup.std(dim=1, correction=1).clamp_min(1e-6)
    level = work.new_zeros((raw.shape[0], 112))
    fast: torch.Tensor | None = None
    slow: torch.Tensor | None = None
    gain_values = torch.as_tensor(KALMAN_GAIN, device=raw.device, dtype=work.dtype)
    innovation_scales = torch.as_tensor(INNOVATION_SCALE, device=raw.device, dtype=work.dtype)

    for start in range(WARMUP_ROWS, raw.shape[1], BLOCK_ROWS):
        end = min(start + BLOCK_ROWS, raw.shape[1])
        observations = (work[:, start:end] - center[:, None, :]) / scale[:, None, :]
        offset = start - WARMUP_ROWS
        gains = gain_values[offset : offset + end - start]
        scales = innovation_scales[offset : offset + end - start]
        levels = _level_block_torch(observations, level, gains)
        previous = torch.cat((level[:, None, :], levels[:, :-1, :]), dim=1)
        innovation = (observations - previous) / scales[None, :, None]
        innovation = innovation.clamp(-8.0, 8.0)
        group_rms = (
            torch.stack(
                [
                    innovation[..., left:right].square().mean(dim=-1)
                    for _, left, right in FEATURE_GROUPS
                ],
                dim=-1,
            )
            .clamp_min(0.0)
            .sqrt()
            .to(raw.dtype)
        )
        energy = group_rms.to(work.dtype).square()
        level = levels[:, -1]

        if fast is None or slow is None:
            first = energy[:, :1]
            fast_state, slow_state = first[:, 0], first[:, 0]
            energy = energy[:, 1:]
            if energy.shape[1]:
                fast_values, fast_state = _ew_block_torch(
                    energy, fast_state, 1.0 - float(FAST_ALPHA)
                )
                slow_values, slow_state = _ew_block_torch(
                    energy, slow_state, 1.0 - float(SLOW_ALPHA)
                )
                fast = torch.cat((first, fast_values), dim=1)
                slow = torch.cat((first, slow_values), dim=1)
            else:
                fast, slow = first, first
                fast_state, slow_state = first[:, 0], first[:, 0]
        else:
            fast_values, fast_state = _ew_block_torch(energy, fast, 1.0 - float(FAST_ALPHA))
            slow_values, slow_state = _ew_block_torch(energy, slow, 1.0 - float(SLOW_ALPHA))
            fast, slow = fast_values, slow_values

        fast_log = torch.log1p(fast)
        slow_log = torch.log1p(slow)
        ratio = torch.log(fast + float(LOG_EPSILON)) - torch.log(slow + float(LOG_EPSILON))
        output[:, start:end] = torch.cat((fast_log, slow_log, ratio), dim=-1).to(raw.dtype)
        fast, slow = fast_state, slow_state
    return output


def initial_stream_state_torch(
    batch_size: int = 1,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Create a compact reset state for the one-row streaming graph."""
    import torch

    if dtype is None:
        dtype = torch.float64
    return torch.zeros((batch_size, STREAM_STATE_WIDTH), device=device, dtype=dtype)


def features_torch_step(
    raw: torch.Tensor,
    state: torch.Tensor,
    control: str = "none",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Update one row of the reset-local Kalman feature state."""
    import torch

    if raw.ndim != 2 or raw.shape[-1] != 112 or raw.dtype != torch.float32:
        raise ValueError("expected a float32 (batch,112) raw row")
    if state.shape != (raw.shape[0], STREAM_STATE_WIDTH) or state.dtype != torch.float64:
        raise ValueError("Kalman stream state has an incompatible shape or dtype")
    if control not in ("none", "all"):
        raise ValueError("Kalman group-state control must be none or all")

    work = raw.to(torch.float64)
    warmup_sum = state[:, :112]
    warmup_square_sum = state[:, 112:224]
    count = state[:, 224]
    level = state[:, 225:337]
    fast = state[:, 337 : 337 + GROUP_COUNT]
    slow = state[:, 337 + GROUP_COUNT : 337 + 2 * GROUP_COUNT]
    initialized = state[:, -1]
    warming = count < WARMUP_ROWS

    next_sum = warmup_sum + torch.where(warming[:, None], work, torch.zeros_like(work))
    next_square_sum = warmup_square_sum + torch.where(
        warming[:, None], work.square(), torch.zeros_like(work)
    )
    center = next_sum / WARMUP_ROWS
    variance = (next_square_sum - next_sum.square() / WARMUP_ROWS) / (WARMUP_ROWS - 1)
    scale = variance.clamp_min(1e-12).sqrt()
    observation = (work - center) / scale

    filter_index = (count - WARMUP_ROWS).to(torch.long).clamp(0, len(KALMAN_GAIN) - 1)
    gains = torch.as_tensor(KALMAN_GAIN, device=raw.device, dtype=work.dtype)[filter_index]
    innovation_scales = torch.as_tensor(INNOVATION_SCALE, device=raw.device, dtype=work.dtype)[
        filter_index
    ]
    innovation = ((observation - level) / innovation_scales[:, None]).clamp(-8.0, 8.0)
    group_rms = (
        torch.stack(
            [innovation[:, left:right].square().mean(dim=-1) for _, left, right in FEATURE_GROUPS],
            dim=-1,
        )
        .clamp_min(0.0)
        .sqrt()
        .to(raw.dtype)
    )
    group_energy = group_rms.to(work.dtype).square()
    next_level = level + gains[:, None] * (observation - level)
    first_energy = initialized < 0.5
    next_fast = torch.where(
        first_energy[:, None],
        group_energy,
        (1.0 - float(FAST_ALPHA)) * fast + float(FAST_ALPHA) * group_energy,
    )
    next_slow = torch.where(
        first_energy[:, None],
        group_energy,
        (1.0 - float(SLOW_ALPHA)) * slow + float(SLOW_ALPHA) * group_energy,
    )
    features = torch.cat(
        (
            torch.log1p(next_fast),
            torch.log1p(next_slow),
            torch.log(next_fast + float(LOG_EPSILON)) - torch.log(next_slow + float(LOG_EPSILON)),
        ),
        dim=-1,
    )
    features = torch.where(warming[:, None], torch.zeros_like(features), features)
    if control == "all":
        features.zero_()

    updated_level = torch.where(warming[:, None], level, next_level)
    updated_fast = torch.where(warming[:, None], fast, next_fast)
    updated_slow = torch.where(warming[:, None], slow, next_slow)
    updated_initialized = torch.where(warming, initialized, torch.ones_like(initialized))
    updated_count = torch.clamp(count + 1.0, max=float(MAX_SEQUENCE_ROWS))
    next_state = torch.cat(
        (
            next_sum,
            next_square_sum,
            updated_count[:, None],
            updated_level,
            updated_fast,
            updated_slow,
            updated_initialized[:, None],
        ),
        dim=-1,
    )
    return features.to(raw.dtype), next_state
