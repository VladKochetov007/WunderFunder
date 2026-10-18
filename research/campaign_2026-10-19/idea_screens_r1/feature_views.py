"""Fixed, causal feature views for bounded linear screening."""

from __future__ import annotations

import numpy as np

VIEW_NAMES = (
    "raw",
    "raw_lags_1_16_128",
    "raw_nonnegative_volume_proportions_log_total",
    "raw_row_shape_location_scale",
    "raw_rolling_beta_residual",
    "raw_net_absinnovation_ema",
    "raw_warmup99_context",
    "i0_only",
    "i1_only",
)

_N_RAW = 112
_GROUPS = (
    slice(0, 22),
    slice(22, 44),
    slice(44, 48),
    slice(48, 52),
    slice(52, 74),
    slice(74, 96),
    slice(96, 100),
    slice(100, 104),
    slice(104, 112),
)
_VOLUME_SLICES = (slice(22, 44), slice(48, 52), slice(74, 96), slice(100, 104))
_EMA_ALPHA = 1.0 - 0.5 ** (1.0 / 128.0)
_INNOVATION_ALPHA = 1.0 - 0.5 ** (1.0 / 32.0)


def _validate(raw: np.ndarray) -> np.ndarray:
    """Validate one complete reset-local canonical F0 sequence."""
    x = np.asarray(raw, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != _N_RAW or x.shape[0] < 1:
        raise ValueError("expected nonempty (T,112) canonical F0 features")
    if not np.isfinite(x).all():
        raise ValueError("canonical F0 features must be finite")
    return x


def _lag_view(x: np.ndarray) -> np.ndarray:
    """Append three zero-initialized causal raw lags without dropping rows."""
    n = len(x)
    lags = np.zeros((n, 3 * _N_RAW), dtype=np.float32)
    for index, lag in enumerate((1, 16, 128)):
        if n > lag:
            lags[lag:, index * _N_RAW : (index + 1) * _N_RAW] = x[:-lag]
    return np.concatenate((x, lags), axis=1)


def _volume_view(x: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    """Append per-instrument volume shapes and log totals on valid rows only."""
    blocks: list[np.ndarray] = [x]
    counts: list[int] = []
    for left, right in (
        (_VOLUME_SLICES[0], _VOLUME_SLICES[1]),
        (_VOLUME_SLICES[2], _VOLUME_SLICES[3]),
    ):
        volume = np.concatenate((x[:, left], x[:, right]), axis=1)
        total = volume.astype(np.float64).sum(axis=1)
        valid = np.all(volume >= 0.0, axis=1) & (total > 0.0)
        shape = np.zeros_like(volume)
        np.divide(volume, total[:, None], out=shape, where=valid[:, None])
        log_total = np.zeros((len(x), 1), dtype=np.float32)
        log_total[valid, 0] = np.log1p(total[valid]).astype(np.float32)
        blocks.extend((shape, log_total))
        counts.append(int(valid.sum()))
    return np.concatenate(blocks, axis=1), {
        "volume_i0_valid_rows": counts[0],
        "volume_i1_valid_rows": counts[1],
        "volume_rows": len(x),
    }


def _row_shape_view(x: np.ndarray) -> np.ndarray:
    """Append within-group row shape, location and log scale features."""
    shape = np.empty_like(x)
    locations: list[np.ndarray] = []
    log_scales: list[np.ndarray] = []
    for columns in _GROUPS:
        block = x[:, columns].astype(np.float64)
        location = block.mean(axis=1, keepdims=True)
        scale = block.std(axis=1, keepdims=True)
        shape[:, columns] = ((block - location) / np.maximum(scale, 1e-6)).astype(np.float32)
        locations.append(location.astype(np.float32))
        log_scales.append(np.log1p(scale).astype(np.float32))
    return np.concatenate((x, shape, *locations, *log_scales), axis=1)


def _beta_view(x: np.ndarray) -> np.ndarray:
    """Append bounded past-only beta, common movement and residual summaries."""
    n = len(x)
    movement = np.zeros((n, 2), dtype=np.float64)
    if n > 1:
        movement[1:, 0] = np.diff(x[:, :22].astype(np.float64).mean(axis=1))
        movement[1:, 1] = np.diff(x[:, 52:74].astype(np.float64).mean(axis=1))
    summaries = np.zeros((n, 5), dtype=np.float32)
    mean0 = mean1 = covariance = variance1 = 0.0
    decay = 1.0 - _EMA_ALPHA
    for row, (move0, move1) in enumerate(movement):
        beta = 0.0 if variance1 <= 1e-12 else float(np.clip(covariance / variance1, -8.0, 8.0))
        common = beta * move1
        summaries[row] = (move0, move1, beta, common, move0 - common)
        delta0, delta1 = move0 - mean0, move1 - mean1
        covariance = decay * (covariance + _EMA_ALPHA * delta0 * delta1)
        variance1 = decay * (variance1 + _EMA_ALPHA * delta1 * delta1)
        mean0 += _EMA_ALPHA * delta0
        mean1 += _EMA_ALPHA * delta1
    return np.concatenate((x, summaries), axis=1)


def _innovation_view(x: np.ndarray) -> np.ndarray:
    """Append reset-local signed-delta EMA and absolute-innovation EMA."""
    n = len(x)
    summaries = np.zeros((n, 2 * _N_RAW), dtype=np.float32)
    previous = x[0].astype(np.float64)
    net = np.zeros(_N_RAW, dtype=np.float64)
    absolute = np.zeros(_N_RAW, dtype=np.float64)
    decay = 1.0 - _INNOVATION_ALPHA
    for row in range(1, n):
        current = x[row].astype(np.float64)
        delta = current - previous
        innovation = delta - net
        net = decay * net + _INNOVATION_ALPHA * delta
        absolute = decay * absolute + _INNOVATION_ALPHA * np.abs(innovation)
        summaries[row, :_N_RAW] = net
        summaries[row, _N_RAW:] = absolute
        previous = current
    return np.concatenate((x, summaries), axis=1)


def _warmup_view(x: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    """Hold the input-only mean of rows 0..98 as a constant from row 99."""
    context = np.zeros_like(x)
    available = min(len(x), 99)
    if len(x) > 99:
        context[99:] = x[:99].mean(axis=0, dtype=np.float64).astype(np.float32)
    return np.concatenate((x, context), axis=1), {"warmup_context_rows": available}


def build_view(raw: np.ndarray, name: str) -> tuple[np.ndarray, dict[str, int]]:
    """Build one predeclared reset-local feature matrix for a full sequence.

    Args:
        raw: Canonical F0 features with shape ``(T,112)``.
        name: One of :data:`VIEW_NAMES`.

    Returns:
        A float32 matrix and small transform metadata. All histories reset at row 0.

    Raises:
        ValueError: The input or feature view name is invalid.
    """
    x = _validate(raw)
    if name == "raw":
        return x, {}
    if name == "raw_lags_1_16_128":
        return _lag_view(x), {}
    if name == "raw_nonnegative_volume_proportions_log_total":
        return _volume_view(x)
    if name == "raw_row_shape_location_scale":
        return _row_shape_view(x), {}
    if name == "raw_rolling_beta_residual":
        return _beta_view(x), {}
    if name == "raw_net_absinnovation_ema":
        return _innovation_view(x), {}
    if name == "raw_warmup99_context":
        return _warmup_view(x)
    if name == "i0_only":
        return x[:, :52], {}
    if name == "i1_only":
        return x[:, 52:104], {}
    raise ValueError(f"unknown fixed feature view: {name}")
