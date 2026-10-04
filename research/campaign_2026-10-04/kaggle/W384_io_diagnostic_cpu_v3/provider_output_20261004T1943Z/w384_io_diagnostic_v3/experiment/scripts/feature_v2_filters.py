from __future__ import annotations

import numpy as np
from scipy.signal import lfilter


def ema(values: np.ndarray, alpha: float) -> np.ndarray:
    """Causal row EMA initialized to the first observation, matching the archived EDA."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or not len(x) or not 0 < alpha <= 1 or not np.isfinite(x).all():
        raise ValueError("EMA requires a finite nonempty matrix and alpha in (0,1]")
    result, _ = lfilter([alpha], [1.0, -(1.0 - alpha)], x, axis=0, zi=(1.0 - alpha) * x[:1])
    return result.astype(np.float32)


def trailing_mean(values: np.ndarray, period: int) -> np.ndarray:
    """Average a trailing cycle, padding only with the first observed row."""
    if period < 1:
        raise ValueError("Positive cycle period required")
    x = np.asarray(values, dtype=np.float64)
    padded = np.concatenate((np.repeat(x[:1], period - 1, axis=0), x))
    cumulative = np.concatenate((np.zeros((1, x.shape[1])), np.cumsum(padded, axis=0)))
    return ((cumulative[period:] - cumulative[:-period]) / period).astype(np.float32)


def phase_ema(values: np.ndarray, horizon: float, period: int) -> np.ndarray:
    """Independent same-phase histories with matched exponential row-time decay."""
    result = np.empty_like(values, dtype=np.float32)
    alpha = 1 - (1 - 1 / horizon) ** period
    for phase in range(min(period, len(values))):
        result[phase::period] = ema(values[phase::period], alpha)
    return result


def lag(values: np.ndarray, rows: int) -> np.ndarray:
    """Zero prehistory, with no sequence wrap or future access."""
    if rows < 1:
        raise ValueError("Positive lag required")
    result = np.zeros_like(values)
    if rows < len(values):
        result[rows:] = values[:-rows]
    return result


def warmup_summary(raw: np.ndarray) -> np.ndarray:
    """Freeze regime-router inputs from exactly rows0..98; later rows cannot affect them."""
    x = np.asarray(raw, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 112 or len(x) < 99 or not np.isfinite(x[:99]).all():
        raise ValueError("Warmup needs99 finite raw rows of width112")
    prefix = x[:99]
    return np.concatenate(
        (prefix.mean(0), prefix.std(0), np.abs(np.diff(prefix, axis=0)).mean(0))
    ).astype(np.float32)


def histories(raw: np.ndarray) -> dict[str, np.ndarray]:
    """Exact reviewed history families, retaining their mechanism-specific controls."""
    x = np.asarray(raw, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 112 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Expected a finite raw sequence of width112")
    price = x[:, :22]
    event = np.r_[True, np.any(price[1:] != price[:-1], axis=1)]
    row = ema(price, 2 / 33)
    event_history = ema(price[event], 2 / 33)[np.cumsum(event) - 1]
    event_displacement = price - event_history
    delta = np.diff(price.astype(np.float64), axis=0, prepend=price[:1])
    rate = event.astype(np.float32)[:, None]
    rate[0] = 0
    variance = np.mean(delta**2, axis=1, keepdims=True)
    activity = np.concatenate(
        [ema(rate, 1 / h) for h in (32, 256)] + [np.log1p(ema(variance, 1 / h)) for h in (32, 256)],
        axis=1,
    )
    indices = np.arange(len(x))
    last = np.maximum.accumulate(np.where(event, indices, 0))
    age = np.log1p(indices - last).astype(np.float32)[:, None]
    values = {
        "clock_row": price - row,
        "clock_event": event_displacement,
        "clock_stale_event": lag(event_displacement, 4096),
        "activity": activity,
        "activity_stale": lag(activity, 128),
        "event_age": age,
        "event_age_stale": lag(age, 128),
    }
    cycle8, cycle7 = trailing_mean(x, 8), trailing_mean(x, 7)
    # Cycle8 adds3.5 rows of mean lag. Match that lag in the row and cycle7 controls.
    values.update(
        cycle8=np.concatenate([ema(cycle8, 1 / h) for h in (32, 256)], axis=1),
        cycle7_matched=np.concatenate([ema(cycle7, 1 / (h + 0.5)) for h in (32, 256)], axis=1),
        row_matched=np.concatenate([ema(x, 1 / (h + 3.5)) for h in (32, 256)], axis=1),
        row_original=np.concatenate([ema(x, 1 / h) for h in (32, 256)], axis=1),
        phase8=np.concatenate([phase_ema(x, h, 8) for h in (32, 256)], axis=1),
        phase7=np.concatenate([phase_ema(x, h, 7) for h in (32, 256)], axis=1),
        raw=x,
        transition=np.concatenate((x - lag(x, 1), x * lag(x, 1)), axis=1),
    )
    values["transition"][0] = 0
    values["transition_stale"] = lag(values["transition"], 128)
    # Store dense causal lag inputs so learned innovations can be fitted on fit rows only.
    for offset in (0, 1, 8, 9, 136, 137):
        values[f"i0_lag{offset}"] = x[:, :52] if offset == 0 else lag(x[:, :52], offset)
        values[f"i1_lag{offset}"] = x[:, 52:104] if offset == 0 else lag(x[:, 52:104], offset)
    return values


def fit_router(
    features: np.ndarray, labels: np.ndarray, ridge: float = 0.01
) -> dict[str, np.ndarray]:
    """Fit one prespecified four-class router; the caller enforces TRAIN-only labels."""
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels)
    if (
        x.ndim != 2
        or y.ndim != 1
        or y.dtype.kind not in "iu"
        or len(x) != len(y)
        or set(y.tolist()) != set(range(4))
        or ridge <= 0
    ):
        raise ValueError("Router requires aligned features and all four TRAIN classes")
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite router input")
    mean, scale = x.mean(0), np.maximum(x.std(0), 1e-4)
    design = np.column_stack((np.ones(len(x)), np.clip((x - mean) / scale, -8, 8)))
    weights = 1 / np.bincount(y, minlength=4)[y]
    weights /= weights.sum()
    gram = design.T @ (design * weights[:, None]) + ridge * np.eye(design.shape[1])
    beta = np.linalg.solve(gram, design.T @ (np.eye(4)[y] * weights[:, None]))
    return {"mean": mean, "scale": scale, "beta": beta}


def route(features: np.ndarray, parameters: dict[str, np.ndarray]) -> np.ndarray:
    """Return fixed hard routing classes using warmup inputs alone."""
    design = np.column_stack(
        (
            np.ones(len(features)),
            np.clip((features - parameters["mean"]) / parameters["scale"], -8, 8),
        )
    )
    return (design @ parameters["beta"]).argmax(axis=1)
