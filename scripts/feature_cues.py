from __future__ import annotations

import numpy as np

from wnn.research_features import ENDPOINT, research_features_numpy

CUE_NAMES = (
    "regime_offsets",
    "regime_heads",
    "target_clocks",
    "activity_interactions",
    "i1_innovations",
    "event_age",
    "observable_transitions",
    "cycle_smoothing",
    "volume28",
)


def lagged(values: np.ndarray, lag: int) -> np.ndarray:
    """Causal lag with explicit zero prehistory; never wrap sequence boundaries."""
    if lag < 1:
        raise ValueError("Lag must be positive")
    result = np.zeros_like(values)
    if lag < len(values):
        result[lag:] = values[:-lag]
    return result


def _ema(values: np.ndarray, horizon: int) -> np.ndarray:
    result = np.empty_like(values)
    state = np.zeros(values.shape[-1], dtype=np.float64)
    alpha = np.exp2(-1 / horizon)
    for index, value in enumerate(values):
        state = alpha * state + (1 - alpha) * value
        result[index] = state
    return result


def causal_cues(raw: np.ndarray) -> dict[str, np.ndarray]:
    """Fixed input-only feature families; names do not assert physical target clocks."""
    x = np.asarray(raw, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 112 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Expected one finite raw sequence of width112")
    delta = np.diff(x, axis=0, prepend=x[:1])
    endpoint = x[:, 110] == ENDPOINT
    any_i0 = np.any(delta[:, :22] != 0, axis=-1)
    any_i1 = np.any(delta[:, 52:74] != 0, axis=-1)
    trade = np.any(delta[:, 44:52] != 0, axis=-1)
    flags = np.stack((endpoint, any_i0, any_i1, trade), axis=-1)
    flags[0] = False
    index = np.arange(len(x))[:, None]
    last = np.maximum.accumulate(np.where(flags, index, 0), axis=0)
    age = np.log1p(index - last).astype(np.float32)
    auxiliary = np.tanh(x[:, 104:112])
    step_volume = np.stack((np.abs(delta[:, 22:33]).sum(-1), np.abs(delta[:, 33:44]).sum(-1)), -1)
    i1 = delta[:, 52:96]
    state = np.concatenate((delta[:, :22], delta[:, 52:74], step_volume, flags), -1)
    slow, fast = _ema(state, 32), _ema(state, 8)
    innovations = i1 - lagged(_ema(i1, 32), 1)
    interactions = (
        auxiliary[:, [2, 3, 5, 6], None] * np.concatenate((step_volume, age), -1)[:, None, :]
    ).reshape(len(x), -1)
    result = {
        "regime_offsets": auxiliary,
        "regime_heads": np.concatenate((auxiliary, auxiliary**2), -1),
        "target_clocks": np.concatenate((age, flags, x[:, [109, 110]]), -1),
        "activity_interactions": interactions,
        "i1_innovations": np.concatenate((innovations, lagged(innovations, 8)), -1),
        "event_age": np.concatenate((age, age**2, flags), -1),
        "observable_transitions": np.concatenate((state, lagged(state, 1)), -1),
        "cycle_smoothing": np.concatenate((fast, slow, fast - slow), -1),
        "volume28": research_features_numpy(x, "volume28"),
    }
    return {name: values.astype(np.float32) for name, values in result.items()}
