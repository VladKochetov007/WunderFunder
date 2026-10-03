from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

HALF_LIVES = (4, 16, 64)
DELAY = 512


def ema(values: np.ndarray, half_life: int) -> np.ndarray:
    """Smooth a stream with zero initial state and no lookahead."""
    decay = np.exp(-np.log(2) / half_life)
    return lfilter([1 - decay], [1, -decay], values, axis=0)


def history_features(features: np.ndarray) -> dict[str, np.ndarray]:
    """Build causal histories exclusively from raw feature rows.

    Each sequence starts with zero increments. Event smoothing changes only on
    i0 price-state events. Column indices are identifiers, not assumed depths.
    No target scale, target-selected event, or score mask is accepted here.
    """
    prices = features[:, :22]
    increments = np.diff(prices, axis=0, prepend=prices[:1]).astype(np.float64)
    events = np.any(increments != 0, axis=1)
    rows = np.arange(len(features))
    anchors = np.r_[0, np.flatnonzero(events)]
    last_event = np.maximum.accumulate(np.where(events, rows, 0))
    age = rows - last_event
    event_index = np.cumsum(events)
    event_memory = np.hstack(
        [ema(increments[anchors], half_life)[event_index] for half_life in HALF_LIVES]
    )
    row_memory = np.hstack([ema(increments, half_life) for half_life in HALF_LIVES])
    delayed = np.zeros_like(event_memory)
    delayed[DELAY:] = event_memory[:-DELAY]
    rate = ema(events.astype(float), 64)
    clock = np.column_stack([np.log1p(age), np.log(rate + 1e-4)])
    instant_clock = np.column_stack([features, increments, clock])
    return {
        "current": features.astype(np.float64),
        "current_instant_clock": instant_clock,
        "event_memory": np.column_stack([instant_clock, event_memory]),
        "row_memory": np.column_stack([instant_clock, row_memory]),
        "delayed_event_memory": np.column_stack([instant_clock, delayed]),
        "age": age,
        "rate": rate,
        "events": events,
    }


def sufficient_statistics(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Return exact additive weighted-Pearson moments for two targets."""
    y = np.clip(target.astype(np.float64), -2, 2)
    p = np.clip(prediction, -2, 2)
    w = np.abs(y)
    return np.stack(
        [
            w.sum(0),
            (w * y).sum(0),
            (w * p).sum(0),
            (w * y * y).sum(0),
            (w * p * p).sum(0),
            (w * y * p).sum(0),
        ],
        axis=-1,
    )


def weighted_correlations(stats: np.ndarray) -> np.ndarray:
    """Calculate pooled correlations without treating rows as replicates."""
    w, y, p, yy, pp, yp = np.moveaxis(stats, -1, 0)
    covariance = yp - y * p / np.maximum(w, 1e-12)
    variance = (yy - y * y / np.maximum(w, 1e-12)) * (pp - p * p / np.maximum(w, 1e-12))
    return covariance / np.sqrt(np.maximum(variance, 1e-24))
