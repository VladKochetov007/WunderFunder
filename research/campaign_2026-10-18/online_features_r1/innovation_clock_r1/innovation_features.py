"""Small causal feature extractor for an all-channel Kalman innovation clock."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

N_FEATURES = 112
WARMUP_ROWS = 99
PROCESS_VARIANCE = 0.001
OBSERVATION_VARIANCE = 1.0
CUSUM_DECAY = 0.95
CUSUM_BASELINE = 1.0
CUSUM_CAP = 20.0

FEATURE_GROUPS: dict[str, tuple[int, int]] = {
    "i0_bid_price_like": (0, 11),
    "i0_ask_price_like": (11, 22),
    "i0_bid_volume_like": (22, 33),
    "i0_ask_volume_like": (33, 44),
    "i0_trade_price": (44, 48),
    "i0_trade_volume": (48, 52),
    "i1_bid_price_like": (52, 63),
    "i1_ask_price_like": (63, 74),
    "i1_bid_volume_like": (74, 85),
    "i1_ask_volume_like": (85, 96),
    "i1_trade_price": (96, 100),
    "i1_trade_volume": (100, 104),
    "additional": (104, 112),
}

FloatArray = NDArray[np.float32]


def innovation_clock(sequence: np.ndarray) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return all-channel surprise, semantic-group surprise, and one scalar clock.

    The first 99 observed rows initialize per-sequence centering/scaling. For
    every later row, innovations are computed before the Kalman state update.
    The returned scalar clock is suitable for appending while retaining raw
    inputs; it is not a fitted feature or a quality claim.
    """
    values = np.asarray(sequence)
    if values.ndim != 2 or values.shape[1] != N_FEATURES or not WARMUP_ROWS < len(values) <= 20_000:
        raise ValueError(
            f"expected between {WARMUP_ROWS + 1} and 20,000 rows of {N_FEATURES} features"
        )
    if not np.isfinite(values).all():
        raise ValueError("feature sequence contains non-finite values")

    warmup = values[:WARMUP_ROWS].astype(np.float64, copy=False)
    center = warmup.mean(axis=0)
    scale = np.maximum(warmup.std(axis=0, ddof=1), 1e-6)
    observations = (values[WARMUP_ROWS:].astype(np.float64) - center) / scale

    level = np.zeros(N_FEATURES, dtype=np.float64)
    covariance = np.ones(N_FEATURES, dtype=np.float64)
    innovation_rms = np.empty(len(observations), dtype=np.float32)
    group_rms = np.empty((len(observations), len(FEATURE_GROUPS)), dtype=np.float32)
    clock = np.empty(len(observations), dtype=np.float32)
    starts = np.fromiter((bounds[0] for bounds in FEATURE_GROUPS.values()), dtype=np.int64)
    widths = np.fromiter(
        (bounds[1] - bounds[0] for bounds in FEATURE_GROUPS.values()), dtype=np.float64
    )
    clock_state = 0.0

    for row_index, observation in enumerate(observations):
        predicted_covariance = covariance + PROCESS_VARIANCE
        residual = observation - level
        standardized = residual / np.sqrt(predicted_covariance + OBSERVATION_VARIANCE)
        clipped_squared = np.square(np.clip(standardized, -8.0, 8.0))
        channel_mean_square = clipped_squared.mean()
        innovation_rms[row_index] = np.sqrt(channel_mean_square)
        group_mean_square = np.add.reduceat(clipped_squared, starts) / widths
        group_rms[row_index] = np.sqrt(group_mean_square)

        clock_state = min(
            CUSUM_CAP,
            max(0.0, CUSUM_DECAY * clock_state + float(innovation_rms[row_index]) - CUSUM_BASELINE),
        )
        clock[row_index] = np.log1p(clock_state)

        gain = predicted_covariance / (predicted_covariance + OBSERVATION_VARIANCE)
        level += gain * residual
        covariance = (1.0 - gain) * predicted_covariance

    if not (
        np.isfinite(innovation_rms).all()
        and np.isfinite(group_rms).all()
        and np.isfinite(clock).all()
    ):
        raise FloatingPointError("innovation clock produced a non-finite value")
    return innovation_rms, group_rms, clock
