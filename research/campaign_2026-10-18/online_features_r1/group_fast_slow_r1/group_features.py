"""Build causal, per-group fast/slow Kalman innovation features."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[4]
INNOVATION_DIR = ROOT / "research/campaign_2026-10-18/online_features_r1/innovation_clock_r1"
sys.path.insert(0, str(INNOVATION_DIR))

from innovation_features import FEATURE_GROUPS, WARMUP_ROWS, innovation_clock  # noqa: E402

FloatArray = NDArray[np.float32]
GROUP_COUNT = len(FEATURE_GROUPS)
FAST_SPAN = 20
SLOW_SPAN = 200
FAST_ALPHA = 2.0 / (FAST_SPAN + 1.0)
SLOW_ALPHA = 2.0 / (SLOW_SPAN + 1.0)
LOG_EPSILON = 1e-8


def extract_online_features(sequence: np.ndarray) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return raw inputs, scalar clock and 39 group-level state features.

    Each output row corresponds to the same raw input row, starting at the
    first post-warmup observation. The fast/slow states include the current
    row's input innovation and use no target, scoring mask or future row.
    """
    values = np.asarray(sequence)
    if values.ndim != 2 or values.shape[1] != 112 or not WARMUP_ROWS < len(values) <= 20_000:
        raise ValueError("expected 100 to 20,000 rows with 112 inputs")
    _, group_rms, clock = innovation_clock(values)
    energy = np.square(group_rms.astype(np.float64, copy=False))
    state_features = np.empty((len(energy), GROUP_COUNT * 3), dtype=np.float32)
    fast = energy[0].copy()
    slow = energy[0].copy()

    for row_index, observation in enumerate(energy):
        if row_index:
            fast += FAST_ALPHA * (observation - fast)
            slow += SLOW_ALPHA * (observation - slow)
        start = 0
        state_features[row_index, start : start + GROUP_COUNT] = np.log1p(fast)
        start += GROUP_COUNT
        state_features[row_index, start : start + GROUP_COUNT] = np.log1p(slow)
        start += GROUP_COUNT
        state_features[row_index, start:] = np.log(fast + LOG_EPSILON) - np.log(slow + LOG_EPSILON)

    raw = np.asarray(values[WARMUP_ROWS:], dtype=np.float32)
    clock_column = clock.reshape(-1, 1)
    if not (
        np.isfinite(raw).all()
        and np.isfinite(clock_column).all()
        and np.isfinite(state_features).all()
    ):
        raise FloatingPointError("group innovation features contain non-finite values")
    return raw, clock_column, state_features
