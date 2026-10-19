"""R2's fixed views, assembled from the causal R1 feature transform."""

from __future__ import annotations

import numpy as np
from idea_screens_r1.feature_views import build_view

VIEW_NAMES = (
    "raw",
    "i0_only",
    "raw_lags_1_16_128",
    "raw_signed_delta_ema_hl32",
    "raw_absolute_innovation_ema_hl32",
    "raw_both_ema_parts_hl32",
)


def build_views(raw: np.ndarray) -> dict[str, np.ndarray]:
    """Build all six frozen R2 representations from one complete sequence.

    Args:
        raw: Canonical reset-local F0 rows with shape ``(T,112)``.

    Returns:
        Float32 features keyed by the fixed view names. The three EMA controls
        share R1's single pass and differ only by included causal channels.
    """
    x = np.asarray(raw, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 112 or len(x) == 0:
        raise ValueError("expected nonempty (T,112) canonical F0 features")
    raw_features, _ = build_view(x, "raw")
    i0_features, _ = build_view(x, "i0_only")
    lag_features, _ = build_view(x, "raw_lags_1_16_128")
    ema_features, _ = build_view(x, "raw_net_absinnovation_ema")
    signed_features = np.concatenate((ema_features[:, :112], ema_features[:, 112:224]), axis=1)
    absolute_features = np.concatenate((ema_features[:, :112], ema_features[:, 224:336]), axis=1)
    return {
        "raw": raw_features,
        "i0_only": i0_features,
        "raw_lags_1_16_128": lag_features,
        "raw_signed_delta_ema_hl32": signed_features,
        "raw_absolute_innovation_ema_hl32": absolute_features,
        "raw_both_ema_parts_hl32": ema_features,
    }


def build_view_for(raw: np.ndarray, name: str) -> np.ndarray:
    """Build one fixed R2 view without allocating the other full-row matrices."""
    if name not in VIEW_NAMES:
        raise ValueError(f"unknown R2 view: {name}")
    if name == "raw_signed_delta_ema_hl32" or name == "raw_absolute_innovation_ema_hl32":
        ema_features, _ = build_view(raw, "raw_net_absinnovation_ema")
        if name == "raw_signed_delta_ema_hl32":
            return np.concatenate((ema_features[:, :112], ema_features[:, 112:224]), axis=1)
        return np.concatenate((ema_features[:, :112], ema_features[:, 224:336]), axis=1)
    r1_name = {
        "raw": "raw",
        "i0_only": "i0_only",
        "raw_lags_1_16_128": "raw_lags_1_16_128",
        "raw_both_ema_parts_hl32": "raw_net_absinnovation_ema",
    }[name]
    return build_view(raw, r1_name)[0]
