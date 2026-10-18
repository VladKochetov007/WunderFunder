"""Nine fixed R3 feature/budget cases built from reset-local R1 histories."""

from __future__ import annotations

import numpy as np
from idea_screens_r1.feature_views import build_view

FULL64_CASES = (
    "full64_i0_52",
    "full64_i0_tail60",
    "full64_books104",
    "full64_raw112",
    "full64_raw_signed_ema_hl32",
    "full64_raw_absolute_innovation_ema_hl32",
    "full64_raw_both_ema_parts_hl32",
)
LOW32_CASES = ("nested32_raw112", "nested32_i0_tail60")
CASE_NAMES = FULL64_CASES + LOW32_CASES
CASE_BUDGET = {name: (64 if name in FULL64_CASES else 32) for name in CASE_NAMES}


def build_case_views(raw: np.ndarray) -> dict[str, np.ndarray]:
    """Build all nine views from one complete sequence without resetting histories."""
    x, _ = build_view(np.asarray(raw, dtype=np.float32), "raw")
    ema, _ = build_view(x, "raw_net_absinnovation_ema")
    i0_tail = np.concatenate((x[:, :52], x[:, 104:112]), axis=1)
    signed = np.concatenate((x, ema[:, 112:224]), axis=1)
    absolute = np.concatenate((x, ema[:, 224:336]), axis=1)
    return {
        "full64_i0_52": x[:, :52],
        "full64_i0_tail60": i0_tail,
        "full64_books104": x[:, :104],
        "full64_raw112": x,
        "full64_raw_signed_ema_hl32": signed,
        "full64_raw_absolute_innovation_ema_hl32": absolute,
        "full64_raw_both_ema_parts_hl32": ema,
        "nested32_raw112": x,
        "nested32_i0_tail60": i0_tail,
    }
