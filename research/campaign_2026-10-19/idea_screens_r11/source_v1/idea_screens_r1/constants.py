"""Fixed feature groups and horizons for the bounded screens."""

from __future__ import annotations

FEATURE_WIDTH = 112
MAX_ROWS = 20_000
MAX_SEQUENCES = 48
HORIZONS = (1, 8, 32, 128, 512)
GROUPS = {
    "i0_p": slice(0, 22),
    "i0_v": slice(22, 44),
    "i0_dp": slice(44, 48),
    "i1_p": slice(52, 74),
    "i1_v": slice(74, 96),
}
