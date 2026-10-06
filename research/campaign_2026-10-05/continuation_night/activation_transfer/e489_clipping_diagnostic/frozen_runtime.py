"""Resolve E489 dependencies exclusively from the immutable continuation snapshot."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NIGHT = HERE.parents[1]
FROZEN = NIGHT / "baseline_snapshot"
E489 = FROZEN / "research/campaign_2026-10-05/fp32_activation_duration"
sys.path[:0] = [
    str(E489),
    str(FROZEN / "research/campaign_2026-10-05/activation_optimizer"),
    str(FROZEN / "research/campaign_2026-10-05/activation_placement"),
    str(FROZEN / "research/campaign_2026-10-05/muon_orthogonalization_precision"),
    str(FROZEN / "src"),
]

import fit_activation_optimizer as e470  # noqa: E402
import fit_activation_placement as e474  # noqa: E402
import fit_fp32_duration as actor  # noqa: E402
import precision_replay as e487  # noqa: E402

from wnn.nn import RNNModel  # noqa: E402
from wnn.resources import configure_torch  # noqa: E402

__all__ = (
    "E489",
    "FROZEN",
    "actor",
    "configure_torch",
    "e470",
    "e474",
    "e487",
    "RNNModel",
)
