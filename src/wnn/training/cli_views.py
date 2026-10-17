from __future__ import annotations

import argparse

from wnn.sentinel import INDICATOR_MODES, VALUE_MODES
from wnn.training_qat import QAT_MODES


def add_views_arguments(ap: argparse.ArgumentParser) -> None:
    """Register views options in their historical order."""
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument(
        "--update-rows",
        type=int,
        default=0,
        help="opt-in distilled macrostep rows; 0 preserves one update per --chunk",
    )
    ap.add_argument(
        "--detach-rows",
        type=int,
        default=0,
        help="opt-in gradient detach interval; 0 uses --chunk, with state values carried",
    )
    ap.add_argument(
        "--detach-boundary-jitter",
        action="store_true",
        help="randomize detach phase inside unchanged macro updates; requires chunk < detach rows",
    )
    ap.add_argument("--detach-jitter-seed", type=int, default=271828)
    ap.add_argument(
        "--origin-view",
        choices=["off", "prefix-fixed", "prefix-random", "prefix-discard"],
        default="off",
        help="replay causal prefixes before updates; discard matches fixed replay work while retaining carried state",
    )
    ap.add_argument("--origin-seed", type=int, default=79079)
    ap.add_argument(
        "--credit-mode",
        choices=["deterministic", "random-uncompensated", "artbp"],
        default="deterministic",
        help="random recurrent credit within fixed update windows only",
    )
    ap.add_argument("--credit-keep-probability", type=float, default=0.8)
    ap.add_argument("--credit-seed", type=int, default=100100)
    ap.add_argument(
        "--qat",
        choices=QAT_MODES,
        default="none",
        help="versioned per-row q4/q8 weight-only QAT; activations/state remain fp32",
    )
    ap.add_argument(
        "--activation-backend",
        choices=("none", "scan_fp32", "scan_a8"),
        default="none",
        help="experimental compiled scan; scan_a8 adds recurrent-boundary activation quantization",
    )
    ap.add_argument(
        "--activation-profile-first-chunk",
        action="store_true",
        help="profile only the first causal chunk; requires an activation backend and profile mode",
    )
    ap.add_argument(
        "--qat-learn-scales",
        action="store_true",
        help="learn row steps with the versioned LSQ-inspired rule; requires a Q4 QAT preset",
    )
    ap.add_argument(
        "--recurrent-rank",
        type=int,
        default=0,
        help="SVD-initialized rank per stacked GRU matrix; 0 preserves dense recurrence",
    )
    ap.add_argument("--sentinel-values", choices=VALUE_MODES, default="unchanged")
    ap.add_argument("--sentinel-indicators", choices=INDICATOR_MODES, default="none")
    ap.add_argument(
        "--sentinel-base-width",
        type=int,
        default=0,
        help="inherited only from a frozen sentinel_prepare initializer",
    )
