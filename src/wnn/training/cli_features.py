from __future__ import annotations

import argparse
from pathlib import Path

from wnn.privileged import PRIVILEGED_MODES
from wnn.research_features import RESEARCH_FEATURE_MODES


def add_weights_arguments(ap: argparse.ArgumentParser) -> None:
    """Register weights options in their historical order."""
    ap.add_argument(
        "--ema-teacher",
        type=float,
        default=0.0,
        help="mean-teacher consistency: weight of (out - EMA-model out)^2 on the two target outputs",
    )
    ap.add_argument(
        "--ema-metric-weight",
        action="store_true",
        help="weight teacher consistency by |clipped target|, matching the scored loss",
    )
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument(
        "--ema-reference-rows",
        type=int,
        default=0,
        help="0 keeps per-attempt decay; otherwise decay**(update_rows/reference_rows)",
    )
    ap.add_argument(
        "--eval-ema",
        action="store_true",
        help="evaluate and save the EMA weights instead of the raw ones",
    )
    ap.add_argument(
        "--mask-weight",
        action="store_true",
        help="main loss only on is_scored rows (needs a validation train source)",
    )
    ap.add_argument(
        "--propensity",
        default="",
        help="run name of a frozen PropensityNet; weights main loss by p(x)",
    )
    ap.add_argument("--prop-power", type=float, default=1.0, help="weights p(x)**power")
    ap.add_argument(
        "--prop-eps",
        type=float,
        default=0.0,
        help="floor added to propensity weights (mask robustness)",
    )
    ap.add_argument(
        "--class-weight",
        default="",
        help="per-market-class multipliers on the main loss weights, e.g. 1.19,1.32,2.02,0.79",
    )
    ap.add_argument(
        "--valid-true-mask",
        type=float,
        default=-1.0,
        help="if >= 0: rows of sequences that carry a true scoring mask (validation) are weighted c*(is_scored + this eps) instead of by the propensity net; c matches the mean propensity weight",
    )
    ap.add_argument(
        "--prop-input",
        default="",
        help="run name of a frozen PropensityNet appended as an input feature",
    )
    ap.add_argument(
        "--swa-frac",
        type=float,
        default=0.0,
        help="average weights over the last fraction of optimizer steps; evaluate the average",
    )


def add_features_arguments(ap: argparse.ArgumentParser) -> None:
    """Register features options in their historical order."""
    ap.add_argument(
        "--train-classes",
        default="",
        help="comma-separated market classes to train on (oracle expert test)",
    )
    ap.add_argument("--online-norm", action="store_true")
    ap.add_argument(
        "--heuristics", action="store_true", help="append the causal heuristic library as inputs"
    )
    ap.add_argument(
        "--raw-only",
        action="store_true",
        help="zero the event-feature inputs (raw 112-column view)",
    )
    ap.add_argument(
        "--research-features",
        choices=RESEARCH_FEATURE_MODES,
        default="none",
        help="append causal volume, moment, or raw-field age features before TBPTT",
    )
    ap.add_argument(
        "--research-feature-control",
        choices=("none", "newest"),
        default="none",
        help="newest zeros the newest moment-feature block while keeping input width fixed",
    )
    ap.add_argument(
        "--representation-mode",
        choices=("base", "ema", "sparse", "ema_sparse"),
        default=None,
        help="append the fixed 256-wide fit-only sparse/causal-history representation",
    )
    ap.add_argument(
        "--representation-cache",
        type=Path,
        default=None,
        help="load or create the source-qualified fitted representation artifact shared by arms",
    )
    ap.add_argument(
        "--privileged-mode",
        choices=PRIVILEGED_MODES,
        default="none",
        help="teacher-only equal-width current/future raw inputs; requires inner component crossfit",
    )
    ap.add_argument("--privileged-horizon", type=int, default=8)
    ap.add_argument(
        "--fit-groups-manifest",
        type=Path,
        help="strict inner component split within --split-manifest, for fresh teacher fitting",
    )
    ap.add_argument("--seed", type=int, default=0)
