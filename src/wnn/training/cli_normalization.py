from __future__ import annotations

import argparse


def add_normalization_arguments(ap: argparse.ArgumentParser) -> None:
    """Register normalization options in their historical order."""
    ap.add_argument(
        "--loss-norm",
        choices=["chunk", "running"],
        default="chunk",
        help="wmse main loss divided by each chunk's weight sum (chunk) or by a running mean of it (running: rows keep the pooled metric's relative weights across chunks; brainteaser card B)",
    )
    ap.add_argument(
        "--distill-denominator",
        choices=["chunk", "fixed", "shuffled"],
        default="chunk",
        help="cached-distill MAIN loss: update mass, frozen reference mass, or shuffled mass-factor sham",
    )
    ap.add_argument(
        "--distill-reference-mass",
        type=float,
        default=0.0,
        help="positive frozen fit-pool main mass for fixed/shuffled controls",
    )
    ap.add_argument(
        "--calibrate-distill-reference",
        default="",
        help="write a fit-pool mass report and exit before optimizer construction",
    )
    ap.add_argument(
        "--calibration-epochs",
        type=int,
        default=1,
        help="complete training-loader passes used only for reference calibration",
    )
    ap.add_argument("--denominator-shuffle-seed", type=int, default=314159)
    ap.add_argument(
        "--since-weight",
        type=float,
        default=0.0,
        help="if > 0: multiply main-loss row weights by max(floor, 1/(1+since/tau)), since = rows since the last i0 price event (causal input only; ideas_round4 B2)",
    )
    ap.add_argument("--since-floor", type=float, default=0.25)
    ap.add_argument(
        "--episode-weight",
        type=float,
        default=0.0,
        help="scale for training-only inverse-sqrt class-3 target episode length",
    )
    ap.add_argument("--episode-floor", type=float, default=0.5)
    ap.add_argument(
        "--td-alpha",
        type=float,
        default=0.0,
        help="TD targets (ideas_round4 B1b): mix the label with the EMA teacher's output at the last row before the next label change (same |y| weights); needs the EMA model",
    )
    ap.add_argument(
        "--td-ramp", type=float, default=0.5, help="epochs over which td-alpha ramps up from 0"
    )
    ap.add_argument(
        "--gcorr-weight",
        type=float,
        default=1.0,
        help="weight of the pooled-Pearson term in --loss wmse_gcorr",
    )
