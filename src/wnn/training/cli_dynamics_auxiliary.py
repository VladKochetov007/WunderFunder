from __future__ import annotations

import argparse


def add_dynamics_auxiliary_arguments(ap: argparse.ArgumentParser) -> None:
    """Register the optional future-input dynamics head and its loss weight."""
    ap.add_argument(
        "--dynamics-aux-head",
        action="store_true",
        help="allocate the training-only 52-output future-input dynamics head",
    )
    ap.add_argument(
        "--dynamics-aux-weight",
        type=float,
        default=0.0,
        help="mean Huber weight for the future-input dynamics head (requires --dynamics-aux-head)",
    )
