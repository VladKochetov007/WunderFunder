from __future__ import annotations

import argparse
from pathlib import Path

from wnn.propensity import LEGACY_PROFILE, STRICT_FP32_PROFILE


def build_propensity_parser() -> argparse.ArgumentParser:
    """Build the legacy-compatible and opt-in strict propensity arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-policy", choices=["best", "fixed"], default="best")
    parser.add_argument("--checkpoint-epoch", type=int, default=None)
    parser.add_argument("--no-eval-during-fit", action="store_true")
    parser.add_argument("--split-manifest", type=Path, default=None)
    parser.add_argument(
        "--precision-profile",
        choices=[LEGACY_PROFILE, STRICT_FP32_PROFILE],
        default=LEGACY_PROFILE,
        help="legacy autocast by default; strict_fp32_v1 is an opt-in clean-fold recipe",
    )
    parser.add_argument(
        "--loader-workers",
        type=int,
        default=None,
        help="data-loader workers; profile defaults are legacy=2 and strict_fp32_v1=0",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume strict_fp32_v1 from the latest completed-epoch state in this run directory",
    )
    parser.add_argument(
        "--heldout-fold",
        type=int,
        default=-1,
        help="if >= 0: train on all validation sequences outside this chain fold (cache/chain_folds.npz), "
        "evaluate on it; default: train devA, evaluate devB",
    )
    parser.add_argument(
        "--dev-all",
        action="store_true",
        help="train on all dev sequences (devA + devB), evaluate on the holdout",
    )
    return parser
