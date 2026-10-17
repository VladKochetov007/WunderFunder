from __future__ import annotations

import argparse
from pathlib import Path

from wnn.numeric_ple import NumericFeatureMap


def add_fit_arguments(ap: argparse.ArgumentParser) -> None:
    """Register fit options in their historical order."""
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument(
        "--numeric-map",
        choices=("none", *NumericFeatureMap.MODES),
        default="none",
        help="E540 matched raw-linear or feature-wise PLE map before the GRU projection",
    )
    ap.add_argument(
        "--ple-intervals",
        type=int,
        default=8,
        help="E540 PLE interval count; both arms allocate K+1 slots per raw feature",
    )
    ap.add_argument(
        "--fit-subset-manifest",
        type=Path,
        default=None,
        help="frozen component-complete fitting subset bound to --split-manifest",
    )
    ap.add_argument(
        "--fit-subset-experiment-id",
        default=None,
        help="required experiment identity for non-E540 component subsets",
    )
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument(
        "--stop-after-epoch",
        type=int,
        default=None,
        help="end training after this epoch while the lr schedule still spans --epochs (cheap screens)",
    )
    ap.add_argument("--checkpoint-policy", choices=["best", "fixed"], default="best")
    ap.add_argument(
        "--checkpoint-epoch", type=int, default=None, help="zero-based fixed checkpoint epoch"
    )
    ap.add_argument(
        "--no-eval-during-fit",
        action="store_true",
        help="never construct or read an evaluation loader",
    )
    ap.add_argument(
        "--split-manifest",
        type=Path,
        default=None,
        help="frozen component-excluded outer-fold manifest",
    )
    ap.add_argument(
        "--student-frontier-contract",
        type=Path,
        help="explicit consumer recipe for the CPU-student architecture frontier",
    )
    ap.add_argument(
        "--student-frontier-stage",
        default="",
        help="named seed/update/checkpoint stage inside the consumer recipe",
    )
    ap.add_argument(
        "--student-frontier-preflight-only",
        action="store_true",
        help="validate recipe, cache payloads, source order and normalization without fitting",
    )
    ap.add_argument(
        "--save-training-state",
        action="store_true",
        help="atomically save full training_state.pt at each epoch boundary; fixed final checkpoint only",
    )
    ap.add_argument(
        "--resume-state",
        type=Path,
        default=None,
        help="resume full epoch training state with the identical recipe and schedule horizon",
    )


def add_membership_arguments(ap: argparse.ArgumentParser) -> None:
    """Register membership options in their historical order."""
    ap.add_argument(
        "--class-balance",
        action="store_true",
        help="sample train sequences so market-class shares match validation",
    )
    ap.add_argument(
        "--train-source",
        choices=["train", "valid_devA", "train+valid", "valid"],
        default="train",
        help="train+valid: all train sequences plus validation devA and devB (holdout never)",
    )
    ap.add_argument(
        "--include-heldout-valid",
        action="store_true",
        help="final fit only: include the reserved validation block in training",
    )
    ap.add_argument(
        "--valid-train-fraction",
        type=float,
        default=1.0,
        help="fixed chain subset of eligible validation training sequences for a data-volume comparison",
    )
    ap.add_argument(
        "--epoch-samples",
        type=int,
        default=0,
        help="draw this many training sequences with replacement each epoch to match optimizer steps",
    )
    ap.add_argument("--eval-source", choices=["dev", "devB", "holdout", "vfold"], default="dev")
    ap.add_argument(
        "--valid-fold",
        type=int,
        default=-1,
        help="K-fold over ALL validation sequences: train on train + other folds, evaluate on this fold",
    )
    ap.add_argument("--n-valid-folds", type=int, default=4)
    ap.add_argument(
        "--valid-repeat",
        type=int,
        default=1,
        help="with train+valid: each validation training sequence appears this many times per epoch",
    )
    ap.add_argument(
        "--init",
        default="",
        help="run name whose model.pt initializes weights; NAME@epN loads model_epN.pt",
    )
    ap.add_argument(
        "--final-model-adaptation",
        action="store_true",
        help="explicit all-data initializer adaptation; no held-out claim",
    )
    ap.add_argument(
        "--final-distill-initializer",
        action="store_true",
        help="replace validated cached main targets with causal predictions from the fixed FP32 initializer",
    )
    ap.add_argument(
        "--inherit-init-preprocessing",
        action="store_true",
        help="inherit parent feature construction and architecture; reject conflicting explicit flags",
    )
    ap.add_argument(
        "--freeze",
        default="",
        help="comma-separated parameter-name prefixes kept fixed, e.g. inp,gru (inp,tcn for --arch tcn)",
    )
