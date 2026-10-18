"""Build the frozen labels-only R4 trainer arguments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from adapter_common import _NAME


def build_training_argv(
    recipe: dict[str, Any],
    *,
    arm: str,
    fold: int,
    seed: int,
    run_name: str,
    split_manifest: Path,
    mode: str,
    resume_state: Path | None = None,
) -> list[str]:
    """Build the frozen R4 argv without copying or reimplementing training."""
    arms = {item["name"]: item for item in recipe["arms"]}
    if (
        arm not in arms
        or not 0 <= fold < 4
        or seed < 0
        or not _NAME.fullmatch(run_name)
    ):
        raise ValueError("invalid arm, fold, seed or run name")
    if seed in recipe.get("excluded_external_seeds", [1001, 1002]):
        raise ValueError(f"seed {seed} is excluded by existing external ownership")
    if mode not in {"profile", "fit", "resume"}:
        raise ValueError("mode must be profile, fit or resume")
    shape = arms[arm]
    hyper = recipe["hyperparameters"]
    endpoint = recipe["primary_endpoint"]
    if (
        endpoint["selection"] != "fixed"
        or endpoint["outer_validation_during_fit"] is not False
        or hyper["precision"] != "fp32"
        or recipe.get("teacher") is not None
        or recipe.get("propensity") is not None
    ):
        raise ValueError(
            "command builder no longer describes the frozen labels-only endpoint"
        )
    args = [
        "--name",
        run_name,
        "--hidden",
        str(shape["hidden"]),
        "--layers",
        str(shape["layers"]),
        "--valid-fold",
        str(fold),
        "--split-manifest",
        str(split_manifest.resolve(strict=True)),
        "--device",
        "cuda",
        "--arch",
        "gru",
        "--cell",
        "gru",
        "--enc",
        "linear",
        "--dropout",
        str(hyper["dropout"]),
        "--train-seqs",
        "10607",
        "--train-source",
        "train+valid",
        "--valid-train-fraction",
        "1",
        "--valid-repeat",
        "1",
        "--n-valid-folds",
        "4",
        "--full-valid",
        "--eval-source",
        "vfold",
        "--batch",
        str(recipe["hyperparameters"]["batch"]),
        "--chunk",
        str(recipe["hyperparameters"]["chunk"]),
        "--epochs",
        str(endpoint["epochs"]),
        "--checkpoint-policy",
        str(endpoint["selection"]),
        "--checkpoint-epoch",
        str(endpoint["checkpoint_zero_based"]),
        "--no-eval-during-fit",
        "--seed",
        str(seed),
        "--precision",
        str(hyper["precision"]),
        "--loss",
        "wmse",
        "--loss-norm",
        "chunk",
        "--batch-order",
        str(hyper["batch_order"]),
        "--optim",
        str(hyper["optim"]),
        "--lr",
        str(hyper["lr"]),
        "--beta2",
        str(hyper["beta2"]),
        "--wd",
        str(hyper["weight_decay"]),
        "--sched",
        str(hyper["scheduler"]),
        "--warmup-frac",
        str(hyper["warmup_fraction"]),
        "--head-lr-mult",
        str(hyper["head_lr_multiplier"]),
        "--clip",
        str(hyper["gradient_clip"]),
    ]
    for key, flag in (
        ("deterministic", "--deterministic"),
        ("sync_free", "--sync-free"),
        ("fused_adam", "--fused-adam"),
        ("grad_stats", "--grad-stats"),
    ):
        if hyper[key]:
            args.append(flag)
    if mode == "profile":
        args.extend(("--profile-steps", "3", "--profile-warmup", "1"))
    else:
        args.append("--save-training-state")
        if mode == "resume":
            if resume_state is None or not resume_state.is_file():
                raise ValueError("resume mode requires an existing training_state.pt")
            args.extend(("--resume-state", str(resume_state.resolve(strict=True))))
    return args
