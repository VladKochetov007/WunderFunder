from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kaggle_h512_f1_worker_support import (
    DERIVED,
    FINALIZER_RESERVE_SECONDS,
    FIT_NAME,
    OUTER_WALL_CAP_SECONDS,
    PROFILE_NAME,
    PROVIDER_SAFETY_SECONDS,
    read_object,
)


def event_capability(argv: list[str], name: str) -> str:
    """Return one explicit CLI value from a frozen canonical argv."""
    try:
        index = argv.index(name)
    except ValueError as error:
        raise ValueError(f"Frozen argv lacks {name}") from error
    if index + 1 >= len(argv):
        raise ValueError(f"Frozen argv lacks value for {name}")
    return argv[index + 1]


def validate_recipe(proposal: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Check exact profile/fit command relationship and frozen RANK01 flags."""
    recipe = proposal["recipe"]
    profile = list(recipe["exact_profile_trainer_argv"])
    fit = list(recipe["exact_conditional_fit_trainer_argv"])
    if not profile or not fit or profile[0] != "scripts/train_rnn.py" or fit[0] != profile[0]:
        raise ValueError("Trainer entrypoint differs from the sealed canonical CLI")
    if (
        event_capability(profile, "--name") != PROFILE_NAME
        or event_capability(fit, "--name") != FIT_NAME
    ):
        raise ValueError("Frozen profile or fit run name differs")
    required = {
        "--hidden": "512",
        "--layers": "1",
        "--valid-fold": "1",
        "--split-manifest": DERIVED.as_posix(),
        "--device": "cuda",
        "--arch": "gru",
        "--cell": "gru",
        "--enc": "linear",
        "--dropout": "0",
        "--batch": "16",
        "--chunk": "2000",
        "--epochs": "6",
        "--checkpoint-policy": "fixed",
        "--checkpoint-epoch": "5",
        "--seed": "1004",
        "--precision": "fp32",
        "--loss": "wmse",
        "--loss-norm": "chunk",
        "--train-seqs": "10607",
        "--lr": "0.0001",
        "--beta2": "0.999",
        "--wd": "0.01",
        "--clip": "1",
        "--train-source": "train+valid",
        "--valid-train-fraction": "1",
        "--valid-repeat": "1",
        "--n-valid-folds": "4",
        "--eval-source": "vfold",
        "--batch-order": "seeded",
        "--optim": "adamw",
        "--sched": "onecycle",
        "--warmup-frac": "0.05",
        "--head-lr-mult": "1",
    }
    required_flags = (
        "--full-valid",
        "--no-eval-during-fit",
        "--deterministic",
        "--sync-free",
        "--fused-adam",
        "--grad-stats",
    )
    for argv in (profile, fit):
        for name, expected in required.items():
            if event_capability(argv, name) != expected:
                raise ValueError(f"Frozen recipe option changed: {name}")
        if any(flag not in argv for flag in required_flags):
            raise ValueError("Frozen recipe is missing a deterministic RANK01 flag")
        if "--raw-only" in argv:
            raise ValueError("Recipe must retain raw112 plus causal event features")
        if any(flag in argv for flag in ("--init", "--resume-state", "--save-training-state")):
            raise ValueError("The cold fit cannot inherit weights or continuation state")
    if event_capability(profile, "--profile-steps") != "3":
        raise ValueError("Profile must contain exactly three measured batches")
    if event_capability(profile, "--profile-warmup") != "1":
        raise ValueError("Profile must contain exactly one warmup batch")
    if "--profile-steps" in fit or "--profile-warmup" in fit:
        raise ValueError("Full fit argv must not include profile flags")
    fit_without_name = fit.copy()
    profile_without_name: list[str] = []
    index = 0
    while index < len(profile):
        if profile[index] in ("--profile-steps", "--profile-warmup"):
            index += 2
        else:
            profile_without_name.append(profile[index])
            index += 1
    profile_without_name[profile_without_name.index(PROFILE_NAME)] = FIT_NAME
    if profile_without_name != fit_without_name:
        raise ValueError("Fit argv differs beyond name and profile-only flags")
    return profile, fit


def validate_admission(path: Path, proposal_sha256: str) -> dict[str, Any]:
    """Require a root-owned fresh quota and conditional-fit authorization receipt."""
    receipt = read_object(path)
    if receipt.get("schema") != "wnn.kaggle-h512-f1-root-admission.v1":
        raise ValueError("Unsupported root admission receipt")
    if receipt.get("proposal_sha256") != proposal_sha256:
        raise ValueError("Root admission is not bound to this proposal")
    if receipt.get("status") != "ROOT_REVIEWED_CONDITIONAL_ADMISSION":
        raise ValueError("Root has not admitted this conditional worker")
    for name in ("profile_launch_admitted", "conditional_fit_admitted", "no_competing_reservation"):
        if receipt.get(name) is not True:
            raise ValueError(f"Root admission gate is false: {name}")
    observed_at = receipt.get("quota_observed_at_utc")
    if not isinstance(observed_at, str):
        raise ValueError("Admission lacks the provider quota observation time")
    try:
        observed = datetime.fromisoformat(observed_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("Admission quota timestamp is invalid") from error
    if observed.tzinfo is None:
        raise ValueError("Admission quota timestamp must include a timezone")
    age = (datetime.now(UTC) - observed.astimezone(UTC)).total_seconds()
    if age < -60 or age > 900:
        raise ValueError("Admission quota observation is not fresh")
    if receipt.get("gpu_reserved_seconds") != 0:
        raise ValueError("Admission requires zero currently reserved provider seconds")
    quota = receipt.get("remaining_unreserved_provider_gpu_seconds")
    safety = receipt.get("safety_reserve_seconds")
    if not isinstance(quota, (int, float)) or not math.isfinite(quota) or quota <= 0:
        raise ValueError("Admission lacks a finite fresh provider quota read")
    if receipt.get("provider_charge_factor") != 1.0:
        raise ValueError("Provider charge factor must use verified 1x session-wall evidence")
    if not isinstance(safety, int) or safety < PROVIDER_SAFETY_SECONDS:
        raise ValueError("Provider safety reserve must be at least 1,200 seconds")
    if receipt.get("outer_wall_cap_seconds") != OUTER_WALL_CAP_SECONDS:
        raise ValueError("Admission must retain the 16,000-second total wall cap")
    if receipt.get("runner_finalizer_reserve_seconds") != FINALIZER_RESERVE_SECONDS:
        raise ValueError("Admission must preserve the 300-second finalizer reserve")
    return receipt
