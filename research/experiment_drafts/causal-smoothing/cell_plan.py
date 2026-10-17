from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from cell_config import (
    ARMS,
    PACKET,
    RECIPE,
    TRAIN_SCRIPT,
    cell_key,
    data_root,
    inspect_manifest,
    manifest_path,
    recipe_config,
    run_name,
    source_identity,
    trainer_command,
)
from cell_support import shell


def paths(root: Path, phase: str, arm: str, fold: int, seed: int, attempt: int) -> tuple[Path, str]:
    """Return immutable receipt directory and identity for the selected phase."""
    artifact_root = root / "artifacts/causal-smoothing-runner"
    if phase == "profile":
        key = f"profile-{arm}-a{attempt}"
        return artifact_root / "profiles" / key, key
    key = cell_key(arm, fold, seed, attempt)
    return artifact_root / "cells" / key, key


def command_for(
    args: argparse.Namespace, root: Path, split: dict[str, Any], training: dict[str, Any]
) -> list[str]:
    """Build the exact CLI for a planned or executed stage."""
    name = run_name(args.arm, args.fold, args.seed, args.attempt, profile=args.phase == "profile")
    if args.phase in ("profile", "fit"):
        return trainer_command(
            args.arm,
            args.fold,
            args.seed,
            split,
            training,
            args.attempt,
            profile=args.phase == "profile",
        )
    if args.phase == "predict":
        return [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--predict",
            name,
            "valid",
            "--campaign-split-manifest",
            split["path"],
            "--campaign-evaluation-only",
            "--inference-precision",
            "fp32",
            "--device",
            "cuda",
            "--batch",
            str(training["batch"]),
            "--chunk",
            str(training["chunk"]),
            "--epoch",
            str(training["checkpoint_epoch"]),
        ]
    return [
        sys.executable,
        str(PACKET / "grade_cell.py"),
        "--root",
        str(root),
        "--run",
        name,
        "--fold",
        str(args.fold),
        "--recipe",
        str(RECIPE),
        "--output",
        str(
            paths(root, "score", args.arm, args.fold, args.seed, args.attempt)[0]
            / "grade_result.json"
        ),
    ]


def current_bindings(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    """Resolve frozen recipe, source tree, split, and dataset identity for this cell."""
    recipe, recipe_sha = recipe_config()
    split = inspect_manifest(manifest_path(root, args.fold), args.fold)
    source = source_identity()
    dataset = {"sources": split["sources"], "cache_identities": split["cache_identities"]}
    dataset_sha = hashlib.sha256(
        json.dumps(dataset, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "recipe": recipe,
        "recipe_sha256": recipe_sha,
        "split": split,
        "source_identity": source,
        "dataset_identity": dataset,
        "dataset_identity_sha256": dataset_sha,
    }


def plan(args: argparse.Namespace) -> None:
    """Print commands and bindings without environment probes or model execution."""
    root = data_root()
    binding = current_bindings(root, args)
    command = command_for(args, root, binding["split"], binding["recipe"]["training"])
    result = {
        "status": "plan_only",
        "execute_required": True,
        "phase": args.phase,
        "arm": args.arm,
        "fold": args.fold,
        "seed": args.seed,
        "attempt": args.attempt,
        "profile_attempt": args.profile_attempt,
        "run_name": run_name(
            args.arm, args.fold, args.seed, args.attempt, profile=args.phase == "profile"
        ),
        "command": command,
        "command_shell": shell(command),
        "recipe_sha256": binding["recipe_sha256"],
        "source_identity": binding["source_identity"],
        "dataset_identity": binding["dataset_identity"],
        "split": binding["split"],
    }
    if args.phase == "fit":
        result["required_profiles"] = [f"profile-{arm}-a{args.profile_attempt}" for arm in ARMS]
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
