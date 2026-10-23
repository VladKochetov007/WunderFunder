"""Exact canonical commands for two teachers and two matched student arms per fold."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from common import PACKET, ROOT, read_json

# Canonical argument/provenance objects have heterogeneous JSON fields.
ROLES = ("teacher_a", "teacher_b", "labels", "kd")


def cell_recipe(cell_id: str) -> dict[str, Any]:
    """Select a predeclared fold/seed cell, never create an adaptive replacement."""
    return next(
        cell for cell in read_json(PACKET / "recipe.json")["cells"] if cell["id"] == cell_id
    )


def run_name(cell: str, role: str, profile: bool = False) -> str:
    """Give every model and bounded profile a separate immutable output name."""
    if cell not in ("F0", "F2") or role not in ROLES:
        raise ValueError("unknown frozen cell or role")
    return f"tl3kd_r1_{cell.lower()}_{role}" + ("_profile" if profile else "")


def fit_args(assets: dict[str, Any], cell_id: str, role: str, profile: bool) -> list[str]:
    """Vary only target alpha between students, including identical teacher-cache I/O."""
    recipe = read_json(PACKET / "recipe.json")
    cell = cell_recipe(cell_id)
    teacher = role.startswith("teacher_")
    epochs = recipe["fit"]["teacher_epochs" if teacher else "student_epochs"]
    seed = cell["teachers"][0 if role == "teacher_a" else 1] if teacher else cell["seed"]
    argv = [
        "--name",
        run_name(cell_id, role, profile),
        "--device",
        "cuda",
        "--seed",
        str(seed),
        "--arch",
        "gru",
        "--cell",
        "gru",
        "--hidden",
        "256",
        "--layers",
        "2",
        "--head-layers",
        "1",
        "--dropout",
        "0",
        "--readout-activation",
        "gelu",
        "--enc",
        "tiedlead" if teacher else "linear",
        "--research-features",
        "tl3_lead" if teacher else "none",
        "--train-source",
        "train+valid",
        "--split-manifest",
        assets["splits"][cell_id]["path"],
        "--valid-train-fraction",
        "1",
        "--batch",
        "32",
        "--batch-order",
        "seeded",
        "--chunk",
        "2000",
        "--epochs",
        str(epochs),
        "--loss",
        "wmse",
        "--optim",
        "adamw",
        "--lr",
        "0.001",
        "--wd",
        "0.01",
        "--sched",
        "onecycle",
        "--clip",
        "1",
        "--grad-stats",
        "--precision",
        "fp32",
        "--deterministic",
        "--checkpoint-policy",
        "fixed",
        "--checkpoint-epoch",
        str(epochs - 1),
        "--no-eval-during-fit",
        "--full-valid",
    ]
    if not teacher:
        argv += [
            "--distill",
            f"{run_name(cell_id, 'teacher_a')},{run_name(cell_id, 'teacher_b')}",
            "--distill-alpha",
            "0" if role == "labels" else "0.5",
        ]
    if profile:
        argv += ["--profile-steps", "2", "--profile-warmup", "1"]
    else:
        argv += ["--save-training-state"]
    return argv


def prediction_command(
    assets: dict[str, Any], cell: str, role: str, source: str, scope: str
) -> list[str]:
    """Export feature-only fixed checkpoints on fitting or complete evaluation components."""
    epoch = 3 if role.startswith("teacher_") else 1
    argv = [
        sys.executable,
        str(ROOT / "scripts/predict_run.py"),
        run_name(cell, role),
        source,
        "--epoch",
        str(epoch),
        "--output-name",
        f"{run_name(cell, role)}_{scope}",
        "--campaign-split-manifest",
        assets["splits"][cell]["path"],
        "--device",
        "cuda",
        "--inference-precision",
        "fp32",
    ]
    if scope == "eval":
        argv += ["--campaign-evaluation-only"]
    return argv


def checkpoint_path(assets: dict[str, Any], cell: str, role: str) -> Path:
    """Resolve only this panel's fixed endpoint, not a best-validation checkpoint."""
    epoch = 3 if role.startswith("teacher_") else 1
    return (
        Path(assets["project_root"])
        / "artifacts/runs"
        / run_name(cell, role)
        / f"model_ep{epoch}.pt"
    )
