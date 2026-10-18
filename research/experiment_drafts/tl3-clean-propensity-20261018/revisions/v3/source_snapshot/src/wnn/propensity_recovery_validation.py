from __future__ import annotations

import stat
from pathlib import Path
from typing import Any

from wnn.splits import file_sha256


def validate_resume_boundary(
    state: dict[str, Any], static_contract: dict[str, Any], run_dir: Path
) -> None:
    """Validate an epoch cursor, including a terminal finalize-only boundary."""
    epochs = int(static_contract["options"]["epochs"])
    next_epoch = state.get("next_epoch")
    if (
        not isinstance(next_epoch, int)
        or isinstance(next_epoch, bool)
        or not 0 <= next_epoch <= epochs
    ):
        raise ValueError("strict propensity state has an invalid epoch cursor")
    if next_epoch < epochs:
        return

    history = state.get("history")
    if not isinstance(history, list) or [entry.get("epoch") for entry in history] != list(
        range(epochs)
    ):
        raise ValueError("terminal strict propensity checkpoint lacks every committed epoch")
    completed_per_epoch: list[int] = []
    for entry in history:
        progress = entry.get("optimizer_progress")
        if not isinstance(progress, dict):
            raise ValueError("terminal strict propensity epoch lacks update accounting")
        completed = progress.get("optimizer_completed_updates")
        if (
            not isinstance(completed, int)
            or completed <= 0
            or progress.get("optimizer_attempts") != completed
            or progress.get("optimizer_step_calls") != completed
            or progress.get("nonfinite_updates") != 0
        ):
            raise ValueError("terminal strict propensity epoch has incomplete update accounting")
        completed_per_epoch.append(completed)
    if len(set(completed_per_epoch)) != 1:
        raise ValueError("terminal strict propensity epoch update counts differ")
    aggregate = state.get("progress")
    expected_total = sum(completed_per_epoch)
    if not isinstance(aggregate, dict) or any(
        aggregate.get(key) != expected_total
        for key in ("optimizer_attempts", "optimizer_step_calls", "optimizer_completed_updates")
    ):
        raise ValueError("terminal strict propensity aggregate update accounting differs")
    if aggregate.get("nonfinite_updates") != 0:
        raise ValueError("terminal strict propensity contains rejected updates")

    provenance = state.get("provenance")
    if (
        not isinstance(provenance, dict)
        or provenance.get("selected_epoch") != static_contract["options"]["checkpoint_epoch"]
    ):
        raise ValueError("terminal strict propensity selected checkpoint is not recorded")
    checkpoint_hashes = provenance.get("checkpoint_sha256")
    if not isinstance(checkpoint_hashes, dict):
        raise ValueError("terminal strict propensity checkpoint hashes are missing")
    expected_files = [*(f"model_ep{epoch}.pt" for epoch in range(epochs)), "model.pt"]
    for name in expected_files:
        path = run_dir / name
        try:
            info = path.lstat()
        except FileNotFoundError as error:
            raise ValueError(f"terminal strict propensity output is missing: {name}") from error
        if not stat.S_ISREG(info.st_mode) or name not in checkpoint_hashes:
            raise ValueError(f"terminal strict propensity output is invalid: {name}")
        if file_sha256(path) != checkpoint_hashes[name]:
            raise ValueError(f"terminal strict propensity output changed: {name}")
