from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, TextIO

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import OneCycleLR


def record_failed_update(
    progress: TextIO,
    *,
    epoch: int,
    attempted_update: int,
    completed_updates: int,
    learning_rate: float,
    reason: str,
    started: float,
) -> None:
    """Persist an optimizer attempt that failed before a successful parameter step."""
    progress.write(
        json.dumps(
            {
                "event": "update_failed",
                "epoch": epoch,
                "attempted_update": attempted_update,
                "completed_updates": completed_updates,
                "learning_rate": learning_rate,
                "reason": reason,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            },
            sort_keys=True,
        )
        + "\n"
    )
    progress.flush()


def save_epoch_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: Optimizer,
    scheduler: OneCycleLR | None,
    recipe: dict[str, Any],
    next_epoch: int,
    completed_updates: int,
    attempted_updates: int,
    device: torch.device,
) -> None:
    """Atomically write model, optimizer, schedule and RNG state at an epoch boundary."""
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "recipe": recipe,
        "next_epoch": next_epoch,
        "completed_updates": completed_updates,
        "attempted_updates": attempted_updates,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
    }
    temporary = path.with_suffix(f".tmp-{os.getpid()}")
    torch.save(state, temporary)
    temporary.replace(path)


def profile_summary(
    device: torch.device,
    attempted_updates: int,
    completed_updates: int,
    updates_this_call: int,
    mean_loss: float | None,
    started: float,
    *,
    complete: bool = False,
) -> dict[str, Any]:
    """Return the bounded profile or completed-fit resource measurements."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        allocated = torch.cuda.max_memory_allocated(device)
        reserved = torch.cuda.max_memory_reserved(device)
    else:
        allocated = reserved = 0
    return {
        "complete": complete,
        "attempted_updates": attempted_updates,
        "completed_updates": completed_updates,
        "updates_this_call": updates_this_call,
        "mean_masked_loss": mean_loss,
        "wall_seconds": round(time.perf_counter() - started, 3),
        "cuda_peak_allocated_bytes": allocated,
        "cuda_peak_reserved_bytes": reserved,
    }
