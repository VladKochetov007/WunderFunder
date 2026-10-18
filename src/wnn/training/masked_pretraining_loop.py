from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from wnn.training.environment import TrainingEnvironment
from wnn.training.masked_pretraining_epoch import UpdateCounts, run_masked_epoch
from wnn.training.masked_pretraining_reporting import (
    profile_summary,
    save_epoch_checkpoint,
)


def fit_masked_groups(
    model: torch.nn.Module,
    loader: DataLoader[torch.Tensor],
    mean: torch.Tensor,
    std: torch.Tensor,
    *,
    environment: TrainingEnvironment,
    device: torch.device,
    seed: int,
    epochs: int,
    chunk_rows: int,
    mask_block_rows: int,
    learning_rate: float,
    weight_decay: float,
    max_grad_norm: float,
    progress_path: Path,
    checkpoint_dir: Path,
    recipe: dict[str, Any],
    profile_updates: int = 0,
    resume_checkpoint: Path | None = None,
) -> dict[str, Any]:
    """Fit a causal masked-input initializer and preserve epoch-boundary resumes."""
    if epochs <= 0 or chunk_rows <= 0 or mask_block_rows <= 0:
        raise ValueError("epochs, chunk rows, and mask block rows must be positive")
    if (profile_updates < 0) or (profile_updates and resume_checkpoint is not None):
        raise ValueError("profile runs cannot resume or use a negative update limit")
    model = model.to(device=device, dtype=torch.float32)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    updates_per_epoch = len(loader) * math.ceil(environment.sequence_length / chunk_rows)
    scheduler = (
        None
        if profile_updates
        else OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            total_steps=epochs * updates_per_epoch,
            pct_start=0.05,
            anneal_strategy="cos",
            div_factor=10.0,
            final_div_factor=100.0,
        )
    )
    counts = UpdateCounts()
    first_epoch = 0
    if resume_checkpoint is not None:
        saved = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
        if saved.get("recipe") != recipe:
            raise ValueError("resume checkpoint recipe differs from the frozen run")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        if scheduler is None or saved.get("scheduler") is None:
            raise ValueError("resume checkpoint does not contain the fixed scheduler")
        scheduler.load_state_dict(saved["scheduler"])
        first_epoch = int(saved["next_epoch"])
        counts.completed = int(saved["completed_updates"])
        counts.attempted = int(saved.get("attempted_updates", counts.completed))
        torch.set_rng_state(saved["torch_rng_state"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng_state"])
    mean_device = mean.to(device=device, dtype=torch.float32)
    std_device = std.to(device=device, dtype=torch.float32)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    mask_generator = torch.Generator(device=device)
    progress_mode = "w" if first_epoch == 0 else "a"
    with progress_path.open(progress_mode, encoding="utf-8") as progress:
        for epoch in range(first_epoch, epochs):
            if loader.generator is not None:
                loader.generator.manual_seed(seed + epoch)
            mask_generator.manual_seed(seed + 1_000_003 * (epoch + 1))
            epoch_updates, epoch_loss, profile_complete = run_masked_epoch(
                model,
                loader,
                mean_device,
                std_device,
                environment=environment,
                device=device,
                epoch=epoch,
                chunk_rows=chunk_rows,
                mask_block_rows=mask_block_rows,
                max_grad_norm=max_grad_norm,
                optimizer=optimizer,
                scheduler=scheduler,
                mask_generator=mask_generator,
                counts=counts,
                progress=progress,
                profile_updates=profile_updates,
                started=started,
            )
            if profile_complete:
                return profile_summary(
                    device,
                    counts.attempted,
                    counts.completed,
                    counts.this_call,
                    epoch_loss,
                    started,
                )
            progress.write(
                json.dumps(
                    {
                        "event": "epoch_complete",
                        "epoch": epoch,
                        "updates": epoch_updates,
                        "mean_masked_loss": epoch_loss,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            progress.flush()
            save_epoch_checkpoint(
                checkpoint_dir / f"epoch_{epoch + 1:02d}.pt",
                model,
                optimizer,
                scheduler,
                recipe,
                epoch + 1,
                counts.completed,
                counts.attempted,
                device,
            )
    return profile_summary(
        device,
        counts.attempted,
        counts.completed,
        counts.this_call,
        None,
        started,
        complete=True,
    )
