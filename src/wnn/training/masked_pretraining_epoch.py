from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import TextIO

import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from wnn.data import N_FEATURES
from wnn.training.environment import TrainingEnvironment
from wnn.training.masked_group_pretraining import masked_model_input, sample_group_mask


@dataclass(slots=True)
class UpdateCounts:
    """Track attempted, successful, and current-call optimizer updates."""

    attempted: int = 0
    completed: int = 0
    this_call: int = 0


def run_masked_epoch(
    model: torch.nn.Module,
    loader: DataLoader[torch.Tensor],
    mean: torch.Tensor,
    std: torch.Tensor,
    *,
    environment: TrainingEnvironment,
    device: torch.device,
    epoch: int,
    chunk_rows: int,
    mask_block_rows: int,
    max_grad_norm: float,
    optimizer: Optimizer,
    scheduler: OneCycleLR | None,
    mask_generator: torch.Generator,
    counts: UpdateCounts,
    progress: TextIO,
    profile_updates: int,
    started: float,
) -> tuple[int, float, bool]:
    """Run one full data pass and return update count, mean loss, and profile stop."""
    model.train()
    epoch_updates = 0
    epoch_loss = 0.0
    for raw in loader:
        raw = raw.to(device=device, dtype=torch.float32, non_blocking=True)
        if raw.ndim != 3 or raw.shape[-1] != N_FEATURES:
            raise ValueError(f"expected batch x time x {N_FEATURES} raw features")
        mask = sample_group_mask(
            raw.shape[0],
            raw.shape[1],
            mask_block_rows,
            device=device,
            generator=mask_generator,
        )
        model_input = masked_model_input(raw, mask, mean, environment)
        target = (raw - mean[:N_FEATURES]) / std[:N_FEATURES]
        state = None
        for start_row in range(0, raw.shape[1], chunk_rows):
            end_row = min(start_row + chunk_rows, raw.shape[1])
            prediction, state = model(model_input[:, start_row:end_row], state)
            if not isinstance(state, torch.Tensor):
                raise TypeError("masked-group pretraining requires a GRU tensor state")
            state = state.detach()
            local_mask = mask[:, start_row:end_row]
            masked_values = int(local_mask.sum().item())
            if masked_values == 0:
                raise RuntimeError("sampled mask contains no reconstruction targets")
            values = F.smooth_l1_loss(prediction, target[:, start_row:end_row], reduction="none")
            per_row_loss = (values * local_mask).sum(dim=-1) / local_mask.sum(dim=-1)
            loss = per_row_loss.mean()
            counts.attempted += 1
            learning_rate = float(optimizer.param_groups[0]["lr"])
            if not torch.isfinite(loss):
                _record_failure(
                    progress,
                    epoch,
                    counts,
                    learning_rate,
                    "nonfinite_loss",
                    started,
                )
                raise FloatingPointError("masked-input reconstruction loss is nonfinite")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            if not torch.isfinite(gradient_norm):
                _record_failure(
                    progress,
                    epoch,
                    counts,
                    learning_rate,
                    "nonfinite_preclip_gradient_norm",
                    started,
                )
                raise FloatingPointError("masked-input gradient norm is nonfinite")
            preclip_norm = float(gradient_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            counts.completed += 1
            counts.this_call += 1
            epoch_updates += 1
            loss_value = float(loss.detach())
            epoch_loss += loss_value
            _record_success(
                progress,
                epoch,
                counts,
                loss_value,
                learning_rate,
                preclip_norm,
                preclip_norm > max_grad_norm,
                masked_values,
                started,
            )
            if profile_updates and counts.this_call >= profile_updates:
                return epoch_updates, epoch_loss / epoch_updates, True
    return epoch_updates, epoch_loss / max(epoch_updates, 1), False


def _record_failure(
    progress: TextIO,
    epoch: int,
    counts: UpdateCounts,
    learning_rate: float,
    reason: str,
    started: float,
) -> None:
    """Write and flush an update attempt that failed before the optimizer step."""
    progress.write(
        json.dumps(
            {
                "event": "update_failed",
                "epoch": epoch,
                "attempted_update": counts.attempted,
                "completed_updates": counts.completed,
                "learning_rate": learning_rate,
                "reason": reason,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            },
            sort_keys=True,
        )
        + "\n"
    )
    progress.flush()


def _record_success(
    progress: TextIO,
    epoch: int,
    counts: UpdateCounts,
    loss: float,
    learning_rate: float,
    gradient_norm: float,
    gradient_clipped: bool,
    masked_values: int,
    started: float,
) -> None:
    """Persist one successful optimizer update and its diagnostic values."""
    progress.write(
        json.dumps(
            {
                "event": "update",
                "epoch": epoch,
                "attempted_update": counts.attempted,
                "update": counts.completed,
                "loss": loss,
                "learning_rate": learning_rate,
                "preclip_gradient_norm": gradient_norm,
                "gradient_clipped": gradient_clipped,
                "optimizer_step_succeeded": True,
                "masked_values": masked_values,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            },
            sort_keys=True,
        )
        + "\n"
    )
    progress.flush()
