from __future__ import annotations

import torch

from wnn.training.sequence_checkpoint_policy import RESUME_STATE_PURPOSE
from wnn.training.sequence_weight_ema import SequenceWeightEMA


def training_state(
    identity: dict[str, object],
    world_size: int,
    next_epoch: int,
    global_step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    history: list[dict[str, object]],
    rank_log_offsets: list[int],
    metrics_offset: int,
    monitor_offsets: dict[str, int] | None = None,
    weight_ema: SequenceWeightEMA | None = None,
) -> dict[str, object]:
    """Serialize the same safe epoch state, optionally including report-only log offsets."""
    state = {
        "checkpoint_schema": "wnn.sequence-resume.v1",
        "checkpoint_purpose": RESUME_STATE_PURPOSE,
        "contract": identity,
        "world_size": world_size,
        "next_epoch": next_epoch,
        "global_step": global_step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "history": history,
        "rank_log_offsets": rank_log_offsets,
        "metrics_offset": metrics_offset,
    }
    if monitor_offsets is not None:
        state["monitor_offsets"] = monitor_offsets
    if weight_ema is not None:
        state["weight_ema"] = weight_ema.state_dict()
    return state
