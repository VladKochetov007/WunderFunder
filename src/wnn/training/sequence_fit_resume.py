from __future__ import annotations

from pathlib import Path
from typing import cast

import torch

from wnn.training.sequence_checkpoint import (
    load_training_state,
    restore_rng_state,
)
from wnn.training.sequence_complement_monitor import ComplementMonitor
from wnn.training.sequence_distributed import (
    DistributedContext,
    all_ranks_true,
    coordinated_call,
)
from wnn.training.sequence_fit_utils import (
    RankJsonlLog,
    optimizer_states_are_finite_fp32,
    truncate_with_evidence,
)


def load_resume(
    state_path: Path,
    identity: dict[str, object],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    context: DistributedContext,
    rank_log: RankJsonlLog,
    output: Path,
    monitor: ComplementMonitor | None = None,
) -> tuple[int, int, list[dict[str, object]]]:
    """Restore model, optimizer, schedule, safe epoch, and this rank's RNG state."""
    state = load_training_state(state_path, identity, context)
    next_epoch, global_step = int(state["next_epoch"]), int(state["global_step"])
    history = state.get("history", [])
    rank_states = cast(list[dict[str, object]], state["rank_rng_states"])
    offsets = cast(list[int], state["rank_log_offsets"])
    metrics_offset = int(state["metrics_offset"])
    if monitor is not None:
        monitor_offsets = state.get("monitor_offsets")
        if not isinstance(monitor_offsets, dict):
            raise ValueError("monitored resume is missing safe-boundary report offsets")
        coordinated_call(
            lambda: monitor.restore_offsets(monitor_offsets, global_step),
            "safe-boundary monitor restore",
            context,
        )

    def restore() -> bool:
        """Restore this rank from the already validated safe checkpoint."""
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        restore_rng_state(rank_states[context.rank], context)
        return True

    coordinated_call(restore, "resume optimizer/model state restore", context)
    if not all_ranks_true(optimizer_states_are_finite_fp32(optimizer), context):
        raise FloatingPointError("optimizer state is not finite strict-FP32")
    coordinated_call(
        lambda: (
            rank_log.truncate(offsets[context.rank], rank=context.rank, global_step=global_step)
            or True
        ),
        "safe-boundary rank log restore",
        context,
    )
    coordinated_call(
        lambda: (
            truncate_with_evidence(
                output / "metrics.jsonl",
                metrics_offset,
                rank=0,
                global_step=global_step,
            )
            if context.is_primary
            else True
        ),
        "safe-boundary metric log restore",
        context,
    )
    return next_epoch, global_step, history
