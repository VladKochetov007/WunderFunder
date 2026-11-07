from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import cast

import torch

from wnn.training.sequence_checkpoint import (
    load_training_state,
    restore_rng_state,
)
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
from wnn.training.sequence_monitor import SequenceMonitor
from wnn.training.sequence_monitor_events import monitor_event_offsets


def load_resume(
    state_path: Path,
    identity: dict[str, object],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    context: DistributedContext,
    rank_log: RankJsonlLog,
    output: Path,
    monitor: SequenceMonitor | None,
    monitoring_enabled: bool,
) -> tuple[int, int, list[dict[str, object]]]:
    """Restore model, optimizer, schedule, safe epoch, and this rank's RNG state."""
    state = load_training_state(state_path, identity, context)
    next_epoch, global_step = int(state["next_epoch"]), int(state["global_step"])
    history = state.get("history", [])
    rank_states = cast(list[dict[str, object]], state["rank_rng_states"])
    offsets = cast(list[int], state["rank_log_offsets"])
    metrics_offset = int(state["metrics_offset"])

    def restore() -> bool:
        """Restore this rank from the already validated safe checkpoint."""
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        restore_rng_state(rank_states[context.rank], context)
        return True

    coordinated_call(restore, "resume optimizer/model state restore", context)
    if not all_ranks_true(optimizer_states_are_finite_fp32(optimizer), context):
        raise FloatingPointError("resume optimizer state is not finite strict-FP32")
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
    if monitoring_enabled:
        monitor_offsets = cast(dict[str, object], state["monitor_offsets"])

        def restore_monitor() -> bool:
            """Restore monitor prefixes and acknowledge the coordinated side effect."""
            if context.is_primary and monitor is not None:
                monitor.restore_offsets(monitor_offsets, global_step)
            return True

        coordinated_call(restore_monitor, "safe-boundary monitor log restore", context)
    return next_epoch, global_step, history


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
) -> dict[str, object]:
    """Create resumable state only at a whole-epoch optimizer boundary."""
    state: dict[str, object] = {
        "checkpoint_schema": "wnn.sequence-resume.v1",
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
    return state


def save_safe_state(
    save_state: Callable[[Path, dict[str, object], DistributedContext], None],
    output: Path,
    identity: dict[str, object],
    context: DistributedContext,
    next_epoch: int,
    global_step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    history: list[dict[str, object]],
    rank_log: RankJsonlLog,
    monitor_offsets: dict[str, int] | None = None,
) -> None:
    """Persist a completed epoch with coordinated log offsets."""
    from wnn.training.sequence_checkpoint import gather_rank_log_offsets
    from wnn.training.sequence_fit_artifacts import metrics_offset as get_metrics_offset

    offsets = gather_rank_log_offsets(rank_log.position(), context)
    if next_epoch:
        metrics_position = coordinated_call(
            lambda: get_metrics_offset(output, context), "aggregate metrics offset", context
        )
    else:
        metrics_position = 0
    state = training_state(
        identity,
        context.world_size,
        next_epoch,
        global_step,
        model,
        optimizer,
        scheduler,
        history,
        offsets,
        metrics_position,
        monitor_offsets,
    )
    save_state(output / "training_state.pt", state, context)


def current_monitor_offsets(
    output: Path, context: DistributedContext, monitoring_enabled: bool
) -> dict[str, int] | None:
    """Read identical shared stream positions on all ranks only for opted-in fits."""
    if not monitoring_enabled:
        return None
    return coordinated_call(
        lambda: monitor_event_offsets(output), "aggregate monitor event offsets", context
    )
