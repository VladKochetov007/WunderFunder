from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import torch

from wnn.research_reporting import RunRecorder
from wnn.training.sequence_checkpoint import (
    gather_rank_log_offsets,
    load_training_state,
    restore_rng_state,
    save_training_state,
)
from wnn.training.sequence_distributed import (
    DistributedContext,
    all_ranks_true,
    barrier,
    coordinated_call,
)
from wnn.training.sequence_fit_artifacts import (
    epoch_summary,
    record_epoch_metrics,
    write_final_outputs,
)
from wnn.training.sequence_fit_artifacts import (
    metrics_offset as get_metrics_offset,
)
from wnn.training.sequence_fit_utils import (
    RankJsonlLog,
    SequenceDataset,
    optimizer_states_are_finite_fp32,
    truncate_with_evidence,
)
from wnn.training.sequence_loop import train_epoch


def _load_resume(
    state_path: Path,
    identity: dict[str, object],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    context: DistributedContext,
    rank_log: RankJsonlLog,
    output: Path,
) -> tuple[int, int, list[dict[str, object]]]:
    """Restore model, optimizer, schedule, safe epoch, and this rank's RNG state."""
    state = load_training_state(state_path, identity, context)
    next_epoch, global_step = int(state["next_epoch"]), int(state["global_step"])
    history = state.get("history", [])
    rank_states = cast(list[dict[str, object]], state["rank_rng_states"])
    offsets = cast(list[int], state["rank_log_offsets"])
    metrics_offset = int(state["metrics_offset"])

    def restore() -> bool:
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
    return next_epoch, global_step, history


def _training_state(
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
) -> dict[str, object]:
    """Create resumable state only at a whole-epoch optimizer boundary."""
    return {
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


def run_full_fit(
    dataset: SequenceDataset,
    model: torch.nn.Module,
    base_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    config: dict[str, Any],
    context: DistributedContext,
    rank_log: RankJsonlLog,
    recorder: RunRecorder | None,
    identity: dict[str, object],
    normalization: dict[str, list[float]],
    data_provenance: Mapping[str, object],
    output: Path,
    resume_path: Path | None,
    profile_receipt: Path,
    profile_hash: str,
) -> None:
    """Resume or complete a fixed-final fit, saving only safe epoch checkpoints."""
    history: list[dict[str, object]] = []
    global_step, start_epoch = 0, 0
    if resume_path is not None:
        start_epoch, global_step, history = _load_resume(
            resume_path, identity, base_model, optimizer, scheduler, context, rank_log, output
        )
    else:
        rank_log_offsets = gather_rank_log_offsets(rank_log.position(), context)
        save_training_state(
            output / "training_state.pt",
            _training_state(
                identity,
                context.world_size,
                0,
                0,
                base_model,
                optimizer,
                scheduler,
                history,
                rank_log_offsets,
                0,
            ),
            context,
        )
    if not 0 <= start_epoch <= int(config["epochs"]):
        raise ValueError("resume checkpoint next_epoch is outside the configured schedule")
    for epoch in range(start_epoch, int(config["epochs"])):
        torch.cuda.reset_peak_memory_stats(context.device)
        result = train_epoch(
            dataset,
            model,
            optimizer,
            scheduler,
            config,
            context,
            rank_log,
            epoch,
            global_step,
        )
        if not result.complete:
            raise RuntimeError("full fit ended before the complete epoch boundary")
        if not all_ranks_true(optimizer_states_are_finite_fp32(optimizer), context):
            raise FloatingPointError("optimizer state is not finite strict-FP32 at epoch boundary")
        global_step += result.updates
        summary = epoch_summary(result, config)
        history.append({"epoch": epoch, **summary})
        coordinated_call(
            lambda current_summary=summary, current_epoch=epoch, current_step=global_step, current_rate=float(optimizer.param_groups[0]["lr"]): (
                record_epoch_metrics(
                    recorder,
                    current_summary,
                    current_epoch,
                    current_step,
                    current_rate,
                )
                if context.is_primary
                else True
            ),
            "epoch metrics record",
            context,
        )
        rank_log_offsets = gather_rank_log_offsets(rank_log.position(), context)
        metrics_offset = coordinated_call(
            lambda: get_metrics_offset(output, context), "aggregate metrics offset", context
        )
        save_training_state(
            output / "training_state.pt",
            _training_state(
                identity,
                context.world_size,
                epoch + 1,
                global_step,
                base_model,
                optimizer,
                scheduler,
                history,
                rank_log_offsets,
                metrics_offset,
            ),
            context,
        )
    coordinated_call(
        lambda: (
            write_final_outputs(
                output,
                base_model,
                config,
                history,
                identity,
                normalization,
                data_provenance,
                profile_receipt,
                profile_hash,
                global_step,
            )
            if context.is_primary
            else True
        ),
        "fixed-final checkpoint write",
        context,
    )
    barrier(context)
