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
from wnn.training.sequence_complement_monitor import ComplementMonitor
from wnn.training.sequence_distributed import (
    DistributedContext,
    all_ranks_true,
    barrier,
    coordinated_call,
)
from wnn.training.sequence_epoch_fit import publish_epoch_snapshot, reconcile_epoch_snapshots
from wnn.training.sequence_epoch_index import MAX_RETAINED_EPOCHS
from wnn.training.sequence_fit_artifacts import (
    epoch_summary,
    record_epoch_metrics,
    write_final_outputs,
)
from wnn.training.sequence_fit_artifacts import (
    metrics_offset as get_metrics_offset,
)
from wnn.training.sequence_fit_state import training_state as _training_state
from wnn.training.sequence_fit_utils import (
    RankJsonlLog,
    SequenceDataset,
    optimizer_states_are_finite_fp32,
    truncate_with_evidence,
)
from wnn.training.sequence_loop import train_epoch
from wnn.training.sequence_optimizer_telemetry import optimizer_learning_rates


def _load_resume(
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
    *,
    monitor: ComplementMonitor | None = None,
) -> None:
    """Resume or complete a fit, retaining inference snapshots only when requested."""
    history: list[dict[str, object]] = []
    global_step, start_epoch = 0, 0
    retention = str(config.get("checkpoint_retention", "final_only"))
    if retention not in ("final_only", "all_epochs"):
        raise ValueError("unsupported checkpoint retention policy")
    if retention == "all_epochs" and int(config["epochs"]) > MAX_RETAINED_EPOCHS:
        raise ValueError("all-epoch retention supports at most 10,000 configured epochs")
    latest_index: dict[str, object] | None = None
    if resume_path is not None:
        start_epoch, global_step, history = _load_resume(
            resume_path,
            identity,
            base_model,
            optimizer,
            scheduler,
            context,
            rank_log,
            output,
            monitor,
        )
    else:
        rank_log_offsets = gather_rank_log_offsets(rank_log.position(), context)
        if monitor is not None and monitor.progress is not None:
            monitor.progress.phase("checkpoint", 0, 0)
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
                monitor.offsets() if monitor is not None else None,
            ),
            context,
        )
    if not 0 <= start_epoch <= int(config["epochs"]):
        raise ValueError("resume checkpoint next_epoch is outside the configured schedule")
    if retention == "all_epochs":
        latest_index = reconcile_epoch_snapshots(
            output,
            base_model,
            config,
            identity,
            normalization,
            data_provenance,
            profile_receipt,
            profile_hash,
            start_epoch,
            global_step,
            history,
            context,
        )
    for epoch in range(start_epoch, int(config["epochs"])):
        if monitor is not None and monitor.progress is not None:
            monitor.progress.phase("training", global_step, epoch)
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
            progress=monitor.progress if monitor is not None else None,
        )
        if not result.complete:
            raise RuntimeError("full fit ended before the complete epoch boundary")
        if not all_ranks_true(optimizer_states_are_finite_fp32(optimizer), context):
            raise FloatingPointError("optimizer state is not finite strict-FP32 at epoch boundary")
        global_step += result.updates
        summary = epoch_summary(result, config)
        history.append({"epoch": epoch, **summary})
        if monitor is not None:
            coordinated_call(
                lambda current_epoch=epoch, current_step=global_step: monitor.evaluate(
                    base_model, config, context, current_epoch, current_step
                ),
                "report-only full official valid evaluation",
                context,
            )
        epoch_rates = optimizer_learning_rates(optimizer)
        coordinated_call(
            lambda current_summary=summary, current_epoch=epoch, current_step=global_step, current_rate=epoch_rates: (
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
        if monitor is not None and monitor.progress is not None:
            monitor.progress.phase("checkpoint", global_step, epoch + 1)
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
                monitor.offsets() if monitor is not None else None,
            ),
            context,
        )
        if retention == "all_epochs":
            latest_index = publish_epoch_snapshot(
                output,
                base_model,
                config,
                identity,
                normalization,
                data_provenance,
                profile_receipt,
                profile_hash,
                history,
                epoch,
                global_step,
                latest_index,
                context,
                "safe-epoch inference snapshot publication",
            )
    if monitor is not None and monitor.progress is not None:
        monitor.progress.phase("checkpoint", global_step, int(config["epochs"]))
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
                str(latest_index["index_sha256"]) if latest_index is not None else None,
            )
            if context.is_primary
            else True
        ),
        "fixed-final checkpoint write",
        context,
    )
    barrier(context)
    if monitor is not None and monitor.progress is not None:
        monitor.progress.phase("completed", global_step, int(config["epochs"]))
