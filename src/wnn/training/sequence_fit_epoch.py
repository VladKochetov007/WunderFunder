from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from wnn.research_reporting import RunRecorder
from wnn.training.sequence_distributed import (
    DistributedContext,
    all_ranks_true,
    barrier,
    coordinated_call,
)
from wnn.training.sequence_epoch_fit import publish_epoch_snapshot
from wnn.training.sequence_fit_artifacts import (
    epoch_summary,
    record_epoch_metrics,
    write_final_outputs,
)
from wnn.training.sequence_fit_resume import current_monitor_offsets, save_safe_state
from wnn.training.sequence_fit_utils import (
    RankJsonlLog,
    SequenceDataset,
    optimizer_states_are_finite_fp32,
)
from wnn.training.sequence_loop import train_epoch
from wnn.training.sequence_monitor import SequenceMonitor
from wnn.training.sequence_optimizer_telemetry import optimizer_learning_rates


@dataclass
class FitEpochRuntime:
    """Dependencies and immutable provenance for each full-fit epoch boundary."""

    dataset: SequenceDataset
    model: torch.nn.Module
    base_model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LambdaLR
    config: dict[str, Any]
    context: DistributedContext
    rank_log: RankJsonlLog
    recorder: RunRecorder | None
    identity: dict[str, object]
    normalization: dict[str, list[float]]
    data_provenance: Mapping[str, object]
    output: Path
    profile_receipt: Path
    profile_hash: str
    save_state: Callable[[Path, dict[str, object], DistributedContext], None]
    monitor: SequenceMonitor | None
    evaluation_dataset: SequenceDataset | None
    train_eval_dataset: SequenceDataset | None


def run_epochs(
    runtime: FitEpochRuntime,
    start_epoch: int,
    global_step: int,
    history: list[dict[str, object]],
    latest_index: dict[str, object] | None,
    monitoring_enabled: bool,
) -> None:
    """Train, report validation before checkpoints, and finish the primary artifacts."""
    config, context, output = runtime.config, runtime.context, runtime.output
    for epoch in range(start_epoch, int(config["epochs"])):
        torch.cuda.reset_peak_memory_stats(context.device)
        result = train_epoch(
            runtime.dataset,
            runtime.model,
            runtime.optimizer,
            runtime.scheduler,
            config,
            context,
            runtime.rank_log,
            epoch,
            global_step,
            monitor=runtime.monitor,
            monitor_enabled=monitoring_enabled,
        )
        if not result.complete:
            raise RuntimeError("full fit ended before the complete epoch boundary")
        if not all_ranks_true(optimizer_states_are_finite_fp32(runtime.optimizer), context):
            raise FloatingPointError("optimizer state is not finite strict-FP32 at epoch boundary")
        global_step += result.updates
        if monitoring_enabled:
            coordinated_call(
                lambda current_epoch=epoch, current_step=global_step: (
                    runtime.monitor.evaluate_epoch(
                        runtime.base_model,
                        runtime.train_eval_dataset,
                        runtime.evaluation_dataset,
                        int(config["chunk_rows"]),
                        int(config["batch_size"]),
                        str(config.get("precision_mode", "fp32")),
                        context,
                        current_epoch,
                        current_step,
                    )
                    if context.is_primary and runtime.monitor is not None
                    else True
                ),
                "full report-only train-eval and valid evaluation",
                context,
            )
        summary = epoch_summary(result, config)
        history.append({"epoch": epoch, **summary})
        rates = optimizer_learning_rates(runtime.optimizer)
        coordinated_call(
            lambda current_summary=summary, current_epoch=epoch, current_step=global_step, current_rate=rates: (
                record_epoch_metrics(
                    runtime.recorder,
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
        monitor_offsets = current_monitor_offsets(output, context, monitoring_enabled)
        save_safe_state(
            runtime.save_state,
            output,
            runtime.identity,
            context,
            epoch + 1,
            global_step,
            runtime.base_model,
            runtime.optimizer,
            runtime.scheduler,
            history,
            runtime.rank_log,
            monitor_offsets,
        )
        if str(config.get("checkpoint_retention", "final_only")) == "all_epochs":
            latest_index = publish_epoch_snapshot(
                output,
                runtime.base_model,
                config,
                runtime.identity,
                runtime.normalization,
                runtime.data_provenance,
                runtime.profile_receipt,
                runtime.profile_hash,
                history,
                epoch,
                global_step,
                latest_index,
                context,
                "safe-epoch inference snapshot publication",
            )
    coordinated_call(
        lambda: (
            write_final_outputs(
                output,
                runtime.base_model,
                config,
                history,
                runtime.identity,
                runtime.normalization,
                runtime.data_provenance,
                runtime.profile_receipt,
                runtime.profile_hash,
                global_step,
                str(latest_index["index_sha256"]) if latest_index is not None else None,
            )
            if context.is_primary
            else True
        ),
        "fixed-final checkpoint write",
        context,
    )
    if monitoring_enabled:
        coordinated_call(
            lambda: (
                runtime.monitor.status("completed", global_step, int(config["epochs"]))
                if context.is_primary and runtime.monitor is not None
                else True
            ),
            "monitor completed status append",
            context,
        )
    barrier(context)
