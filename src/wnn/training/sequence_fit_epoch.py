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
from wnn.training.sequence_fit_monitor_epoch import (
    evaluate_full_valid,
    publish_checkpoint_status,
)
from wnn.training.sequence_fit_resume import current_monitor_offsets, save_safe_state
from wnn.training.sequence_fit_utils import (
    RankJsonlLog,
    SequenceDataset,
    optimizer_states_are_finite_fp32,
)
from wnn.training.sequence_loop import train_epoch
from wnn.training.sequence_monitor import SequenceMonitor
from wnn.training.sequence_monitor_checkpoints import (
    advance_early_stop,
)
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
    monitoring = config.get("monitoring")
    early_policy = monitoring.get("early_stopping") if isinstance(monitoring, Mapping) else None
    evaluation_role = str(runtime.data_provenance.get("evaluation_role", "valid"))
    stopped_early = bool(history and history[-1].get("early_stopped", False))
    for epoch in range(start_epoch, int(config["epochs"])):
        if stopped_early:
            break
        if runtime.monitor is not None:
            runtime.monitor.progress.phase("training", global_step, epoch)
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
        evaluation: dict[str, int | float] | None = None
        if monitoring_enabled:
            if not isinstance(monitoring, Mapping):
                raise TypeError("enabled monitor configuration must be a mapping")
            evaluation = evaluate_full_valid(
                runtime.monitor,
                runtime.base_model,
                runtime.train_eval_dataset,
                runtime.evaluation_dataset,
                config,
                monitoring,
                context,
                epoch,
                global_step,
            )
        summary = epoch_summary(result, config)
        history_row: dict[str, object] = {"epoch": epoch, **summary}
        if evaluation is not None:
            stopped_early = _record_evaluation_history(
                history, history_row, evaluation, evaluation_role, epoch + 1, early_policy
            )
        else:
            stopped_early = False
        history.append(history_row)
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
        if isinstance(early_policy, Mapping):
            if latest_index is None:
                raise RuntimeError("early-stop monitoring requires retained epoch checkpoints")
            epoch_status = (
                "early_stopped"
                if stopped_early
                else "completed"
                if epoch + 1 == int(config["epochs"])
                else "running"
            )

            publish_checkpoint_status(
                output,
                runtime.identity,
                epoch_status,
                global_step,
                epoch + 1,
                history,
                latest_index,
                early_policy,
                runtime.monitor,
                context,
                {
                    "valid_wp": float(evaluation["valid/wp"]),
                    "stop_reason": "patience_exhausted" if stopped_early else None,
                },
                "monitored checkpoint pointer update",
            )
        if stopped_early:
            break
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
        "completed checkpoint write",
        context,
    )
    if monitoring_enabled:
        final_status = "early_stopped" if stopped_early else "completed"
        publish_checkpoint_status(
            output,
            runtime.identity,
            final_status,
            global_step,
            len(history),
            history,
            latest_index,
            early_policy if isinstance(early_policy, Mapping) else None,
            runtime.monitor,
            context,
            {"completion_kind": "declared_early_stop" if stopped_early else "max_epochs"},
            "monitor terminal status append",
        )
    barrier(context)


def _record_evaluation_history(
    history: list[dict[str, object]],
    history_row: dict[str, object],
    evaluation: dict[str, int | float],
    evaluation_role: str,
    completed_epoch: int,
    early_policy: Mapping[str, object] | None,
) -> bool:
    """Store source-qualified metrics and advance primary-only stopping state."""
    history_row.update(
        {
            "train_eval_wp": float(evaluation["train_eval/wp"]),
            "train_eval_wp_t0": float(evaluation["train_eval/wp/t0"]),
            "train_eval_wp_t1": float(evaluation["train_eval/wp/t1"]),
            f"{evaluation_role}_wp": float(evaluation[f"{evaluation_role}/wp"]),
            f"{evaluation_role}_wp_t0": float(evaluation[f"{evaluation_role}/wp/t0"]),
            f"{evaluation_role}_wp_t1": float(evaluation[f"{evaluation_role}/wp/t1"]),
        }
    )
    if early_policy is None:
        return False
    if evaluation_role != "valid":
        raise ValueError("early-stop checkpoint pointers currently require primary valid")
    decision = advance_early_stop(
        history,
        completed_epoch,
        float(evaluation[f"{evaluation_role}/wp"]),
        early_policy,
    )
    history_row.update(
        {
            "early_stop_best_raw_wp": decision.best_wp,
            "early_stop_best_epoch": decision.best_epoch,
            "early_stop_patience_reference_wp": decision.reference_wp,
            "early_stop_delta_improved": decision.delta_improved,
            "early_stop_best_improved": decision.best_improved,
            "early_stop_bad_checks": decision.bad_checks,
            "early_stopped": decision.should_stop,
        }
    )
    return decision.should_stop
