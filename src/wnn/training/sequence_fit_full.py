from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch

from wnn.research_reporting import RunRecorder
from wnn.training.sequence_checkpoint import gather_rank_log_offsets, save_training_state
from wnn.training.sequence_distributed import (
    DistributedContext,
    coordinated_call,
)
from wnn.training.sequence_epoch_fit import reconcile_epoch_snapshots
from wnn.training.sequence_epoch_index import MAX_RETAINED_EPOCHS
from wnn.training.sequence_fit_epoch import FitEpochRuntime, run_epochs
from wnn.training.sequence_fit_resume import (
    current_monitor_offsets,
    load_resume,
    training_state,
)
from wnn.training.sequence_fit_utils import (
    RankJsonlLog,
    SequenceDataset,
)
from wnn.training.sequence_monitor import SequenceMonitor


def run_full_fit_impl(
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
    evaluation_dataset: SequenceDataset | None = None,
    train_eval_dataset: SequenceDataset | None = None,
    monitor: SequenceMonitor | None = None,
    save_state: Callable[[Path, dict[str, object], DistributedContext], None] = save_training_state,
) -> None:
    """Resume or complete a fit, with report-only online validation when opted in."""
    history: list[dict[str, object]] = []
    global_step, start_epoch = 0, 0
    retention = str(config.get("checkpoint_retention", "final_only"))
    if retention not in ("final_only", "all_epochs"):
        raise ValueError("unsupported checkpoint retention policy")
    if retention == "all_epochs" and int(config["epochs"]) > MAX_RETAINED_EPOCHS:
        raise ValueError("all-epoch retention supports at most 10,000 configured epochs")
    monitoring_enabled = "monitoring" in config
    if monitoring_enabled and int(config["epochs"]) > 10_000:
        raise ValueError("monitored fits support at most 10,000 configured epochs")
    if monitoring_enabled and (evaluation_dataset is None or train_eval_dataset is None):
        raise ValueError("monitored fit requires the fixed train-eval cohort and full valid set")
    if monitoring_enabled:
        coordinated_call(
            lambda: _require_primary_monitor(context, monitor),
            "rank-zero monitor availability",
            context,
        )
    latest_index: dict[str, object] | None = None
    if resume_path is not None:
        start_epoch, global_step, history = load_resume(
            resume_path,
            identity,
            base_model,
            optimizer,
            scheduler,
            context,
            rank_log,
            output,
            monitor,
            monitoring_enabled,
        )
    else:
        offsets = gather_rank_log_offsets(rank_log.position(), context)
        monitor_offsets = current_monitor_offsets(output, context, monitoring_enabled)
        save_state(
            output / "training_state.pt",
            training_state(
                identity,
                context.world_size,
                0,
                0,
                base_model,
                optimizer,
                scheduler,
                history,
                offsets,
                0,
                monitor_offsets,
            ),
            context,
        )
    if not 0 <= start_epoch <= int(config["epochs"]):
        raise ValueError("resume checkpoint next_epoch is outside the configured schedule")
    if monitoring_enabled:
        coordinated_call(
            lambda: (
                monitor.status(
                    "running",
                    global_step,
                    start_epoch,
                    monitor.fit_status_context(resume_path is not None),
                )
                if context.is_primary and monitor is not None
                else True
            ),
            "monitor running status append",
            context,
        )
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
    run_epochs(
        FitEpochRuntime(
            dataset,
            model,
            base_model,
            optimizer,
            scheduler,
            config,
            context,
            rank_log,
            recorder,
            identity,
            normalization,
            data_provenance,
            output,
            profile_receipt,
            profile_hash,
            save_state,
            monitor,
            evaluation_dataset,
            train_eval_dataset,
        ),
        start_epoch,
        global_step,
        history,
        latest_index,
        monitoring_enabled,
    )


def _require_primary_monitor(context: DistributedContext, monitor: SequenceMonitor | None) -> bool:
    """Fail collectively if rank zero did not initialize the event writer."""
    if context.is_primary and monitor is None:
        raise RuntimeError("rank zero monitor was not initialized")
    return True
