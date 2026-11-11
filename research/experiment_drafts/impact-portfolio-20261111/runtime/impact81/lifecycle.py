"""Impact81 fit lifecycle over a packet-owned epoch/objective driver."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from wnn.training.sequence_complement_monitor import ComplementMonitor

from impact81.engine import TrainingEngine
from impact81.lifecycle_support import (
    begin_fit,
    finish_fit,
    save_epoch_boundary,
)
from wnn.research_reporting import RunRecorder
from wnn.training.sequence_distributed import (
    DistributedContext,
    all_ranks_true,
    coordinated_call,
)
from wnn.training.sequence_fit_artifacts import record_epoch_metrics
from wnn.training.sequence_fit_utils import (
    RankJsonlLog,
    SequenceDataset,
    optimizer_states_are_finite_fp32,
)
from wnn.training.sequence_optimizer_telemetry import optimizer_learning_rates
from wnn.training.sequence_provenance import json_value


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
    engine: TrainingEngine,
    monitor: ComplementMonitor | None = None,
) -> None:
    """Run the shared checkpoint lifecycle while delegating objective math to ``engine``."""
    epochs = int(config["epochs"])
    start_epoch, global_step, history, latest_index, retention = begin_fit(
        config,
        base_model,
        optimizer,
        scheduler,
        context,
        rank_log,
        identity,
        output,
        resume_path,
        normalization,
        data_provenance,
        profile_receipt,
        profile_hash,
        monitor,
    )
    for epoch in range(start_epoch, epochs):
        if monitor is not None and monitor.progress is not None:
            monitor.progress.phase("training", global_step, epoch)
        torch.cuda.reset_peak_memory_stats(context.device)
        result = engine.train_epoch(
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
        if not result.complete or result.updates <= 0:
            raise RuntimeError("full fit ended before a complete, updating epoch boundary")
        if not all_ranks_true(optimizer_states_are_finite_fp32(optimizer), context):
            raise FloatingPointError("optimizer state is not finite strict-FP32 at epoch boundary")
        global_step += result.updates
        summary = json_value(dict(engine.summarize_epoch(result, config)))
        if not isinstance(summary, dict) or "total_loss" not in summary:
            raise ValueError("epoch summary must include JSON-safe total_loss")
        history.append({"epoch": epoch, **summary})
        if monitor is not None:
            coordinated_call(
                lambda current_epoch=epoch, current_step=global_step: monitor.evaluate(
                    base_model, config, context, current_epoch, current_step
                ),
                "report-only full official valid evaluation",
                context,
            )
        rates = optimizer_learning_rates(optimizer)
        coordinated_call(
            lambda current_summary=summary, current_epoch=epoch, current_step=global_step, current_rates=rates: (
                record_epoch_metrics(
                    recorder,
                    current_summary,
                    current_epoch,
                    current_step,
                    current_rates,
                )
                if context.is_primary
                else True
            ),
            "epoch metrics record",
            context,
        )
        latest_index = save_epoch_boundary(
            output=output,
            context=context,
            rank_log=rank_log,
            monitor=monitor,
            identity=identity,
            epoch=epoch,
            global_step=global_step,
            base_model=base_model,
            optimizer=optimizer,
            scheduler=scheduler,
            history=history,
            retention=retention,
            latest_index=latest_index,
            config=config,
            normalization=normalization,
            data_provenance=data_provenance,
            profile_receipt=profile_receipt,
            profile_hash=profile_hash,
        )
    finish_fit(
        output=output,
        context=context,
        monitor=monitor,
        global_step=global_step,
        epochs=epochs,
        latest_index=latest_index,
        base_model=base_model,
        config=config,
        history=history,
        identity=identity,
        normalization=normalization,
        data_provenance=data_provenance,
        profile_receipt=profile_receipt,
        profile_hash=profile_hash,
    )
