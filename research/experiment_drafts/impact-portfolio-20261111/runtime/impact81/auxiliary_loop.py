"""Experiment-local epoch driver for new losses; canonical runtime remains untouched."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from wnn.training.sequence_ensemble_reporting import objective_metrics
from wnn.training.sequence_monitor_progress import MonitorProgress
from wnn.training.sequence_window_reporting import window_record

from impact81.auxiliary_backward import auxiliary_denominators, backward_window
from impact81.auxiliary_reporting import (
    WIDTH,
    metrics,
    profile_fields,
    summarize_epoch,
    window_totals,
)
from impact81.engine import TrainingEngine
from wnn.training.sequence_distributed import (
    DistributedContext,
    coordinated_call,
    global_sum,
    local_sequence_indices,
)
from wnn.training.sequence_fit_utils import (
    SEQUENCE_ROWS,
    RankJsonlLog,
    SequenceDataset,
    collate_sequences,
    move_and_validate_batch,
)
from wnn.training.sequence_loop import _chunks, _window_denominators
from wnn.training.sequence_optimizer_telemetry import optimizer_learning_rates
from wnn.training.sequence_update import finish_sampled_update, finish_update

# Model state and canonical config/progress receipts have validated heterogeneous types.


@dataclass(frozen=True)
class ExperimentEpochResult:
    """Canonical primary sufficient sums plus auxiliary telemetry over actual updates."""

    updates: int
    complete: bool
    totals: Tensor
    elapsed_seconds: float
    objective_totals: Tensor
    auxiliary_totals: Tensor


def get_engine() -> TrainingEngine:
    """Bind explicit callables without changing canonical module globals or functions."""
    return TrainingEngine(train_epoch, summarize_epoch, profile_fields)


def train_epoch(
    dataset: SequenceDataset,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    config: dict[str, Any],
    context: DistributedContext,
    rank_log: RankJsonlLog,
    epoch: int,
    global_step: int,
    max_updates: int | None = None,
    *,
    progress: MonitorProgress | None = None,
) -> ExperimentEpochResult:
    """Execute unchanged row/update scheduling with an explicit auxiliary objective seam."""
    started = time.perf_counter()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    if (
        config["kd_weight"]
        or config["hint_weight"]
        or config["objective_mode"] != "standalone_labels_wmse_v2"
    ):
        raise ValueError("cards04–06 require the frozen standalone label objective")
    variant = config["impact81"]["cell"]["variant"]
    if config["impact81"]["cell"]["card"] not in ("04", "05", "06"):
        raise ValueError("auxiliary epoch engine is restricted to cards04–06")
    indices = local_sequence_indices(len(dataset), context, int(config["seed"]), epoch)
    batch_size = int(config["batch_size"])
    intervals_all = _chunks(SEQUENCE_ROWS, int(config["chunk_rows"]))
    accumulation = int(config["accumulate_chunks"])
    steps_per_epoch = ((len(indices) + batch_size - 1) // batch_size) * (
        (len(intervals_all) + accumulation - 1) // accumulation
    )
    totals = torch.zeros(8, dtype=torch.float64, device=context.device)
    objective_totals = torch.zeros(5, dtype=torch.float64, device=context.device)
    auxiliary_totals = torch.zeros(WIDTH, dtype=torch.float64, device=context.device)
    updates = 0
    for batch_start in range(0, len(indices), batch_size):
        ids = indices[batch_start : batch_start + batch_size]
        batch = coordinated_call(
            lambda ids=ids: move_and_validate_batch(
                collate_sequences(dataset, ids), context.device, 0
            ),
            "experiment sequence batch load",
            context,
        )
        real_sequences = sum(index >= 0 for index in ids)
        state: Any = None
        for start in range(0, len(intervals_all), accumulation):
            if max_updates is not None and updates >= max_updates:
                return ExperimentEpochResult(
                    updates,
                    False,
                    totals,
                    time.perf_counter() - started,
                    objective_totals,
                    auxiliary_totals,
                )
            window_started = time.perf_counter()
            intervals = intervals_all[start : start + accumulation]
            denominators = _window_denominators(
                batch["y"], batch["mask"], batch["hints"], intervals, context
            )
            pair_mass = auxiliary_denominators(
                batch["y"], batch["mask"], intervals, variant, context
            )
            optimizer.zero_grad(set_to_none=True)
            rates = optimizer_learning_rates(optimizer)
            lr = float(optimizer.param_groups[0]["lr"])
            state, local, raw_auxiliary = backward_window(
                model,
                batch,
                intervals,
                denominators,
                pair_mass,
                state,
                variant,
                real_sequences,
                config,
                context,
            )
            diagnostics_interval = config.get("optimizer_update_diagnostics_every")
            if diagnostics_interval is None:
                gradient_norm, optimizer_seconds = finish_update(
                    model, optimizer, scheduler, context, float(config["grad_clip"])
                )
                update_diagnostics = None
            else:
                gradient_norm, optimizer_seconds, update_diagnostics = finish_sampled_update(
                    model,
                    optimizer,
                    scheduler,
                    context,
                    float(config["grad_clip"]),
                    global_step + 1,
                    int(diagnostics_interval),
                )
            updates += 1
            global_step += 1
            stats = global_sum(local, context)
            auxiliary = window_totals(global_sum(raw_auxiliary, context), denominators[0], variant)
            totals[:6] += stats
            totals[6:] += denominators
            objective = torch.zeros(5, dtype=torch.float64, device=context.device)
            objective[3] = stats[0] / denominators[0].clamp_min(1e-8)
            objective[4] = 1
            objective_totals += objective
            auxiliary_totals += auxiliary
            record = window_record(
                stats,
                denominators,
                context.rank,
                epoch,
                global_step,
                lr,
                gradient_norm,
                time.perf_counter() - window_started,
                torch.cuda.max_memory_allocated(context.device),
                float(config["grad_clip"]),
                0.0,
                0.0,
            )
            record.update(objective_metrics(config, objective))
            auxiliary_metrics = metrics(auxiliary)
            record["primary_loss"] = float(objective[3].item())
            record["total_loss"] = (
                record["primary_loss"] + auxiliary_metrics["weighted_auxiliary_loss"]
            )
            record["impact81_auxiliary"] = auxiliary_metrics
            record["learning_rates"] = rates
            record["optimizer_seconds"] = optimizer_seconds
            if update_diagnostics is not None:
                record["optimizer_update_diagnostics"] = update_diagnostics
            rank_log.write(record)
            if progress is not None:
                progress.optimizer_update(
                    global_step,
                    epoch + updates / steps_per_epoch,
                    float(record["forwarded_rows_per_second"]),
                )
        state = None
    return ExperimentEpochResult(
        updates, True, totals, time.perf_counter() - started, objective_totals, auxiliary_totals
    )
