from __future__ import annotations

import time
from typing import Any

import torch

from wnn.training.sequence_auxiliary import (
    build_auxiliary_targets,
)
from wnn.training.sequence_blend import blend_loss_metrics
from wnn.training.sequence_distributed import (
    DistributedContext,
    coordinated_call,
    global_sum,
    local_sequence_indices,
)
from wnn.training.sequence_ensemble_reporting import objective_metrics
from wnn.training.sequence_fit_utils import (
    SEQUENCE_ROWS,
    RankJsonlLog,
    SequenceDataset,
    collate_sequences,
    move_and_validate_batch,
)
from wnn.training.sequence_loop_backprop import (
    EpochResult,
    _backward_window,
    _window_denominators,
)
from wnn.training.sequence_monitor_progress import MonitorProgress
from wnn.training.sequence_objectives import (
    since_event_row_weights,
)
from wnn.training.sequence_optimizer_telemetry import optimizer_learning_rates
from wnn.training.sequence_update import finish_sampled_update, finish_update
from wnn.training.sequence_window_reporting import window_record as _window_record


def _chunks(row_count: int, chunk_rows: int) -> list[tuple[int, int]]:
    """Return nonoverlapping row intervals covering the entire fixed sequence."""
    return [
        (start, min(start + chunk_rows, row_count)) for start in range(0, row_count, chunk_rows)
    ]


def train_epoch(
    dataset: SequenceDataset,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    config: dict[str, object],
    context: DistributedContext,
    rank_log: RankJsonlLog,
    epoch: int,
    global_step: int,
    max_updates: int | None = None,
    *,
    progress: MonitorProgress | None = None,
) -> EpochResult:
    """Run complete sequences in padded distributed order with fixed-window DDP scaling."""
    started = time.perf_counter()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    size = len(dataset)
    indices = local_sequence_indices(size, context, int(config["seed"]), epoch)
    batch_size = int(config["batch_size"])
    chunk_intervals = _chunks(SEQUENCE_ROWS, int(config["chunk_rows"]))
    accumulation = int(config["accumulate_chunks"])
    steps_per_epoch = ((len(indices) + batch_size - 1) // batch_size) * (
        (len(chunk_intervals) + accumulation - 1) // accumulation
    )
    total_count = torch.zeros(8, dtype=torch.float64, device=context.device)
    objective_totals = (
        torch.zeros(5, dtype=torch.float64, device=context.device)
        if config.get("objective_mode")
        else None
    )
    auxiliary_totals = (
        torch.zeros(4, dtype=torch.float64, device=context.device)
        if config["model"].get("auxiliary_heads")
        else None
    )
    updates = 0
    for batch_start in range(0, len(indices), batch_size):
        batch_ids = indices[batch_start : batch_start + batch_size]
        batch = coordinated_call(
            lambda batch_ids=batch_ids: move_and_validate_batch(
                collate_sequences(dataset, batch_ids),
                context.device,
                int(config["model"]["hint_dim"]),
            ),
            "sequence batch load and validation",
            context,
        )
        target, teacher, mask, hints, x = (
            batch["y"],
            batch["teacher"],
            batch["mask"],
            batch["hints"],
            batch["x"],
        )
        label_row_weights = None
        auxiliary_targets = build_auxiliary_targets(
            target, mask, config["model"].get("auxiliary_heads", [])
        )
        since_weight = float(config.get("since_weight", 0.0))
        if since_weight > 0:
            label_row_weights = since_event_row_weights(
                x, since_weight, float(config.get("since_floor", 0.25))
            )
        real_sequence_count = sum(index >= 0 for index in batch_ids)
        state: Any = None
        for window_start in range(0, len(chunk_intervals), accumulation):
            if max_updates is not None and updates >= max_updates:
                return EpochResult(
                    updates,
                    False,
                    total_count,
                    time.perf_counter() - started,
                    objective_totals,
                    auxiliary_totals,
                )
            window_started = time.perf_counter()
            intervals = chunk_intervals[window_start : window_start + accumulation]
            denominators = _window_denominators(
                target, mask, hints, intervals, auxiliary_targets, context
            )
            optimizer.zero_grad(set_to_none=True)
            rates = optimizer_learning_rates(optimizer)
            lr = float(optimizer.param_groups[0]["lr"])
            objective_stats = (
                torch.zeros_like(objective_totals) if objective_totals is not None else None
            )
            diagnostics_interval = config.get("optimizer_update_diagnostics_every")
            state, local_stats = _backward_window(
                model,
                optimizer,
                target,
                teacher,
                hints,
                mask,
                x,
                label_row_weights,
                intervals,
                real_sequence_count,
                denominators,
                auxiliary_targets,
                state,
                config,
                context,
                objective_stats,
            )
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
            global_stats = global_sum(local_stats, context)
            total_count[:6] += global_stats[:6]
            total_count[6:] += denominators[:2]
            if auxiliary_totals is not None:
                if global_stats.numel() != 8 or denominators.numel() != 4:
                    raise RuntimeError("auxiliary training statistics are incomplete")
                auxiliary_totals[:2] += global_stats[6:8]
                auxiliary_totals[2:] += denominators[2:]
            if objective_stats is not None:
                objective_stats[3] = global_stats[0] / denominators[0].clamp_min(1e-8)
                objective_stats[4] = 1
                objective_totals += objective_stats
            elapsed = time.perf_counter() - window_started
            record = _window_record(
                global_stats,
                denominators,
                context.rank,
                epoch,
                global_step,
                lr,
                gradient_norm,
                elapsed,
                torch.cuda.max_memory_allocated(context.device),
                float(config["grad_clip"]),
                float(config["kd_weight"]),
                float(config["hint_weight"]),
            )
            record["learning_rates"] = rates
            record.update(blend_loss_metrics(config, float(record["label_loss"])))
            record.update(objective_metrics(config, objective_stats))
            if auxiliary_totals is not None:
                structure_loss = (
                    float(global_stats[6].item()) / float(denominators[2].item())
                    if denominators[2] > 0
                    else 0.0
                )
                next_loss = (
                    float(global_stats[7].item()) / float(denominators[3].item())
                    if denominators[3] > 0
                    else 0.0
                )
                record["aux_structure_loss"] = structure_loss
                record["aux_next_change_loss"] = next_loss
                record["total_loss"] += (
                    float(config["aux_structure_weight"]) * structure_loss
                    + float(config["aux_next_change_weight"]) * next_loss
                )
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
    return EpochResult(
        updates,
        True,
        total_count,
        time.perf_counter() - started,
        objective_totals,
        auxiliary_totals,
    )
