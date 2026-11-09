from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.parallel import DistributedDataParallel

from wnn.training.sequence_blend import blend_loss_metrics
from wnn.training.sequence_distributed import (
    DistributedContext,
    all_ranks_true,
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
from wnn.training.sequence_models import detach_state
from wnn.training.sequence_monitor_progress import MonitorProgress
from wnn.training.sequence_objectives import (
    detached_sums,
    objective_numerators,
    scaled_objective,
    window_denominators,
)
from wnn.training.sequence_optimizer_telemetry import optimizer_learning_rates
from wnn.training.sequence_precision import backward_context, forward_context
from wnn.training.sequence_update import finish_sampled_update, finish_update
from wnn.training.sequence_window_reporting import window_record as _window_record


@dataclass(frozen=True)
class EpochResult:
    """Completed or bounded sequence-pass metrics, including aggregate sufficient sums."""

    updates: int
    complete: bool
    totals: torch.Tensor
    elapsed_seconds: float
    objective_totals: torch.Tensor | None = None


def _chunks(row_count: int, chunk_rows: int) -> list[tuple[int, int]]:
    """Return nonoverlapping row intervals covering the entire fixed sequence."""
    return [
        (start, min(start + chunk_rows, row_count)) for start in range(0, row_count, chunk_rows)
    ]


def _check_finite_loss(loss: torch.Tensor, context: DistributedContext) -> None:
    local_finite = bool(torch.isfinite(loss.detach()).all())
    if not all_ranks_true(local_finite, context):
        raise FloatingPointError("nonfinite objective detected consistently across torchrun ranks")


def _backward_window(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    target: torch.Tensor,
    teacher: torch.Tensor,
    hint_target: torch.Tensor,
    mask: torch.Tensor,
    x: torch.Tensor,
    intervals: list[tuple[int, int]],
    real_sequence_count: int,
    denominators: torch.Tensor,
    state: Any,  # Model-owned recurrent state has architecture-specific container structure.
    config: dict[str, object],
    context: DistributedContext,
    objective_stats: torch.Tensor | None = None,
) -> tuple[Any, torch.Tensor]:
    """Forward and backpropagate one pre-normalized TBPTT accumulation window."""
    local_stats = torch.zeros(6, dtype=torch.float64, device=context.device)
    for index, (start, stop) in enumerate(intervals):
        final_microchunk = index == len(intervals) - 1
        sync_context = (
            model.no_sync()
            if isinstance(model, DistributedDataParallel) and not final_microchunk
            else nullcontext()
        )
        attention_backend = str(config["model"].get("attention_backend", "math"))
        attention_context = (
            sdpa_kernel(SDPBackend.MATH) if attention_backend == "math" else nullcontext()
        )
        mode = str(config.get("precision_mode", "fp32"))
        with sync_context, attention_context:
            with forward_context(mode, context.device.type):
                prediction, hint_prediction, next_state = model(x[:, start:stop], state)
            state = detach_state(next_state)
            prediction = prediction.float()
            hint_prediction = hint_prediction.float()
            numerators = objective_numerators(
                prediction,
                hint_prediction,
                target[:, start:stop],
                teacher[:, start:stop],
                hint_target[:, start:stop],
                mask[:, start:stop],
                blend_objective_alpha=float(config.get("blend_objective_alpha", 0.0)),
                objective_mode=config.get("objective_mode"),
                context=context,
                correlation_prediction_std_floor=float(
                    config.get("correlation_prediction_std_floor", 0.01)
                ),
            )
            loss = scaled_objective(
                numerators,
                denominators,
                context.world_size,
                float(config["kd_weight"]),
                float(config["hint_weight"]),
                microchunks=len(intervals),
            )
            _check_finite_loss(loss, context)
            with backward_context(mode, context.device.type):
                loss.backward()
            if objective_stats is not None and numerators.correlation is not None:
                objective_stats[:3] += torch.cat(
                    (
                        numerators.correlation.detach(),
                        numerators.correlation_loss.detach().reshape(1),
                    )
                ).double() / len(intervals)
        local_stats[:5] += detached_sums(numerators, target[:, start:stop], mask[:, start:stop])
        local_stats[5] += real_sequence_count * (stop - start)
    return state, local_stats


def _window_denominators(
    target: torch.Tensor,
    mask: torch.Tensor,
    hints: torch.Tensor,
    intervals: list[tuple[int, int]],
    context: DistributedContext,
) -> torch.Tensor:
    """Reduce both objective denominators across every rank before any window backward."""
    local = torch.zeros(2, dtype=torch.float32, device=context.device)
    for start, stop in intervals:
        local += window_denominators(
            target[:, start:stop], mask[:, start:stop], hints[:, start:stop]
        )
    global_denominators = global_sum(local, context)
    if not all_ranks_true(bool(torch.isfinite(global_denominators).all()), context):
        raise FloatingPointError("nonfinite objective denominator detected across torchrun ranks")
    return global_denominators


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
        real_sequence_count = sum(index >= 0 for index in batch_ids)
        state: Any = None
        for window_start in range(0, len(chunk_intervals), accumulation):
            if max_updates is not None and updates >= max_updates:
                return EpochResult(
                    updates, False, total_count, time.perf_counter() - started, objective_totals
                )
            window_started = time.perf_counter()
            intervals = chunk_intervals[window_start : window_start + accumulation]
            denominators = _window_denominators(target, mask, hints, intervals, context)
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
                intervals,
                real_sequence_count,
                denominators,
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
            total_count[:6] += global_stats
            total_count[6:] += denominators
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
    return EpochResult(updates, True, total_count, time.perf_counter() - started, objective_totals)
