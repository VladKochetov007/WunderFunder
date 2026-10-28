from __future__ import annotations

import math
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch

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
from wnn.training.sequence_loop_window import (
    backward_window,
    window_denominators_global,
    window_record,
)
from wnn.training.sequence_monitor import SequenceMonitor
from wnn.training.sequence_monitor_diagnostics import DiagnosticProbe
from wnn.training.sequence_optimizer_telemetry import optimizer_learning_rates
from wnn.training.sequence_update import finish_update


@dataclass(frozen=True)
class EpochResult:
    """Completed or bounded sequence-pass metrics and aggregate sufficient sums."""

    updates: int
    complete: bool
    totals: torch.Tensor
    elapsed_seconds: float


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
    monitor: SequenceMonitor | None = None,
    monitor_enabled: bool = False,
) -> EpochResult:
    """Run complete sequences in padded order with fixed-window DDP scaling."""
    started = time.perf_counter()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    indices = local_sequence_indices(len(dataset), context, int(config["seed"]), epoch)
    batch_size = int(config["batch_size"])
    chunk_rows = int(config["chunk_rows"])
    intervals = [
        (start, min(start + chunk_rows, SEQUENCE_ROWS))
        for start in range(0, SEQUENCE_ROWS, chunk_rows)
    ]
    accumulation = int(config["accumulate_chunks"])
    batches_per_rank = math.ceil(math.ceil(len(dataset) / context.world_size) / batch_size)
    windows_per_batch = math.ceil(len(intervals) / accumulation)
    updates_per_epoch = batches_per_rank * windows_per_batch
    total_count = torch.zeros(8, dtype=torch.float64, device=context.device)
    updates = 0
    for batch_start in range(0, len(indices), batch_size):
        batch_ids = indices[batch_start : batch_start + batch_size]
        batch = coordinated_call(
            lambda current_batch_ids=batch_ids: move_and_validate_batch(
                collate_sequences(dataset, current_batch_ids),
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
        real_count = sum(index >= 0 for index in batch_ids)
        state: Any = None
        for window_start in range(0, len(intervals), accumulation):
            if max_updates is not None and updates >= max_updates:
                return EpochResult(updates, False, total_count, time.perf_counter() - started)
            window_started = time.perf_counter()
            chunks = intervals[window_start : window_start + accumulation]
            denominators = window_denominators_global(target, mask, hints, chunks, context)
            optimizer.zero_grad(set_to_none=True)
            rates = optimizer_learning_rates(optimizer)
            current_lr = float(optimizer.param_groups[0]["lr"])
            step = global_step + 1
            interval = _diagnostic_interval(config, monitor_enabled)
            sample = interval > 0 and step % interval == 0
            probe = (
                monitor.diagnostic_probe(model)
                if monitor is not None and sample
                else nullcontext(None)
            )
            with probe as diagnostics:
                state, local_stats = backward_window(
                    model,
                    optimizer,
                    target,
                    teacher,
                    hints,
                    mask,
                    x,
                    chunks,
                    real_count,
                    denominators,
                    state,
                    config,
                    context,
                    diagnostics,
                )
            gradient_norm, optimizer_seconds = finish_update(
                model, optimizer, scheduler, context, float(config["grad_clip"])
            )
            updates += 1
            global_step += 1
            epoch_position = epoch + updates / updates_per_epoch
            global_stats = global_sum(local_stats, context)
            total_count[:6] += global_stats
            total_count[6:] += denominators
            elapsed = time.perf_counter() - window_started
            record = window_record(
                global_stats,
                denominators,
                context.rank,
                epoch,
                global_step,
                current_lr,
                gradient_norm,
                elapsed,
                torch.cuda.max_memory_allocated(context.device),
                float(config["grad_clip"]),
                float(config["kd_weight"]),
                float(config["hint_weight"]),
            )
            record["learning_rates"] = rates
            record["optimizer_seconds"] = optimizer_seconds
            rank_log.write(record)
            if monitor_enabled:
                _append_monitor_events(
                    record,
                    epoch,
                    epoch_position,
                    global_step,
                    sample,
                    diagnostics,
                    monitor,
                    context,
                )
        state = None
    return EpochResult(updates, True, total_count, time.perf_counter() - started)


def _diagnostic_interval(config: dict[str, object], enabled: bool) -> int:
    monitor_config = config.get("monitoring")
    if not enabled or not isinstance(monitor_config, dict):
        return 0
    return int(monitor_config["diagnostics_every_steps"])


def _append_monitor_events(
    record: dict[str, object],
    epoch: int,
    epoch_position: float,
    step: int,
    sample: bool,
    diagnostics: DiagnosticProbe | None,
    monitor: SequenceMonitor | None,
    context: DistributedContext,
) -> None:
    """Coordinate aggregate event appends so rank-zero I/O errors reach all ranks."""

    def append() -> bool:
        """Write the current optimizer event and optional sampled diagnostics."""
        if context.is_primary:
            if monitor is None:
                raise RuntimeError("rank zero monitor was not initialized")
            monitor.train_step(record, epoch, epoch_position, context.device)
            if sample:
                if diagnostics is None:
                    raise RuntimeError("rank zero diagnostic probe is missing")
                monitor.diagnostics(step, epoch, epoch_position, diagnostics.metrics())
                monitor.resources(step, epoch, epoch_position, context.device)
        return True

    coordinated_call(append, "online monitor event append", context)
