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
from wnn.training.sequence_features import sequence_input_width
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
from wnn.training.sequence_monitor_progress import record_successful_update
from wnn.training.sequence_optimizer_telemetry import optimizer_learning_rates
from wnn.training.sequence_prefetch import SequenceBatchPrefetcher
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
    model_config = config["model"]
    feature_count = sequence_input_width(str(model_config.get("input_feature_mode", "raw112")))
    batches_per_rank = math.ceil(math.ceil(len(dataset) / context.world_size) / batch_size)
    windows_per_batch = math.ceil(len(intervals) / accumulation)
    updates_per_epoch = batches_per_rank * windows_per_batch
    total_count = torch.zeros(8, dtype=torch.float64, device=context.device)
    updates = 0
    prefetch_batches = config.get("input_prefetch_batches", 0)
    if type(prefetch_batches) is not int or prefetch_batches not in (0, 1):
        raise ValueError("input_prefetch_batches must be 0 or 1")
    prefetcher = None
    if prefetch_batches:
        if context.world_size != 1:
            raise ValueError("input batch prefetch is currently supported only for world_size=1")
        if context.device.type != "cuda":
            raise ValueError("pinned sequence prefetch requires a CUDA training device")
        prefetcher = SequenceBatchPrefetcher(dataset, indices, batch_size, context.device)
    try:
        for batch_start in range(0, len(indices), batch_size):
            if prefetcher is None:
                batch_ids = indices[batch_start : batch_start + batch_size]
                batch = coordinated_call(
                    lambda current_batch_ids=batch_ids: move_and_validate_batch(
                        collate_sequences(dataset, current_batch_ids),
                        context.device,
                        int(config["model"]["hint_dim"]),
                        feature_count,
                    ),
                    "sequence batch load and validation",
                    context,
                )
            else:
                prefetched = next(prefetcher)
                batch_ids = list(prefetched.indices)
                prefetched.wait_on_current_stream()
                batch = move_and_validate_batch(
                    prefetched.batch,
                    context.device,
                    int(config["model"]["hint_dim"]),
                    feature_count,
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
                hook_sample = interval > 0 and step % interval == 0
                optimizer_interval = int(config.get("optimizer_update_diagnostics_every", 0))
                optimizer_sample = optimizer_interval > 0 and (
                    step == 1 or step % optimizer_interval == 0
                )
                probe = (
                    monitor.diagnostic_probe(model)
                    if monitor is not None and hook_sample
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
                    if diagnostics is not None:
                        diagnostics.capture_parameter_gradient_status()
                gradient_norm, optimizer_seconds, optimizer_update = finish_update(
                    model,
                    optimizer,
                    scheduler,
                    context,
                    float(config["grad_clip"]),
                    update_step=step,
                    diagnostics_interval=optimizer_interval,
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
                if optimizer_update is not None:
                    record["optimizer_update_diagnostics"] = optimizer_update
                if monitor is not None:
                    record_successful_update(
                        monitor,
                        global_step,
                        epoch_position,
                        float(record["forwarded_rows_per_second"]),
                    )
                if monitor_enabled:
                    monitor_config = config.get("monitoring")
                    if not isinstance(monitor_config, dict):
                        raise TypeError("enabled monitor configuration must be a mapping")
                    telemetry_interval = int(
                        monitor_config.get("training_telemetry_every_steps", 1)
                    )
                    telemetry_sample = _training_telemetry_due(
                        global_step, updates, updates_per_epoch, telemetry_interval
                    )
                    if telemetry_sample or optimizer_sample:
                        # The same opt-in cadence bounds rank-local diagnostic JSONL growth.
                        rank_log.write(record)
                else:
                    telemetry_sample = True
                    rank_log.write(record)
                if monitor_enabled and (telemetry_sample or hook_sample or optimizer_sample):
                    _append_monitor_events(
                        record,
                        epoch,
                        epoch_position,
                        global_step,
                        telemetry_sample,
                        hook_sample,
                        optimizer_sample,
                        diagnostics,
                        monitor,
                        context,
                    )
            state = None
            if prefetcher is not None:
                prefetched.release_pinned_source()
        return EpochResult(updates, True, total_count, time.perf_counter() - started)
    finally:
        if prefetcher is not None:
            prefetcher.close()


def _training_telemetry_due(
    step: int, updates_in_epoch: int, updates_per_epoch: int, cadence: int
) -> bool:
    """Retain sampled optimizer records and every epoch's final update."""
    if min(step, updates_in_epoch, updates_per_epoch, cadence) < 1:
        raise ValueError("telemetry counters and cadence must be positive")
    return step % cadence == 0 or updates_in_epoch == updates_per_epoch


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
    telemetry_sample: bool,
    hook_sample: bool,
    optimizer_sample: bool,
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
            if telemetry_sample:
                monitor.train_step(record, epoch, epoch_position, context.device)
            if hook_sample or optimizer_sample:
                if hook_sample and diagnostics is None:
                    raise RuntimeError("rank zero diagnostic probe is missing")
                hook_metrics = diagnostics.metrics() if diagnostics is not None else {}
                optimizer_update = record.get("optimizer_update_diagnostics")
                monitor.diagnostics(
                    step,
                    epoch,
                    epoch_position,
                    hook_metrics,
                    optimizer_update if isinstance(optimizer_update, dict) else None,
                )
            if hook_sample:
                monitor.resources(step, epoch, epoch_position, context.device)
        return True

    coordinated_call(append, "online monitor event append", context)
