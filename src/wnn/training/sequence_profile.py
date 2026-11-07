from __future__ import annotations

import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

try:
    import resource
except ImportError:  # pragma: no cover - exercised on Windows hosts
    resource = None

from wnn.research_reporting import RunRecorder
from wnn.training.sequence_checkpoint import atomic_write_json
from wnn.training.sequence_distributed import (
    DistributedContext,
    all_ranks_true,
    barrier,
    coordinated_call,
)
from wnn.training.sequence_fit_utils import (
    RankJsonlLog,
    SequenceDataset,
    optimizer_states_are_finite_fp32,
)
from wnn.training.sequence_loop import train_epoch
from wnn.training.sequence_monitor import SequenceMonitor


def _write_profile_receipt(receipt: dict[str, object], path: Path) -> bool:
    """Write one immutable profile receipt and return a coordination sentinel."""
    atomic_write_json(receipt, path)
    return True


def run_profile(
    dataset: SequenceDataset,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    config: dict[str, Any],
    context: DistributedContext,
    rank_log: RankJsonlLog,
    recorder: RunRecorder | None,
    identity: dict[str, object],
    output: Path,
    profile_updates: int,
    max_updates: int,
    monitor: SequenceMonitor | None = None,
    monitor_enabled: bool = False,
    monitoring_plan: Mapping[str, object] | None = None,
    monitor_event_budget: Mapping[str, object] | None = None,
) -> None:
    """Run bounded real-data updates and emit an unapproved engineering-only receipt."""
    if not 1 <= profile_updates <= max_updates:
        raise ValueError("--profile-updates must be within the configured full-fit update budget")
    if monitor_enabled:
        monitor_config = config.get("monitoring")
        if not isinstance(monitor_config, Mapping):
            raise ValueError("profile monitoring requires its validated monitoring config")
        diagnostic_interval = int(monitor_config["diagnostics_every_steps"])
        if profile_updates < diagnostic_interval:
            raise ValueError(
                "profile updates must reach diagnostics_every_steps to exercise sampled hooks"
            )
        _append_profile_status(monitor, "running", 0, 0, context)
    torch.cuda.reset_peak_memory_stats(context.device)
    started = time.perf_counter()
    updates, epochs_seen = 0, 0
    totals = torch.zeros(8, dtype=torch.float64, device=context.device)
    while updates < profile_updates and epochs_seen < int(config["epochs"]):
        result = train_epoch(
            dataset,
            model,
            optimizer,
            scheduler,
            config,
            context,
            rank_log,
            epochs_seen,
            updates,
            profile_updates - updates,
            monitor=monitor,
            monitor_enabled=monitor_enabled,
        )
        updates += result.updates
        totals += result.totals
        epochs_seen += int(result.complete)
        if result.updates == 0:
            break
    optimizer_finite = all_ranks_true(optimizer_states_are_finite_fp32(optimizer), context)
    errors = [] if optimizer_finite else ["optimizer state is not finite strict-FP32"]
    valid_rows = int(totals[3].item())
    forwarded_rows = int(totals[4].item())
    real_rows = int(totals[5].item())
    complete = updates == profile_updates and not errors
    local_memory = {
        "rank": context.rank,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(context.device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(context.device)),
        "device_total_memory_bytes": int(
            torch.cuda.get_device_properties(context.local_rank).total_memory
        ),
        "host_peak_rss_bytes": _host_peak_rss_bytes(),
    }
    if context.process_group:
        memory_by_rank: list[dict[str, int] | None] = [None] * context.world_size
        dist.all_gather_object(memory_by_rank, local_memory)
        memory = [item for item in memory_by_rank if item is not None]
    else:
        memory = [local_memory]
    if len(memory) != context.world_size:
        raise RuntimeError("profile resource receipt is missing a torchrun rank")
    memory.sort(key=lambda item: item["rank"])
    label_denom, hint_denom = float(totals[6].item()), float(totals[7].item())
    label_loss = float(totals[0].item()) / label_denom if label_denom > 0 else 0.0
    kd_loss = float(totals[1].item()) / label_denom if label_denom > 0 else 0.0
    hint_loss = float(totals[2].item()) / hint_denom if hint_denom > 0 else 0.0
    elapsed = time.perf_counter() - started
    if updates != profile_updates:
        errors.append("profile stopped before its requested optimizer update count")
        complete = False
    if min(valid_rows, forwarded_rows, real_rows) <= 0 or forwarded_rows < real_rows:
        errors.append("profile did not produce positive, internally consistent work counts")
        complete = False
    if updates < 1 or elapsed <= 0:
        errors.append("profile did not produce positive optimizer work and elapsed time")
        complete = False
    receipt = {
        "receipt_schema": "wnn.sequence-profile.v1",
        "status": "profile_complete" if complete else "profile_incomplete",
        "approved": False,
        "identity": identity,
        "updates": updates,
        "epochs_traversed": epochs_seen,
        "elapsed_seconds": elapsed,
        "forwarded_rows": forwarded_rows,
        "real_rows": real_rows,
        "valid_rows": valid_rows,
        "losses": {
            "label": label_loss,
            "distillation": kd_loss,
            "hint": hint_loss,
            "total": label_loss
            + float(config["kd_weight"]) * kd_loss
            + float(config["hint_weight"]) * hint_loss,
        },
        "model_parameters": identity["model_parameters"],
        "memory_by_rank": memory,
        "peak_memory_bytes": max(item["peak_allocated_bytes"] for item in memory),
        "peak_reserved_bytes": max(item["peak_reserved_bytes"] for item in memory),
        "device_total_memory_bytes": [item["device_total_memory_bytes"] for item in memory],
        "host_peak_rss_bytes": [item["host_peak_rss_bytes"] for item in memory],
        "errors": errors,
        "error_count": len(errors),
        "scientific_checkpoint_written": False,
        "approval_note": "Review the engineering evidence and set approved=true before a full fit.",
    }
    if monitoring_plan is not None:
        receipt["evaluation_annotation"] = {
            "no_eval_during_fit": True,
            "evaluation_reporting": "not evaluated during bounded engineering profile",
            "intended_evaluation_reporting": monitoring_plan.get(
                "evaluation_reporting", monitoring_plan.get("validation_reporting")
            ),
        }
        if "validation_reporting" in monitoring_plan:
            receipt["evaluation_annotation"].update(
                validation_reporting="not evaluated during bounded engineering profile",
                intended_validation_reporting=monitoring_plan["validation_reporting"],
            )
    if monitor_event_budget is not None:
        receipt["monitor_event_journal_budget"] = dict(monitor_event_budget)
    coordinated_call(
        lambda: (
            _write_profile_receipt(receipt, output / "profile_receipt.json")
            if context.is_primary
            else True
        ),
        "profile receipt write",
        context,
    )
    barrier(context)
    coordinated_call(
        lambda: (
            _record_profile_metrics(
                recorder,
                config,
                updates,
                elapsed,
                label_loss,
                kd_loss,
                hint_loss,
                valid_rows,
                forwarded_rows,
            )
            if context.is_primary and recorder is not None
            else True
        ),
        "profile metrics record",
        context,
    )
    if monitor_enabled:
        _append_profile_status(
            monitor,
            "profile_complete" if complete else "profile_incomplete",
            updates,
            epochs_seen,
            context,
        )
    if not complete:
        raise RuntimeError("profile failed its requested work or optimizer-state checks")


def _append_profile_status(
    monitor: SequenceMonitor | None,
    state: str,
    updates: int,
    epochs_seen: int,
    context: DistributedContext,
) -> None:
    """Append an honest profile-only status while coordinating rank-zero I/O errors."""

    def append() -> bool:
        """Write profile status only from rank zero and identify validation as unrun."""
        if context.is_primary:
            if monitor is None:
                raise RuntimeError("rank zero monitor was not initialized for the profile")
            monitor.status(state, updates, epochs_seen, monitor.profile_status_context())
        return True

    coordinated_call(append, "profile monitor status append", context)


def _host_peak_rss_bytes() -> int | None:
    """Normalize the platform-specific resource.ru_maxrss unit to bytes."""
    if resource is None:
        return None
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _record_profile_metrics(
    recorder: RunRecorder | None,
    config: dict[str, Any],
    updates: int,
    elapsed: float,
    label_loss: float,
    kd_loss: float,
    hint_loss: float,
    valid_rows: int,
    forwarded_rows: int,
) -> bool:
    """Record one bounded engineering profile summary in repository run metadata."""
    if recorder is None:
        return True
    recorder.record_metrics(
        step=updates,
        epoch=0,
        loss=label_loss
        + float(config["kd_weight"]) * kd_loss
        + float(config["hint_weight"]) * hint_loss,
        throughput={
            "valid_rows_per_second": valid_rows / max(elapsed, 1e-12),
            "forwarded_rows_per_second": forwarded_rows / max(elapsed, 1e-12),
        },
        timing={"profile_seconds": elapsed},
    )
    return True
