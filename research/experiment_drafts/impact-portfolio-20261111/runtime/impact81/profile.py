"""Impact81 bounded profile lifecycle over an injected epoch engine."""

from __future__ import annotations

import math
import resource
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from wnn.training.sequence_monitor_progress import MonitorProgress
from wnn.training.sequence_profile_reload import check_profile_reload

from impact81.engine import EpochResult, TrainingEngine
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
from wnn.training.sequence_profile import _record_profile_metrics
from wnn.training.sequence_provenance import json_value


def _write_profile_receipt(receipt: dict[str, Any], path: Path) -> bool:
    """Write an immutable profile receipt and return a coordination sentinel."""
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
    *,
    engine: TrainingEngine,
    progress: MonitorProgress | None = None,
) -> None:
    """Run exactly the requested updates and write engineering-only evidence."""
    if not 1 <= profile_updates <= max_updates:
        raise ValueError("--profile-updates must be within the configured full-fit update budget")
    torch.cuda.reset_peak_memory_stats(context.device)
    started = time.perf_counter()
    updates, epochs_seen = 0, 0
    totals = torch.zeros(8, dtype=torch.float64, device=context.device)
    results: list[EpochResult] = []
    while updates < profile_updates and epochs_seen < int(config["epochs"]):
        result = engine.train_epoch(
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
            progress=progress,
        )
        if result.updates < 0 or updates + result.updates > profile_updates:
            raise RuntimeError("profile epoch driver exceeded its requested update bound")
        results.append(result)
        updates += result.updates
        totals += result.totals
        epochs_seen += int(result.complete)
        if result.updates == 0:
            break

    reload_check = check_profile_reload(model, optimizer, scheduler, context, output)
    optimizer_finite = all_ranks_true(optimizer_states_are_finite_fp32(optimizer), context)
    errors = [] if optimizer_finite else ["optimizer state is not finite strict-FP32"]
    valid_rows = int(totals[3].item())
    forwarded_rows = int(totals[4].item())
    real_rows = int(totals[5].item())
    elapsed = time.perf_counter() - started
    if updates != profile_updates:
        errors.append("profile stopped before its requested optimizer update count")
    if min(valid_rows, forwarded_rows, real_rows) <= 0 or forwarded_rows < real_rows:
        errors.append("profile did not produce positive, internally consistent work counts")
    if updates < 1 or elapsed <= 0:
        errors.append("profile did not produce positive optimizer work and elapsed time")
    complete = not errors

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
    profile_fields = json_value(dict(engine.profile_fields(results, config)))
    if not isinstance(profile_fields, dict):
        raise ValueError("profile_fields must return a JSON object")
    total_loss = profile_fields.pop("total_loss", None)
    if (
        isinstance(total_loss, bool)
        or not isinstance(total_loss, (int, float))
        or not math.isfinite(total_loss)
    ):
        raise ValueError("profile_fields must include a finite combined total_loss")

    signed_objective = config["impact81"]["cell"]["variant"] == "nll_transfer"
    receipt_total = float(total_loss)
    if signed_objective:
        # Canonical profile admission requires nonnegative losses. Retain its
        # primary field and expose the actual signed NLL objective explicitly.
        receipt_total = float(profile_fields["primary_loss"])
        profile_fields["total_loss"] = float(total_loss)

    receipt: dict[str, Any] = {
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
            "total": receipt_total,
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
        "engineering_reload": reload_check,
        "approval_note": "Review the engineering evidence and set approved=true before a full fit.",
    }
    if profile_fields:
        receipt["objective_terms"] = profile_fields
    if signed_objective:
        receipt["losses_total_scope"] = "primary_only_signed_objective_in_terms"

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
                float(total_loss),
            )
            if context.is_primary and recorder is not None
            else True
        ),
        "profile metrics record",
        context,
    )
    if not complete:
        raise RuntimeError("profile failed its requested work or optimizer-state checks")


def _host_peak_rss_bytes() -> int:
    """Normalize the platform-specific resource.ru_maxrss unit to bytes."""
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024
