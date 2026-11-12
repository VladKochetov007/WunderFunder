"""Canonical safe-boundary helpers for the Impact81 fit lifecycle."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from wnn.training.sequence_complement_monitor import ComplementMonitor
from wnn.training.sequence_fit_state import training_state

from wnn.training.sequence_checkpoint import (
    gather_rank_log_offsets,
    save_training_state,
)
from wnn.training.sequence_distributed import DistributedContext, barrier, coordinated_call
from wnn.training.sequence_epoch_fit import (
    publish_epoch_snapshot,
    reconcile_epoch_snapshots,
)
from wnn.training.sequence_epoch_index import MAX_RETAINED_EPOCHS
from wnn.training.sequence_fit_artifacts import (
    metrics_offset,
    write_final_outputs,
)
from wnn.training.sequence_fit_utils import RankJsonlLog

try:
    # PR79 moved the helper to a public module and name.  Keep the exact PR75
    # fallback for the Transformer family, whose pinned runtime predates it.
    from wnn.training.sequence_fit_resume import load_resume as _load_resume
except ModuleNotFoundError as error:
    if error.name != "wnn.training.sequence_fit_resume":
        raise
    from wnn.training.sequence_fit_modes import _load_resume


def begin_fit(
    config: dict[str, Any],
    base_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    context: DistributedContext,
    rank_log: RankJsonlLog,
    identity: dict[str, object],
    output: Path,
    resume_path: Path | None,
    normalization: dict[str, list[float]],
    data_provenance: Mapping[str, object],
    profile_receipt: Path,
    profile_hash: str,
    monitor: ComplementMonitor | None,
) -> tuple[int, int, list[dict[str, object]], dict[str, object] | None, str]:
    """Restore or create the canonical safe boundary and reconcile retained snapshots."""
    epochs = int(config["epochs"])
    retention = str(config.get("checkpoint_retention", "final_only"))
    if retention not in ("final_only", "all_epochs"):
        raise ValueError("unsupported checkpoint retention policy")
    if retention == "all_epochs" and epochs > MAX_RETAINED_EPOCHS:
        raise ValueError("all-epoch retention supports at most 10,000 configured epochs")

    history: list[dict[str, object]] = []
    global_step, start_epoch = 0, 0
    latest_index: dict[str, object] | None = None
    if resume_path is not None:
        start_epoch, global_step, history = _load_resume(
            resume_path,
            identity,
            base_model,
            optimizer,
            scheduler,
            context,
            rank_log,
            output,
            monitor,
        )
    else:
        rank_offsets = gather_rank_log_offsets(rank_log.position(), context)
        if monitor is not None and monitor.progress is not None:
            monitor.progress.phase("checkpoint", 0, 0)
        save_training_state(
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
                rank_offsets,
                0,
                monitor.offsets() if monitor is not None else None,
            ),
            context,
        )
    if not 0 <= start_epoch <= epochs:
        raise ValueError("resume checkpoint next_epoch is outside the configured schedule")
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
    return start_epoch, global_step, history, latest_index, retention


def save_epoch_boundary(
    *,
    output: Path,
    context: DistributedContext,
    rank_log: RankJsonlLog,
    monitor: ComplementMonitor | None,
    identity: dict[str, object],
    epoch: int,
    global_step: int,
    base_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    history: list[dict[str, object]],
    retention: str,
    latest_index: dict[str, object] | None,
    config: dict[str, Any],
    normalization: dict[str, list[float]],
    data_provenance: Mapping[str, object],
    profile_receipt: Path,
    profile_hash: str,
) -> dict[str, object] | None:
    """Persist logs, optimizer/model/RNG state, and an optional inference snapshot."""
    rank_offsets = gather_rank_log_offsets(rank_log.position(), context)
    metrics_position = coordinated_call(
        lambda: metrics_offset(output, context), "aggregate metrics offset", context
    )
    if monitor is not None and monitor.progress is not None:
        monitor.progress.phase("checkpoint", global_step, epoch + 1)
    save_training_state(
        output / "training_state.pt",
        training_state(
            identity,
            context.world_size,
            epoch + 1,
            global_step,
            base_model,
            optimizer,
            scheduler,
            history,
            rank_offsets,
            metrics_position,
            monitor.offsets() if monitor is not None else None,
        ),
        context,
    )
    if retention == "all_epochs":
        latest_index = publish_epoch_snapshot(
            output,
            base_model,
            config,
            identity,
            normalization,
            data_provenance,
            profile_receipt,
            profile_hash,
            history,
            epoch,
            global_step,
            latest_index,
            context,
            "safe-epoch inference snapshot publication",
        )
    return latest_index


def finish_fit(
    *,
    output: Path,
    context: DistributedContext,
    monitor: ComplementMonitor | None,
    global_step: int,
    epochs: int,
    latest_index: dict[str, object] | None,
    base_model: torch.nn.Module,
    config: dict[str, Any],
    history: list[dict[str, object]],
    identity: dict[str, object],
    normalization: dict[str, list[float]],
    data_provenance: Mapping[str, object],
    profile_receipt: Path,
    profile_hash: str,
) -> None:
    """Write the fixed configured-final checkpoint and complete progress state."""
    if monitor is not None and monitor.progress is not None:
        monitor.progress.phase("checkpoint", global_step, epochs)
    coordinated_call(
        lambda: (
            write_final_outputs(
                output,
                base_model,
                config,
                history,
                identity,
                normalization,
                data_provenance,
                profile_receipt,
                profile_hash,
                global_step,
                str(latest_index["index_sha256"]) if latest_index is not None else None,
            )
            if context.is_primary
            else True
        ),
        "fixed-final checkpoint write",
        context,
    )
    barrier(context)
    if monitor is not None and monitor.progress is not None:
        monitor.progress.phase("completed", global_step, epochs)
