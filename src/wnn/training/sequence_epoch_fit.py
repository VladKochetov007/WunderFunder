from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from wnn.training.sequence_distributed import DistributedContext, coordinated_call
from wnn.training.sequence_epoch_checkpoints import (
    preflight_epoch_storage,
    save_epoch_checkpoint,
)
from wnn.training.sequence_epoch_index import (
    publish_epoch_index,
    validate_epoch_index_state,
)


def reconcile_epoch_snapshots(
    output: Path,
    base_model: torch.nn.Module,
    config: dict[str, Any],
    identity: dict[str, object],
    normalization: dict[str, list[float]],
    data_provenance: Mapping[str, object],
    profile_receipt: Path,
    profile_hash: str,
    start_epoch: int,
    global_step: int,
    history: list[dict[str, object]],
    context: DistributedContext,
) -> dict[str, object] | None:
    """Validate the safe index prefix and reconstruct only a missing final boundary."""
    latest: list[dict[str, object] | None] = [None]

    def reconcile() -> bool:
        """Reconcile safe epoch metadata on the primary worker."""
        if context.is_primary:
            latest[0] = validate_epoch_index_state(
                output, identity, start_epoch, global_step, history
            )
        return True

    coordinated_call(reconcile, "epoch-index resume validation", context)
    previous_count = int(latest[0]["completed_epochs"]) if latest[0] is not None else 0
    frontier: list[int | None] = [previous_count if context.is_primary else None]
    if context.process_group:
        dist.broadcast_object_list(frontier, src=0, device=context.device)
    if not isinstance(frontier[0], int):
        raise RuntimeError("rank zero did not broadcast the retained epoch frontier")
    previous_count = frontier[0]
    remaining = int(config["epochs"]) - previous_count

    def preflight() -> bool:
        """Check remaining snapshot storage before advancing training."""
        if context.is_primary:
            preflight_epoch_storage(output, base_model, remaining)
        return True

    coordinated_call(preflight, "all-epoch checkpoint storage preflight", context)
    if previous_count < start_epoch:
        latest[0] = publish_epoch_snapshot(
            output,
            base_model,
            config,
            identity,
            normalization,
            data_provenance,
            profile_receipt,
            profile_hash,
            history,
            start_epoch - 1,
            global_step,
            latest[0],
            context,
            "resume missing epoch inference snapshot recovery",
        )
    return latest[0]


def publish_epoch_snapshot(
    output: Path,
    base_model: torch.nn.Module,
    config: dict[str, Any],
    identity: dict[str, object],
    normalization: dict[str, list[float]],
    data_provenance: Mapping[str, object],
    profile_receipt: Path,
    profile_hash: str,
    history: list[dict[str, object]],
    epoch: int,
    global_step: int,
    previous_index: dict[str, object] | None,
    context: DistributedContext,
    label: str,
) -> dict[str, object] | None:
    """Publish a model snapshot and its append-only index from rank zero."""
    latest: list[dict[str, object] | None] = [previous_index]

    def publish() -> bool:
        """Publish weights before making their safe index visible."""
        if context.is_primary:
            completed_epochs = epoch + 1
            entry = save_epoch_checkpoint(
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
            )
            latest[0] = publish_epoch_index(
                output,
                identity,
                epoch,
                completed_epochs,
                global_step,
                entry,
                latest[0],
            )
        return True

    coordinated_call(publish, label, context)
    return latest[0]
