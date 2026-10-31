from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch.distributed as dist
from torch import nn

from wnn.training.sequence_distributed import DistributedContext, coordinated_call
from wnn.training.sequence_fit_utils import SequenceDataset
from wnn.training.sequence_monitor import SequenceMonitor
from wnn.training.sequence_monitor_checkpoints import write_checkpoint_pointers
from wnn.training.sequence_provenance import canonical_sha256


def evaluate_full_valid(
    monitor: SequenceMonitor | None,
    model: nn.Module,
    train_eval: SequenceDataset | None,
    valid: SequenceDataset | None,
    fit_config: Mapping[str, object],
    monitor_config: Mapping[str, object],
    context: DistributedContext,
    epoch: int,
    global_step: int,
) -> dict[str, int | float]:
    """Run complete report cohorts on rank zero and share their finite metrics."""
    evaluation_holder: list[dict[str, int | float] | None] = [None]

    def evaluate() -> bool:
        """Evaluate once on the primary rank while coordinating failures."""
        if context.is_primary:
            if monitor is None:
                raise RuntimeError("rank zero monitor is missing")
            evaluation_holder[0] = monitor.evaluate_epoch(
                model,
                train_eval,
                valid,
                int(fit_config["chunk_rows"]),
                int(monitor_config.get("eval_batch_size", fit_config["batch_size"])),
                str(fit_config.get("precision_mode", "fp32")),
                context,
                epoch,
                global_step,
            )
        return True

    coordinated_call(evaluate, "full train-eval and official-valid evaluation", context)
    if context.process_group:
        dist.broadcast_object_list(evaluation_holder, src=0, device=context.device)
    evaluation = evaluation_holder[0]
    if not isinstance(evaluation, dict):
        raise RuntimeError("rank zero did not provide full-validation metrics")
    return evaluation


def publish_checkpoint_status(
    output: Path,
    identity: Mapping[str, object],
    status: str,
    global_step: int,
    completed_epochs: int,
    history: list[dict[str, object]],
    epoch_index: Mapping[str, object] | None,
    early_policy: Mapping[str, object] | None,
    monitor: SequenceMonitor | None,
    context: DistributedContext,
    event_context: dict[str, object],
    status_label: str,
) -> None:
    """Publish a checkpoint pointer and status event after the validation boundary."""

    def publish() -> bool:
        """Write shared status on rank zero after checking its safe snapshot index."""
        if context.is_primary:
            if monitor is None:
                raise RuntimeError("rank zero monitor was not initialized")
            payload = dict(event_context)
            if early_policy is not None:
                if epoch_index is None:
                    raise RuntimeError("early-stop status requires retained epoch snapshots")
                pointers = write_checkpoint_pointers(
                    output,
                    canonical_sha256(identity),
                    status,
                    history,
                    epoch_index,
                    early_policy,
                )
                payload.update(
                    {
                        "best_checkpoint": pointers["best"],
                        "last_checkpoint": pointers["last"],
                        "early_stopping": pointers["early_stopping"],
                    }
                )
            monitor.status(status, global_step, completed_epochs, payload)
        return True

    coordinated_call(publish, status_label, context)
