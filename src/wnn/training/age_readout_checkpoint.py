"""Strict epoch-bundle checkpoint restore and create-only persistence."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


def restore_panel_checkpoint(
    path: Path,
    identity: dict[str, object],
    models: dict[str, nn.Module],
    optimizers: dict[str, torch.optim.Optimizer],
    device: torch.device,
    arms: Sequence[str],
    epoch_count: int,
) -> tuple[int, list[dict[str, object]]]:
    """Restore all arm/optimizer clocks only from an identical frozen identity."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("schema") != "wnn.age-readout-checkpoint.v1":
        raise ValueError("resume checkpoint schema is unsupported")
    if checkpoint.get("identity") != identity:
        raise ValueError("resume checkpoint identity differs from frozen screen inputs")
    cpu_rng = checkpoint.get("torch_rng_state")
    if isinstance(cpu_rng, Tensor):
        torch.set_rng_state(cpu_rng.cpu())
    cuda_rng = checkpoint.get("cuda_rng_state")
    if device.type == "cuda" and isinstance(cuda_rng, Tensor):
        torch.cuda.set_rng_state(cuda_rng.cpu(), device=device)
    epoch = int(checkpoint["epoch"])
    if epoch < 0 or epoch >= epoch_count:
        raise ValueError("resume checkpoint epoch is outside the fixed schedule")
    for arm in arms:
        models[arm].load_state_dict(checkpoint["models"][arm])
        optimizers[arm].load_state_dict(checkpoint["optimizers"][arm])
    return epoch + 1, list(checkpoint["records"])


def save_panel_checkpoint(
    path: Path,
    identity: dict[str, object],
    epoch: int,
    models: dict[str, nn.Module],
    optimizers: dict[str, torch.optim.Optimizer],
    records: list[dict[str, object]],
) -> None:
    """Save resumable arm and optimizer state without replacing an earlier epoch."""
    device = next(iter(models.values())).parameters().__next__().device
    payload: dict[str, Any] = {
        "schema": "wnn.age-readout-checkpoint.v1",
        "identity": identity,
        "epoch": epoch,
        "models": {
            arm: {name: value.detach().cpu() for name, value in model.state_dict().items()}
            for arm, model in models.items()
        },
        "optimizers": {arm: optimizer.state_dict() for arm, optimizer in optimizers.items()},
        "records": records,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }
    publish_torch_create_only(payload, path)


def publish_torch_create_only(payload: dict[str, Any], destination: Path) -> None:
    """Publish a torch artifact with a same-directory hard link that cannot replace."""
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.partial")
    with temporary.open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
