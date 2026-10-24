"""Read and validate an immutable epoch-boundary training snapshot."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

RESUME_SCHEMA = "wnn.sequence-resume.v1"


def _counter(value: object, name: str) -> int:
    """Return a nonnegative exact integer checkpoint counter."""
    if type(value) is not int or value < 0:
        raise ValueError(f"checkpoint {name} must be a nonnegative integer")
    return value


def _stat(path: Path) -> dict[str, int]:
    info = path.stat()
    return {"inode": info.st_ino, "size_bytes": info.st_size, "mtime_ns": info.st_mtime_ns}


def read_snapshot(
    path: Path,
    identity: Mapping[str, Any],
    *,
    config_sha256: str,
    source_sha256: str,
    split_sha256: str,
    epochs: int,
) -> dict[str, Any]:
    """Load optimizer-bearing checkpoint on CPU, return only finite model weights."""
    import torch

    snapshot = path.resolve(strict=True)
    if not snapshot.is_file() or snapshot.name == "training_state.pt":
        raise ValueError("checkpoint must be a captured regular snapshot, not the live file")
    before = _stat(snapshot)
    state = torch.load(snapshot, map_location="cpu", mmap=True, weights_only=False)
    if not isinstance(state, Mapping) or state.get("checkpoint_schema") != RESUME_SCHEMA:
        raise ValueError("unsupported sequence resume checkpoint schema")
    if state.get("contract") != dict(identity):
        raise ValueError("checkpoint contract differs from the approved profile identity")
    if identity.get("config_sha256") != config_sha256:
        raise ValueError("approved profile config hash differs from the loaded config")
    if identity.get("source_sha256") != source_sha256:
        raise ValueError("approved profile source hash differs from the pinned source")
    data = identity.get("data")
    if not isinstance(data, Mapping) or data.get("split_sha256") != split_sha256:
        raise ValueError("approved profile split differs from the selected fold")
    world_size = _counter(state.get("world_size"), "world_size")
    if world_size != identity.get("world_size"):
        raise ValueError("checkpoint world size differs from its approved profile")
    completed = _counter(state.get("next_epoch"), "next_epoch")
    global_step = _counter(state.get("global_step"), "global_step")
    if not 1 <= completed <= epochs:
        raise ValueError("snapshot is not a completed epoch in the configured schedule")
    history = state.get("history")
    if not isinstance(history, list) or len(history) != completed:
        raise ValueError("checkpoint history does not match its completed epoch count")
    updates = 0
    for index, row in enumerate(history):
        if (
            not isinstance(row, Mapping)
            or row.get("epoch") != index
            or row.get("complete") is not True
        ):
            raise ValueError("checkpoint history contains an incomplete or misindexed epoch")
        updates += _counter(row.get("updates"), "history updates")
    if updates != global_step:
        raise ValueError("checkpoint global step differs from its complete epoch history")
    raw_model = state.get("model")
    if not isinstance(raw_model, Mapping) or not raw_model:
        raise ValueError("checkpoint is missing model weights")
    model: dict[str, torch.Tensor] = {}
    for name, tensor in raw_model.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise TypeError("checkpoint model state must map names to tensors")
        if tensor.dtype != torch.float32 or not torch.isfinite(tensor).all().item():
            raise ValueError(f"checkpoint model tensor is non-FP32 or nonfinite: {name}")
        model[name] = tensor
    after = _stat(snapshot)
    if before != after:
        raise RuntimeError("captured checkpoint changed while it was being read")
    del state
    return {
        "model": model,
        "completed_epochs": completed,
        "checkpoint_epoch": completed - 1,
        "global_step": global_step,
        "history": history,
        "snapshot": before,
        "world_size": world_size,
    }


def load_model_state(model: Any, weights: Mapping[str, Any]) -> None:
    """Check tensor names and shapes, then copy the snapshot into a fresh model."""
    reference = model.state_dict()
    if set(weights) != set(reference):
        raise ValueError("checkpoint model tensor names differ from the canonical model")
    for name, expected in reference.items():
        actual = weights[name]
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            raise ValueError(f"checkpoint tensor shape or dtype differs: {name}")
    model.load_state_dict(weights, strict=True)
