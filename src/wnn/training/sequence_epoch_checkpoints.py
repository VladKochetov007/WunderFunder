from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path

import torch

from wnn.training.sequence_epoch_index_io import file_sha256, fsync_directory
from wnn.training.sequence_precision import precision_identity
from wnn.training.sequence_provenance import json_value

_SAFETY_BYTES = 64 << 20
_SAFETY_FRACTION = 0.10


def preflight_epoch_storage(output: Path, model: torch.nn.Module, snapshots: int) -> int:
    """Require space for remaining immutable weights snapshots before fitting."""
    if snapshots < 0:
        raise ValueError("remaining snapshot count must be nonnegative")
    if snapshots == 0:
        return 0
    if not output.is_dir():
        raise FileNotFoundError(f"fit output directory does not exist: {output}")
    state_bytes = 0
    for name, tensor in model.state_dict().items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"model state entry {name!r} is not a tensor")
        state_bytes += tensor.numel() * tensor.element_size()
    required = int(state_bytes * snapshots * (1.0 + _SAFETY_FRACTION)) + _SAFETY_BYTES
    free = shutil.disk_usage(output).free
    if free < required:
        raise OSError(
            f"all-epoch checkpoint retention needs at least {required} free bytes; found {free}"
        )
    return required


def save_epoch_checkpoint(
    output: Path,
    model: torch.nn.Module,
    config: Mapping[str, object],
    identity: Mapping[str, object],
    normalization: Mapping[str, list[float]],
    data_provenance: Mapping[str, object],
    profile_receipt: Path,
    profile_hash: str,
    history: list[dict[str, object]],
    epoch: int,
    global_step: int,
) -> dict[str, object]:
    """Write one immutable inference-only CPU snapshot for a safe epoch boundary."""
    if epoch < 0 or global_step < 0 or epoch >= int(config["epochs"]):
        raise ValueError("epoch snapshot counters are outside the configured fit")
    completed_epochs = epoch + 1
    if len(history) != completed_epochs:
        raise ValueError("epoch snapshot history does not reach the safe epoch boundary")
    checkpoint_dir = output / "epoch_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if checkpoint_dir.is_symlink() or not checkpoint_dir.is_dir():
        raise ValueError("epoch checkpoint path must be a real directory")
    model_config = config.get("model")
    if not isinstance(model_config, Mapping):
        raise TypeError("sequence recipe is missing its model settings")
    provenance: dict[str, object] = {
        "schema": "wnn.sequence-provenance.v1",
        "identity": dict(identity),
        "normalization": json_value(normalization),
        "profile_receipt": {"path": str(profile_receipt), "sha256": profile_hash},
        "history": json_value(history),
        "checkpoint_policy": "all-safe-epochs-inference-only",
        "completed_epochs": completed_epochs,
        "global_updates": global_step,
        "precision": precision_identity(
            str(config.get("precision_mode", "fp32")),
            str(model_config.get("attention_backend", "math")),
        ),
    }
    payload = {
        "checkpoint_schema": "wnn.sequence-epoch.v1",
        "model": _cpu_model_state(model),
        "config": json_value(config),
        "epoch": epoch,
        "completed_epochs": completed_epochs,
        "global_step": global_step,
        "identity": json_value(identity),
        "data": json_value(data_provenance),
        "provenance": provenance,
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".epoch-{completed_epochs:06d}.", dir=checkpoint_dir
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        checkpoint_sha = file_sha256(temporary)
        filename = f"epoch-{completed_epochs:06d}-{checkpoint_sha}.pt"
        path = checkpoint_dir / filename
        if path.exists():
            if path.is_symlink() or file_sha256(path) != checkpoint_sha:
                raise FileExistsError(f"conflicting immutable epoch checkpoint: {path}")
            temporary.unlink()
        else:
            os.link(temporary, path)
            fsync_directory(checkpoint_dir)
        return {
            "epoch": epoch,
            "completed_epochs": completed_epochs,
            "global_step": global_step,
            "checkpoint_path": f"epoch_checkpoints/{filename}",
            "checkpoint_sha256": checkpoint_sha,
        }
    finally:
        temporary.unlink(missing_ok=True)


def _cpu_model_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Clone inference weights to CPU so later optimizer steps cannot mutate a snapshot."""
    result: dict[str, torch.Tensor] = {}
    for name, tensor in model.state_dict().items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"model state entry {name!r} is not a tensor")
        if tensor.is_floating_point() and tensor.dtype != torch.float32:
            raise TypeError(f"model state entry {name!r} is not an FP32 master tensor")
        copied = tensor.detach().to(device="cpu").clone()
        if copied.is_floating_point() and not bool(torch.isfinite(copied).all()):
            raise FloatingPointError(f"model state entry {name!r} is not finite")
        result[name] = copied
    return result
