from __future__ import annotations

import json
import os
import random
import tempfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from wnn.training.sequence_distributed import DistributedContext, all_ranks_true, barrier
from wnn.training.sequence_provenance import json_value


def capture_rng_state(context: DistributedContext) -> dict[str, object]:
    """Capture rank-local Python, NumPy, CPU Torch, and CUDA RNG at an epoch boundary."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),  # noqa: NPY002 - serialize the process-global legacy RNG
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(context.device),
    }


def restore_rng_state(state: Mapping[str, object], context: DistributedContext) -> None:
    """Restore one rank's stochastic state after loading model and optimizer tensors."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])  # noqa: NPY002 - restore the serialized global RNG
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state(state["torch_cuda"], device=context.device)


def _atomic_torch_save(value: object, path: Path) -> None:
    """Durably replace one safe-boundary checkpoint after serialization succeeds."""
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _gather_rng_states(context: DistributedContext) -> list[dict[str, object]]:
    local = capture_rng_state(context)
    if not context.process_group:
        return [local]
    gathered: list[dict[str, object] | None] = [None] * context.world_size
    dist.all_gather_object(gathered, local)
    if any(item is None for item in gathered):
        raise RuntimeError("a rank did not contribute its epoch-boundary RNG state")
    return [item for item in gathered if item is not None]


def gather_rank_log_offsets(offset: int, context: DistributedContext) -> list[int]:
    """Collect flushed per-rank JSONL positions for one resumable epoch boundary."""
    if offset < 0:
        raise ValueError("rank log offset must be nonnegative")
    if not context.process_group:
        return [offset]
    gathered: list[int | None] = [None] * context.world_size
    dist.all_gather_object(gathered, offset)
    if any(value is None or value < 0 for value in gathered):
        raise RuntimeError("a rank did not contribute a valid safe log offset")
    return [int(value) for value in gathered if value is not None]


def save_training_state(path: Path, state: dict[str, object], context: DistributedContext) -> None:
    """Atomically save optimizer/model state and each rank's RNG at a safe boundary."""
    rank_rng_states = _gather_rng_states(context)
    result: list[dict[str, str | bool]] = [{"ok": True, "error": ""}]
    if context.is_primary:
        try:
            _atomic_torch_save({**state, "rank_rng_states": rank_rng_states}, path)
        except Exception as error:
            result[0] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    if context.process_group:
        dist.broadcast_object_list(result, src=0, device=context.device)
    if not result[0]["ok"]:
        raise OSError(str(result[0]["error"]))
    barrier(context)


def load_training_state(
    path: Path, expected_identity: dict[str, object], context: DistributedContext
) -> dict[str, object]:
    """Load a safe checkpoint and reject config/data/source/runtime/world-size drift."""
    state: dict[str, object] | None = None
    error = ""
    try:
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(loaded, dict):
            raise TypeError("resume checkpoint must contain an object")
        if loaded.get("checkpoint_schema") != "wnn.sequence-resume.v1":
            raise ValueError("unsupported resume checkpoint schema")
        if loaded.get("contract") != expected_identity:
            raise ValueError("resume config/data/source/runtime identity does not match")
        if not _counter(loaded.get("world_size")) or loaded["world_size"] != context.world_size:
            raise ValueError("resume requires the exact original torchrun world size")
        for field in ("next_epoch", "global_step", "metrics_offset"):
            if not _counter(loaded.get(field)):
                raise ValueError(f"resume checkpoint {field} must be a nonnegative integer")
        if not isinstance(loaded.get("model"), dict):
            raise ValueError("resume checkpoint is missing model state")
        if not isinstance(loaded.get("optimizer"), dict):
            raise ValueError("resume checkpoint is missing optimizer state")
        if not isinstance(loaded.get("scheduler"), dict):
            raise ValueError("resume checkpoint is missing scheduler state")
        if not isinstance(loaded.get("history"), list):
            raise ValueError("resume history must be a list")
        history = loaded["history"]
        next_epoch = int(loaded["next_epoch"])
        global_step = int(loaded["global_step"])
        if len(history) != next_epoch or not all(isinstance(row, dict) for row in history):
            raise ValueError("resume history does not match its completed epoch counter")
        if any(
            not _counter(row.get("epoch"))
            or row["epoch"] != index
            or not _counter(row.get("updates"))
            or row.get("complete") is not True
            for index, row in enumerate(history)
        ):
            raise ValueError("resume history contains malformed epoch records")
        if sum(int(row["updates"]) for row in history) != global_step:
            raise ValueError("resume global step does not match completed epoch updates")
        ranks = loaded.get("rank_rng_states")
        offsets = loaded.get("rank_log_offsets")
        if not isinstance(ranks, list) or len(ranks) != context.world_size:
            raise ValueError("resume checkpoint is missing rank-local RNG states")
        if not all(
            isinstance(item, Mapping)
            and set(item) >= {"python", "numpy", "torch_cpu", "torch_cuda"}
            for item in ranks
        ):
            raise ValueError("resume checkpoint contains malformed rank-local RNG state")
        if (
            not isinstance(offsets, list)
            or len(offsets) != context.world_size
            or not all(_counter(value) for value in offsets)
        ):
            raise ValueError("resume checkpoint is missing per-rank safe log offsets")
        state = loaded
    except Exception as exception:
        error = f"{type(exception).__name__}: {exception}"
    if not all_ranks_true(state is not None, context):
        messages: list[str | None] = [None] * context.world_size
        if context.process_group:
            dist.all_gather_object(messages, error)
        else:
            messages[0] = error
        detail = "; ".join(message for message in messages if message)
        raise RuntimeError(f"resume checkpoint failed on at least one rank: {detail}")
    if state is None:
        raise RuntimeError("resume checkpoint did not load")
    return state


def _counter(value: object) -> bool:
    """Accept only nonnegative integer checkpoint counters, excluding booleans."""
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def atomic_write_json(value: object, path: Path) -> None:
    """Atomically write one JSON receipt without replacing an existing artifact."""
    if path.exists():
        raise FileExistsError(f"refusing to replace existing receipt: {path}")
    encoded = (
        json.dumps(json_value(value), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def save_final_model(
    path: Path,
    model_state: dict[str, torch.Tensor],
    config: dict[str, object],
    epoch: int,
    completed_epochs: int,
    data_provenance: Mapping[str, object],
    provenance: dict[str, object],
) -> None:
    """Write the fixed-final inference envelope once after every configured epoch."""
    if completed_epochs != int(config["epochs"]):
        raise ValueError("final model requires all configured epochs to be complete")
    if path.exists():
        raise FileExistsError(f"refusing to replace existing final model: {path}")
    cpu_state = {name: tensor.detach().cpu() for name, tensor in model_state.items()}
    payload = {
        "checkpoint_schema": "wnn.sequence-fit.v1",
        "model": cpu_state,
        "config": config,
        "epoch": epoch,
        "completed_epochs": completed_epochs,
        "data": json_value(data_provenance),
        "provenance": provenance,
    }
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
