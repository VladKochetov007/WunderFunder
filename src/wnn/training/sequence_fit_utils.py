from __future__ import annotations

import json
import math
import os
import random
import stat
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Protocol, TypedDict

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate

from wnn.training.sequence_provenance import json_value

SEQUENCE_ROWS = 20_000
FEATURE_COUNT = 112
TARGET_COUNT = 2


class SequenceItem(TypedDict):
    """One canonical sequence record returned by the fit dataset."""

    x: torch.Tensor
    y: torch.Tensor
    mask: torch.Tensor
    teacher: torch.Tensor
    hints: torch.Tensor


class SequenceDataset(Protocol):
    """Minimal complete-sequence dataset boundary used by the torchrun loop."""

    def __len__(self) -> int:
        """Return the number of canonical fit sequences."""
        ...

    def __getitem__(self, index: int) -> SequenceItem:
        """Return one record, including the all-zero masked record for index -1."""
        ...


def seed_worker(seed: int, rank: int) -> None:
    """Seed construction identically, then give each rank independent stochastic updates."""
    random.seed(seed + rank)
    np.random.seed((seed + rank) % (2**32))  # noqa: NPY002 - seed process-global NumPy users
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed(seed + rank)


def collate_sequences(dataset: SequenceDataset, indices: Sequence[int]) -> SequenceItem:
    """Collate complete records directly without creating DataLoader worker processes."""
    if not indices:
        raise ValueError("cannot collate an empty sequence batch")
    records = [dataset[index] for index in indices]
    batch = default_collate(records)
    if not isinstance(batch, dict):
        raise TypeError("sequence dataset records must collate to a mapping")
    return batch


def move_and_validate_batch(
    batch: SequenceItem, device: torch.device, hint_dim: int, feature_count: int = FEATURE_COUNT
) -> SequenceItem:
    """Move one batch and enforce configured feature, target, mask, and hint shapes."""
    if type(feature_count) is not int or feature_count < 1:
        raise ValueError("feature_count must be a positive integer")
    moved = {key: value.to(device=device, non_blocking=True) for key, value in batch.items()}
    x, target, mask, teacher, hints = (
        moved["x"],
        moved["y"],
        moved["mask"],
        moved["teacher"],
        moved["hints"],
    )
    expected_rows = (x.shape[0], SEQUENCE_ROWS)
    if x.shape != (*expected_rows, feature_count):
        raise ValueError(f"expected x[B,20000,{feature_count}], received {tuple(x.shape)}")
    if target.shape != (*expected_rows, TARGET_COUNT) or teacher.shape != target.shape:
        raise ValueError("label and teacher batches must have shape [B,20000,2]")
    if mask.shape != expected_rows or mask.dtype != torch.bool:
        raise ValueError("sequence mask must be boolean with shape [B,20000]")
    if hints.shape != (*expected_rows, hint_dim):
        raise ValueError(f"hint target must have shape [B,20000,{hint_dim}]")
    for name, value in (("x", x), ("y", target), ("teacher", teacher), ("hints", hints)):
        if value.dtype != torch.float32:
            raise TypeError(f"{name} must use strict FP32, received {value.dtype}")
    return moved


def make_optimizer(
    model: torch.nn.Module, config: Mapping[str, object], total_updates: int
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    """Create the configured optimizer and update-indexed schedule."""
    from wnn.training.sequence_optimizer_factory import make_optimizer as factory

    return factory(model, config, total_updates)


def sequence_lr_multiplier(
    schedule: str, total: int, warmup_fraction: float, min_lr_ratio: float
) -> Callable[[int], float]:
    """Create a legacy or endpoint-aligned warmup-cosine multiplier.

    Args:
        schedule: ``legacy_warmup_cosine`` to preserve prior steps or ``warmup_cosine_v2``.
        total: Number of optimizer updates in the complete fit.
        warmup_fraction: Fraction of updates reserved for linear warmup.
        min_lr_ratio: Final fraction of the shared peak learning rate for v2.

    Returns:
        A callable indexed by ``LambdaLR`` optimizer step number.

    Raises:
        ValueError: If the schedule name, duration, or floor is invalid.
    """
    if total < 1 or not 0 <= warmup_fraction < 1 or not 0 <= min_lr_ratio <= 1:
        raise ValueError("invalid sequence schedule duration, warmup, or floor")
    if schedule == "legacy_warmup_cosine":
        if min_lr_ratio != 0:
            raise ValueError("legacy_warmup_cosine requires min_lr_ratio=0")
        warmup_updates = min(total, math.ceil(total * warmup_fraction))

        def legacy_multiplier(step: int) -> float:
            """Preserve the previous warmup/cosine indexing exactly."""
            if warmup_updates > 0 and step < warmup_updates:
                return (step + 1) / warmup_updates
            cosine_updates = max(1, total - warmup_updates)
            progress = min(1.0, max(0.0, (step - warmup_updates + 1) / cosine_updates))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return legacy_multiplier
    if schedule != "warmup_cosine_v2":
        raise ValueError(f"unknown sequence learning-rate schedule: {schedule}")
    if total < 2:
        raise ValueError("warmup_cosine_v2 requires at least two optimizer updates")
    warmup_updates = math.ceil(total * warmup_fraction)
    if warmup_updates >= total - 1:
        raise ValueError("warmup_cosine_v2 requires at least one post-warmup endpoint interval")

    def warmup_cosine_v2_multiplier(step: int) -> float:
        """Reach the configured floor on the final actual update."""
        if step < warmup_updates:
            return (step + 1) / warmup_updates if warmup_updates else 1.0
        denominator = max(1, total - 1 - warmup_updates)
        progress = min(1.0, max(0.0, (step - warmup_updates) / denominator))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return warmup_cosine_v2_multiplier


def total_updates(config: Mapping[str, object], sequence_count: int, world_size: int) -> int:
    """Count fixed schedule steps with the same padded per-rank batching as training."""
    local_sequences = math.ceil(sequence_count / world_size)
    batches = math.ceil(local_sequences / int(config["batch_size"]))
    chunks_per_sequence = math.ceil(SEQUENCE_ROWS / int(config["chunk_rows"]))
    windows = math.ceil(chunks_per_sequence / int(config["accumulate_chunks"]))
    return int(config["epochs"]) * batches * windows


class RankJsonlLog:
    """Append rank-local progress records beside the rank-zero aggregate reporter."""

    def __init__(self, path: Path, *, resume: bool) -> None:
        """Open a new rank log or append only when resuming the same run directory."""
        self._path = path
        self._stream = path.open("a+b" if resume else "x+b")

    def write(self, record: Mapping[str, object]) -> None:
        """Write one normalized finite JSON line and flush it for failure diagnosis."""
        value = json_value(record)
        line = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
        self._stream.write((line + "\n").encode("utf-8"))
        self._stream.flush()

    def position(self) -> int:
        """Return the flushed byte offset used by the latest safe checkpoint."""
        self._stream.flush()
        return self._stream.tell()

    def truncate(self, offset: int, *, rank: int, global_step: int) -> int:
        """Preserve abandoned attempt bytes, then restore the safe log boundary."""
        return truncate_with_evidence(self._path, offset, rank=rank, global_step=global_step)

    def close(self) -> None:
        """Flush and close the rank-owned progress file."""
        self._stream.close()


def truncate_with_evidence(path: Path, offset: int, *, rank: int, global_step: int) -> int:
    """Preserve post-checkpoint bytes before truncating an interrupted attempt log."""
    if offset < 0 or global_step < 0 or rank < 0:
        raise ValueError("resume offsets, rank, and global step must be nonnegative")
    if not path.exists():
        if offset == 0:
            return 0
        raise ValueError(f"log is missing its checkpointed safe offset: {path}")
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError(f"resume log must be a regular file: {path}")
    with path.open("r+b") as stream:
        stream.seek(0, 2)
        size = stream.tell()
        if size < offset:
            raise ValueError(f"log is shorter than its checkpointed safe offset: {path}")
        if size == offset:
            return 0
        stream.seek(offset)
        discarded = stream.read()
        evidence = path.with_name(f"{path.name}.replay_tail-{time.time_ns()}.jsonl")
        with evidence.open("xb") as saved:
            saved.write(discarded)
            saved.flush()
            os.fsync(saved.fileno())
        audit = path.parent / f"resume_replay_rank{rank}.jsonl"
        record = {
            "rank": rank,
            "safe_global_step": global_step,
            "source_log": path.name,
            "evidence_file": evidence.name,
            "discarded_bytes": len(discarded),
        }
        with audit.open("ab") as log:
            log.write((json.dumps(record, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"))
            log.flush()
            os.fsync(log.fileno())
        stream.truncate(offset)
        stream.flush()
        os.fsync(stream.fileno())
    return len(discarded)


def optimizer_states_are_finite_fp32(optimizer: torch.optim.Optimizer) -> bool:
    """Require every floating optimizer slot to remain finite strict-FP32 state."""
    by_device: dict[torch.device, list[torch.Tensor]] = {}
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                if value.dtype != torch.float32:
                    return False
                by_device.setdefault(value.device, []).append(torch.isfinite(value).all())
    return all(bool(torch.stack(flags).all()) for flags in by_device.values())
