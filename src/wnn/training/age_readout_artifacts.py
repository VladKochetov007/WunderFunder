"""Create-only checkpoint, prediction and JSON artifact publication."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .age_readout_cohort import ROWS_PER_SEQUENCE
from .age_readout_fit import ARMS


class PredictionSink:
    """Write one ordered prediction array per arm using a private disk mapping."""

    def __init__(self, output_dirs: dict[str, Path], sequence_count: int) -> None:
        self.output_dirs = output_dirs
        self.sequence_count = sequence_count
        self.temporary: dict[str, Path] = {}
        self.arrays: dict[str, np.memmap] = {}
        self.seen = {arm: np.zeros(sequence_count, dtype=np.bool_) for arm in ARMS}
        for arm in ARMS:
            output_dirs[arm].mkdir(parents=True, exist_ok=True)
            destination = output_dirs[arm] / "prediction.npy"
            if destination.exists():
                raise FileExistsError(f"refusing to replace {destination}")
            temporary = output_dirs[arm] / f".prediction.{os.getpid()}.partial"
            if temporary.exists():
                raise FileExistsError(f"temporary prediction path already exists: {temporary}")
            self.temporary[arm] = temporary
            self.arrays[arm] = np.lib.format.open_memmap(
                temporary,
                mode="w+",
                dtype="<f4",
                shape=(sequence_count, ROWS_PER_SEQUENCE, 2),
            )

    def write(self, arm: str, index: int, values: np.ndarray) -> None:
        """Write one full-sequence prediction at its frozen cache position."""
        if arm not in self.arrays or index < 0 or index >= self.sequence_count:
            raise ValueError("prediction callback arm or sequence index is invalid")
        if values.shape != (ROWS_PER_SEQUENCE, 2) or not np.isfinite(values).all():
            raise ValueError("candidate prediction must be finite with shape (20000,2)")
        if self.seen[arm][index]:
            raise ValueError(f"duplicate prediction callback for arm {arm}, sequence {index}")
        self.arrays[arm][index] = values
        self.seen[arm][index] = True

    def publish(self) -> dict[str, dict[str, object]]:
        """Flush and create-only publish every candidate prediction array."""
        incomplete = {
            arm: np.flatnonzero(~seen).tolist()
            for arm, seen in self.seen.items()
            if not bool(seen.all())
        }
        if incomplete:
            raise RuntimeError(f"refusing incomplete candidate prediction coverage: {incomplete}")
        result: dict[str, dict[str, object]] = {}
        for arm in ARMS:
            self.arrays[arm].flush()
            temporary = self.temporary[arm]
            _fsync_file(temporary)
            destination = self.output_dirs[arm] / "prediction.npy"
            os.link(temporary, destination)
            temporary.unlink()
            result[arm] = {
                "path": str(destination),
                "sha256": file_sha256(destination),
                "shape": [self.sequence_count, ROWS_PER_SEQUENCE, 2],
                "dtype": "<f4",
            }
        self.arrays.clear()
        return result


def publish_head_checkpoints(
    models: dict[str, torch.nn.Module],
    output_dirs: dict[str, Path],
    *,
    identity: dict[str, object],
    parent_mean: np.ndarray,
    parent_std: np.ndarray,
) -> dict[str, dict[str, object]]:
    """Publish fixed-epoch heads to each frozen recipe cell output."""
    published: dict[str, dict[str, object]] = {}
    for arm in ARMS:
        destination = output_dirs[arm] / "model_ep2.pt"
        payload: dict[str, Any] = {
            "schema": "wnn.age-readout-head.v1",
            "identity": identity,
            "arm": arm,
            "fixed_epoch": 2,
            "architecture": {"input_width": 323, "hidden_width": 64, "activation": "tanh"},
            "residual_scale": 0.1,
            "parent_mean": np.asarray(parent_mean, dtype="<f4").tolist(),
            "parent_std": np.asarray(parent_std, dtype="<f4").tolist(),
            "age_encoding": "log1p(min(age,10000))/log1p(10000)",
            "state_dict": {
                key: value.detach().cpu().clone() for key, value in models[arm].state_dict().items()
            },
        }
        publish_torch(payload, destination)
        published[arm] = {"path": str(destination), "sha256": file_sha256(destination)}
    return published


def publish_json(value: dict[str, object], destination: Path) -> str:
    """Publish a stable JSON receipt without replacing an existing file."""
    payload = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode("utf-8")
    _publish_bytes(payload, destination)
    return file_sha256(destination)


def publish_torch(value: dict[str, Any], destination: Path) -> None:
    """Publish a serialized artifact via a same-directory create-only hard link."""
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.partial")
    with temporary.open("xb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def file_sha256(path: Path) -> str:
    """Hash a file incrementally without retaining its payload in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _publish_bytes(payload: bytes, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.partial")
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())
