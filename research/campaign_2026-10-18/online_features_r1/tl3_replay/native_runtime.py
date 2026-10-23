"""Small helpers for loading and streaming an exact packaged native callback."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
import time
import zipfile
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Protocol, cast

import numpy as np


class CallbackPoint(Protocol):
    """Feature-only input object consumed by the packaged callback."""

    seq_ix: int
    step_in_seq: int
    need_prediction: bool
    state: np.ndarray


class TL3Model(Protocol):
    """Runtime surface exported by the exact team ZIP."""

    member_count: int
    quant_backend: str

    def predict(self, data_point: CallbackPoint) -> np.ndarray | None:
        """Advance one causal row and optionally return its two predictions."""


def sha256_file(path: Path) -> str:
    """Hash a file with bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_safe(archive: zipfile.ZipFile, destination: Path) -> None:
    """Extract regular archive members after checking their relative paths."""
    for member in archive.infolist():
        name = PurePosixPath(member.filename)
        if name.is_absolute() or ".." in name.parts or "\\" in member.filename:
            raise ValueError(f"unsafe ZIP member path: {member.filename!r}")
        if member.is_dir():
            continue
        mode = member.external_attr >> 16
        if mode and (mode & 0o170000) not in (0, 0o100000):
            raise ValueError(f"non-regular ZIP member: {member.filename!r}")
        target = destination.joinpath(*name.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(member) as source, target.open("wb") as output:
            shutil.copyfileobj(source, output, length=1 << 20)


def load_callback(package: Path) -> tuple[ModuleType, TL3Model, Path]:
    """Import the exact package and construct its native ensemble."""
    spec = importlib.util.spec_from_file_location("team_tl3_solution", package / "solution.py")
    if spec is None or spec.loader is None:
        raise ImportError("ZIP does not contain an importable root solution.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    callback = cast(TL3Model, module.PredictionModel())
    native = sys.modules.get("wnn_cpp")
    if native is None:
        raise ImportError("solution.py did not load the packaged wnn_cpp extension")
    native_path = Path(native.__file__).resolve()
    if not native_path.is_relative_to(package.resolve()):
        raise RuntimeError(f"imported native extension escaped extracted package: {native_path}")
    return module, callback, native_path


def run_sequence(
    callback: TL3Model,
    point_factory: Callable[[int, int, bool, np.ndarray], CallbackPoint],
    sequence_id: int,
    features: np.ndarray,
    need: np.ndarray,
) -> tuple[np.ndarray, int, int, int]:
    """Run every row in one validated sequence, including warmup rows."""
    predictions = np.zeros((len(features), 2), dtype=np.float32)
    warmup_rows = 0
    predicted_rows = 0
    callback_wall_ns = 0
    for step, row in enumerate(features):
        point = point_factory(sequence_id, step, bool(need[step]), row)
        started = time.perf_counter_ns()
        value = callback.predict(point)
        callback_wall_ns += time.perf_counter_ns() - started
        if not need[step]:
            warmup_rows += 1
            if value is not None:
                raise ValueError(f"callback returned a prediction during warmup at step {step}")
            continue
        if value is None:
            raise ValueError(f"required prediction missing at sequence {sequence_id}, step {step}")
        result = np.asarray(value, dtype=np.float32)
        if result.shape != (2,) or not np.isfinite(result).all():
            raise ValueError(f"invalid prediction at sequence {sequence_id}, step {step}")
        predictions[step] = result
        predicted_rows += 1
    return predictions, warmup_rows, predicted_rows, callback_wall_ns


def write_json_create_only(path: Path, value: dict[str, object]) -> None:
    """Write deterministic JSON evidence without replacing prior output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        json.dump(value, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")
