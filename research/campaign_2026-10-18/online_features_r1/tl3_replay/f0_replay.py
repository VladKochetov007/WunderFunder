"""Replay validated F0 groups through one exact native package callback."""

from __future__ import annotations

import os
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from f0_source import F0Source
from native_runtime import extract_safe, load_callback, run_sequence, sha256_file


@dataclass(frozen=True)
class ReplayResult:
    """Measured callback and prediction-cache results for one replay scope."""

    rows: int
    warmup_rows: int
    predicted_rows: int
    callback_wall_ns: int
    elapsed_wall_seconds: float
    elapsed_cpu_seconds: float
    reset_check_ns: int
    reset_aba_equal: bool | None
    sequence_receipts: list[dict[str, int]]
    native_relative_path: str
    native_sha256: str
    native_level: str
    dispatch_level: str
    quant_backend: str
    member_count: int


def _read_features(table: Any, source: F0Source, sequence_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Validate and materialize only the current row group's input features."""
    from utils import validate_sequence

    actual_sequence_id, need = validate_sequence(table)
    if actual_sequence_id != sequence_id:
        raise ValueError("producer row-group sequence ID differs from split manifest")
    features = np.column_stack(
        [table[name].to_numpy() for name in source.feature_columns]  # type: ignore[index]
    ).astype(np.float32, copy=False)
    if features.shape != (source.sequence_length, len(source.feature_columns)):
        raise ValueError("producer row group has an unexpected feature shape")
    if not np.isfinite(features).all():
        raise ValueError("producer row group contains nonfinite feature state")
    return features, need


def replay_groups(
    source: F0Source,
    archive_path: Path,
    scratch: Path,
    predictions_path: Path,
    reset_check: bool,
    progress_interval: int = 32,
) -> ReplayResult:
    """Run all supplied complete groups and atomically write the aligned cache."""
    from utils import DataPoint

    started = time.perf_counter()
    cpu_started = time.process_time()
    callback_wall_ns = 0
    warmup_rows = 0
    predicted_rows = 0
    receipts: list[dict[str, int]] = []
    pending_path = predictions_path.with_name(f".{predictions_path.name}.pending.npy")
    if pending_path.exists() or predictions_path.exists():
        raise FileExistsError("refusing to overwrite an existing prediction cache")
    native_details: tuple[str, str, str, str, int] | None = None
    reset_check_ns = 0
    reset_aba_equal: bool | None = None
    try:
        with zipfile.ZipFile(archive_path) as archive:
            if archive.testzip() is not None:
                raise ValueError("TL3 archive CRC validation failed")
            from tempfile import TemporaryDirectory

            with TemporaryDirectory(prefix="tl3-replay-", dir=scratch) as temporary:
                package = Path(temporary)
                extract_safe(archive, package)
                module, callback, native_path = load_callback(package)
                output = np.lib.format.open_memmap(
                    pending_path,
                    mode="w+",
                    dtype=np.float32,
                    shape=(len(source.groups), source.sequence_length, 2),
                )
                first_group: tuple[int, int, np.ndarray, np.ndarray] | None = None
                for index, row_group in enumerate(source.groups):
                    table = source.parquet.read_row_group(
                        row_group, columns=list(source.columns), use_threads=False
                    )
                    sequence_id = source.sequence_by_group.get(row_group)
                    if sequence_id is None:
                        raise ValueError(f"row group {row_group} has no split sequence identity")
                    features, need = _read_features(table, source, sequence_id)
                    values, warm, predicted, elapsed_ns = run_sequence(
                        callback, DataPoint, sequence_id, features, need
                    )
                    callback_wall_ns += elapsed_ns
                    warmup_rows += warm
                    predicted_rows += predicted
                    output[index] = values
                    receipts.append(
                        {"row_group": row_group, "sequence_id": sequence_id, "rows": len(features)}
                    )
                    if index == 0 and reset_check:
                        first_group = (row_group, sequence_id, features.copy(), need.copy())
                    del table, features, need, values
                    if progress_interval > 0 and (index + 1) % progress_interval == 0:
                        print(
                            f"groups={index + 1}/{len(source.groups)} "
                            f"elapsed={time.perf_counter() - started:.1f}s",
                            flush=True,
                        )
                if reset_check:
                    if first_group is None or len(source.groups) < 2:
                        raise ValueError("A-B-A check requires two complete sequences")
                    row_group, sequence_id, features, need = first_group
                    reset_started = time.perf_counter_ns()
                    replayed, _, _, _ = run_sequence(
                        callback, DataPoint, sequence_id, features, need
                    )
                    reset_check_ns = time.perf_counter_ns() - reset_started
                    reset_aba_equal = bool(np.array_equal(replayed, output[0]))
                    if not reset_aba_equal:
                        raise ValueError(f"A-B-A reset replay differs for row group {row_group}")
                    del features, need, replayed
                output.flush()
                del output
                os.replace(pending_path, predictions_path)
                native_details = (
                    native_path.relative_to(package).as_posix(),
                    sha256_file(native_path),
                    native_path.parent.name,
                    str(module.LEVEL),
                    int(callback.member_count),
                )
                quant_backend = str(callback.quant_backend)
    except BaseException:
        pending_path.unlink(missing_ok=True)
        predictions_path.unlink(missing_ok=True)
        raise
    if native_details is None:
        raise RuntimeError("native callback was not loaded")
    elapsed_wall = time.perf_counter() - started
    return ReplayResult(
        rows=len(source.groups) * source.sequence_length,
        warmup_rows=warmup_rows,
        predicted_rows=predicted_rows,
        callback_wall_ns=callback_wall_ns,
        elapsed_wall_seconds=elapsed_wall,
        elapsed_cpu_seconds=time.process_time() - cpu_started,
        reset_check_ns=reset_check_ns,
        reset_aba_equal=reset_aba_equal,
        sequence_receipts=receipts,
        native_relative_path=native_details[0],
        native_sha256=native_details[1],
        native_level=native_details[2],
        dispatch_level=native_details[3],
        quant_backend=quant_backend,
        member_count=native_details[4],
    )
