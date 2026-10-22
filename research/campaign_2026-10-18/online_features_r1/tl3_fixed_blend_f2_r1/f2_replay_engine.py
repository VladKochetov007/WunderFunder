"""Run an exact packaged callback over complete feature-only sequences."""

from __future__ import annotations

import os
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pyarrow as pa

PACKET = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKET.parent / "tl3_replay"))

from f2_candidates import CANDIDATES  # noqa: E402
from f2_native_runtime import load_callback_f2  # noqa: E402
from f2_source import F2Source  # noqa: E402
from native_runtime import extract_safe, run_sequence, sha256_file  # noqa: E402

EXPECTED_ZIPS = CANDIDATES


@dataclass(frozen=True)
class ReplayResult:
    """Measured callback work and native package identity."""

    rows: int
    warmup_rows: int
    predicted_rows: int
    callback_wall_ns: int
    elapsed_wall_seconds: float
    elapsed_cpu_seconds: float
    reset_check_ns: int
    reset_aba_equal: bool | None
    sequences: list[dict[str, int]]
    native_relative_path: str
    native_sha256: str
    member_count: int
    precision: str


def replay(
    source: F2Source,
    candidate: str,
    archive_path: Path,
    scratch: Path,
    predictions_path: Path,
    reset_check: bool,
) -> ReplayResult:
    """Run complete selected groups and atomically save sequence-aligned outputs."""
    from utils import DataPoint

    started, cpu_started = time.perf_counter(), time.process_time()
    callback_wall_ns = warmup_rows = predicted_rows = reset_check_ns = 0
    reset_aba_equal: bool | None = None
    sequence_receipts: list[dict[str, int]] = []
    pending_path = predictions_path.with_name(f".{predictions_path.name}.pending.npy")
    if pending_path.exists() or predictions_path.exists():
        raise FileExistsError(f"refusing to overwrite {predictions_path}")
    temporary = predictions_path.parent
    if not scratch.is_dir() or not temporary.is_dir():
        raise NotADirectoryError("scratch and prediction directories must already exist")
    if candidate not in EXPECTED_ZIPS:
        raise ValueError(f"unknown candidate {candidate!r}")
    expected = EXPECTED_ZIPS[candidate]
    if sha256_file(archive_path) != expected["zip_sha256"]:
        raise ValueError(f"{candidate} ZIP hash differs from the frozen candidate")

    try:
        with zipfile.ZipFile(archive_path) as archive:
            if archive.testzip() is not None:
                raise ValueError("candidate ZIP CRC validation failed")
            with TemporaryDirectory(prefix="f2-replay-", dir=scratch) as package_dir:
                package = Path(package_dir)
                extract_safe(archive, package)
                callback, native_path = load_callback_f2(package)
                precision_path = package / "precision.txt"
                precision = (
                    precision_path.read_text(encoding="utf-8").strip()
                    if precision_path.is_file()
                    else "package-defined"
                )
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
                    sequence_id, need, features = _validate_features(table, source, row_group)
                    values, warm, predicted, elapsed_ns = run_sequence(
                        callback, DataPoint, sequence_id, features, need
                    )
                    callback_wall_ns += elapsed_ns
                    warmup_rows += warm
                    predicted_rows += predicted
                    output[index] = values
                    sequence_receipts.append(
                        {"row_group": row_group, "sequence_id": sequence_id, "rows": len(features)}
                    )
                    if index == 0 and reset_check:
                        first_group = (row_group, sequence_id, features.copy(), need.copy())
                    del table, features, need, values
                    if (index + 1) % 32 == 0:
                        print(
                            f"candidate={candidate} groups={index + 1}/{len(source.groups)} "
                            f"elapsed={time.perf_counter() - started:.1f}s",
                            flush=True,
                        )
                if reset_check:
                    if first_group is None or len(source.groups) < 2:
                        raise ValueError("A-B-A check requires at least two complete groups")
                    row_group, sequence_id, features, need = first_group
                    reset_started = time.perf_counter_ns()
                    replayed, _, _, _ = run_sequence(
                        callback, DataPoint, sequence_id, features, need
                    )
                    reset_check_ns = time.perf_counter_ns() - reset_started
                    reset_aba_equal = bool(np.array_equal(replayed, output[0]))
                    if not reset_aba_equal:
                        raise ValueError(f"A-B-A reset replay differs at row group {row_group}")
                    del features, need, replayed
                output.flush()
                del output
                native_relative_path = native_path.relative_to(package).as_posix()
                native_hash = sha256_file(native_path)
                member_count = int(getattr(callback, "member_count", 1))
                os.replace(pending_path, predictions_path)
    except BaseException:
        pending_path.unlink(missing_ok=True)
        predictions_path.unlink(missing_ok=True)
        raise

    return ReplayResult(
        rows=len(source.groups) * source.sequence_length,
        warmup_rows=warmup_rows,
        predicted_rows=predicted_rows,
        callback_wall_ns=callback_wall_ns,
        elapsed_wall_seconds=time.perf_counter() - started,
        elapsed_cpu_seconds=time.process_time() - cpu_started,
        reset_check_ns=reset_check_ns,
        reset_aba_equal=reset_aba_equal,
        sequences=sequence_receipts,
        native_relative_path=native_relative_path,
        native_sha256=native_hash,
        member_count=member_count,
        precision=precision,
    )


def _validate_features(
    table: pa.Table, source: F2Source, row_group: int
) -> tuple[int, np.ndarray, np.ndarray]:
    """Validate the expected source sequence and finite feature state."""
    from utils import validate_sequence

    sequence_id, need = validate_sequence(table)
    if sequence_id != source.sequence_by_group[row_group]:
        raise ValueError("producer sequence identity differs from fold manifest")
    expected_need = np.arange(source.sequence_length) >= 99
    if not np.array_equal(need, expected_need):
        raise ValueError("producer prediction mask differs from the fixed 99-row warmup")
    features = np.column_stack(
        [table.column(name).to_numpy() for name in source.feature_columns]
    ).astype(np.float32, copy=False)
    if features.shape != (source.sequence_length, len(source.feature_columns)):
        raise ValueError("producer row group has an unexpected feature shape")
    if not np.isfinite(features).all():
        raise ValueError(f"producer row group {row_group} has nonfinite features")
    return sequence_id, need, features
