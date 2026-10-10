"""Dataset layout and parquet row-group readers.

Each parquet row group is one complete 20,000-row sequence. Feature indices are
identifiers, not depth levels (docs/data_overview.md).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

SEQUENCE_LENGTH = 20_000
WARMUP = 99
N_FEATURES = 112
FEATURE_COLUMNS = [
    f"i{i}_{g}{j}"
    for i in range(2)
    for g, n in [("p", 22), ("v", 22), ("dp", 4), ("dv", 4)]
    for j in range(n)
] + [f"a{i}" for i in range(8)]
TARGET_COLUMNS = ("t0", "t1")

# Row groups that fail to decode in the local copy (research E-032/E-035).
CORRUPT_TRAIN_ROW_GROUPS = frozenset({528})

# Target values are integer multiples of one of these quanta (tick / scale).
# Observed on validation; identifies the market class of a labelled sequence.
TARGET_QUANTA = np.array([0.12543, 0.24133, 0.46854, 0.65687])


@dataclass(frozen=True)
class Instrument:
    offset: int

    @property
    def bid_px(self) -> slice:
        return slice(self.offset, self.offset + 11)

    @property
    def ask_px(self) -> slice:
        return slice(self.offset + 11, self.offset + 22)

    @property
    def bid_vol(self) -> slice:
        return slice(self.offset + 22, self.offset + 33)

    @property
    def ask_vol(self) -> slice:
        return slice(self.offset + 33, self.offset + 44)

    @property
    def trade_px(self) -> slice:
        return slice(self.offset + 44, self.offset + 48)

    @property
    def trade_vol(self) -> slice:
        return slice(self.offset + 48, self.offset + 52)


I0 = Instrument(0)
I1 = Instrument(52)
EXTRA = slice(104, 112)


@dataclass(frozen=True)
class Sequence:
    seq_ix: int
    features: np.ndarray  # (T, 112) float32
    targets: np.ndarray | None  # (T, 2) float32
    scored: np.ndarray | None  # (T,) bool


class SequenceFile:
    """Random access to sequences of a competition parquet file."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._pf = pq.ParquetFile(self.path)
        names = self._pf.schema_arrow.names
        self.has_targets = "t0" in names
        self.has_mask = "is_scored" in names

    def __len__(self) -> int:
        return self._pf.num_row_groups

    def read(self, index: int, *, targets: bool = True, mask: bool = True) -> Sequence:
        columns = ["seq_ix", *FEATURE_COLUMNS]
        want_targets = targets and self.has_targets
        want_mask = mask and self.has_mask
        if want_targets:
            columns += list(TARGET_COLUMNS)
        if want_mask:
            columns.append("is_scored")
        table = self._pf.read_row_group(index, columns=columns, use_threads=False)
        if table.num_rows != SEQUENCE_LENGTH:
            raise ValueError(f"row group {index} has {table.num_rows} rows")
        # column-major stack then one transpose: same values as filling columns into a row-major array,
        # without 112 strided writes (tests/test_perf_parity.py checks bit identity)
        features = np.ascontiguousarray(
            np.stack([table.column(name).to_numpy() for name in FEATURE_COLUMNS], axis=0)
            .astype(np.float32, copy=False)
            .T
        )
        y = None
        if want_targets:
            y = np.column_stack([table.column(c).to_numpy() for c in TARGET_COLUMNS])
            y = y.astype(np.float32, copy=False)
        scored = table.column("is_scored").to_numpy().astype(bool) if want_mask else None
        _record_cache_recovery_read(self.path, index, self._pf)
        return Sequence(int(table.column("seq_ix")[0].as_py()), features, y, scored)


def _record_cache_recovery_read(path: Path, index: int, parquet_file: pq.ParquetFile) -> None:
    """Trace byte ranges from the existing reader for bounded pressure relief."""
    trace_path = os.environ.get("WNN_CACHE_TRACE_PATH")
    if not trace_path:
        return
    canonical_path = str(path.resolve(strict=True))
    split = None
    for candidate in ("train", "valid"):
        expected_path = os.environ.get(f"WNN_CACHE_{candidate.upper()}_PATH")
        if expected_path and canonical_path == expected_path:
            split = candidate
            break
    if split is None:
        return
    metadata = parquet_file.metadata
    if index < 0 or index >= metadata.num_row_groups:
        raise ValueError("cache recovery trace rowgroup is outside the opened parquet footer")
    group = metadata.row_group(index)
    starts = []
    ends = []
    for column_index in range(group.num_columns):
        column = group.column(column_index)
        offsets = [
            int(value)
            for value in (
                column.file_offset,
                column.data_page_offset,
                column.dictionary_page_offset,
            )
            if value is not None and int(value) >= 4
        ]
        compressed_size = int(column.total_compressed_size)
        if not offsets or compressed_size <= 0:
            raise ValueError("cache recovery trace rowgroup lacks valid column byte ranges")
        chunk_start = min(offsets)
        starts.append(chunk_start)
        ends.append(chunk_start + compressed_size)
    start, end = min(starts), max(ends)
    if start < 4 or end <= start or end > path.stat().st_size:
        raise ValueError("cache recovery trace rowgroup range is outside the parquet input")
    descriptor = os.open(trace_path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
    try:
        record = f"{split}\t{int(index)}\t{start}\t{end}\n".encode("ascii")
        if os.write(descriptor, record) != len(record):
            raise OSError("short rowgroup trace write")
    finally:
        os.close(descriptor)


def market_class(targets: np.ndarray) -> int:
    """Index into TARGET_QUANTA of a labelled sequence, or -1 when undetermined."""
    values = np.unique(np.abs(np.asarray(targets, dtype=np.float64)))
    gaps = np.diff(values)
    gaps = gaps[gaps > 1e-6]
    if gaps.size == 0:
        return -1
    quantum = gaps.min()
    k = int(np.argmin(np.abs(TARGET_QUANTA - quantum)))
    return k if abs(TARGET_QUANTA[k] - quantum) < 1e-3 else -1
