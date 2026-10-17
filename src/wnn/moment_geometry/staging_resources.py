"""Bound page-cache use and estimate the disk footprint of geometry staging."""

from __future__ import annotations

import mmap
import os
from pathlib import Path

import numpy as np

SEQUENCE_ROWS = 20_000
NPY_HEADER_BYTES = 128
RAW_WIDTH = 112
BASE_WIDTH = 231
DISCRETE_WIDTH = 271


def advise_file_range(file_descriptor: int, offset: int, length: int) -> None:
    """Advise Linux to discard clean physical-file pages for one byte range."""
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        raise RuntimeError("bounded staging requires POSIX_FADV_DONTNEED support")
    page_size = mmap.PAGESIZE
    start = offset // page_size * page_size
    end = (offset + length + page_size - 1) // page_size * page_size
    if end > start:
        os.posix_fadvise(file_descriptor, start, end - start, os.POSIX_FADV_DONTNEED)


def flush_and_discard_sequence(path: Path, array: np.memmap, index: int, *, durable: bool) -> None:
    """Flush and evict one sequence range from a staging memmap."""
    page_size = mmap.PAGESIZE
    offset = int(array.offset + index * array.strides[0])
    length = int(array.strides[0])
    start = offset // page_size * page_size
    requested_end = (offset + length + page_size - 1) // page_size * page_size
    mapped_end = min(requested_end, array._mmap.size())
    if mapped_end > start:
        if durable:
            array._mmap.flush(start, mapped_end - start)
        if hasattr(array._mmap, "madvise"):
            array._mmap.madvise(mmap.MADV_DONTNEED, start, mapped_end - start)
    flags = os.O_RDWR if durable else os.O_RDONLY
    descriptor = os.open(path, flags)
    try:
        if durable:
            os.fdatasync(descriptor)
        advise_file_range(descriptor, offset, length)
    finally:
        os.close(descriptor)


def _sequence_array_bytes(sequences: int, width: int, dtype_bytes: int) -> int:
    """Return logical bytes for one per-row feature array, including its header."""
    return NPY_HEADER_BYTES + sequences * SEQUENCE_ROWS * width * dtype_bytes


def _index_array_bytes(sequences: int) -> int:
    """Return logical bytes for a one-dimensional int64 sequence index array."""
    return NPY_HEADER_BYTES + sequences * 8


def estimate_staging_bytes(fit_sequences: int, eval_sequences: int) -> dict[str, int | str]:
    """Estimate disk bytes from sequence counts without reading dataset payloads."""
    raw_stage = 0
    base_features = 0
    discrete_features = 0
    for sequences in (fit_sequences, eval_sequences):
        raw_stage += _sequence_array_bytes(sequences, RAW_WIDTH, 4)
        raw_stage += _sequence_array_bytes(sequences, 2, 4)
        raw_stage += _sequence_array_bytes(sequences, 1, 1)
        raw_stage += _index_array_bytes(sequences) * 2
        base_features += _sequence_array_bytes(sequences, BASE_WIDTH, 4)
        discrete_features += _sequence_array_bytes(sequences, DISCRETE_WIDTH, 4)
    return {
        "raw_stage_bytes": raw_stage,
        "base_features_incremental_bytes": base_features,
        "discrete_features_incremental_bytes": discrete_features,
        "persistent_total_bytes": raw_stage + base_features + discrete_features,
        "estimate_scope": "logical NPY sizes including 128-byte headers; excludes small JSON and filesystem allocation rounding",
    }
