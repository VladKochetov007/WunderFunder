"""Rebuild pinned cache bytes on a remote worker with bounded resident memory."""

from __future__ import annotations

import base64
import hashlib
import io
import mmap
import os
import shutil
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from wnn.data import SequenceFile

ROWS = 20_000
WIDTH = 117
RAW_WIDTH = 112
FIT_SUFFIX_WIDTH = 5
PARENT_WIDTH = 2
Reader = Callable[[str, int], np.ndarray]


def decode_header(value: dict[str, Any], expected_shape: tuple[int, ...]) -> bytes:
    """Decode and validate an attested NPY v1 header without reading its payload."""
    header = base64.b64decode(value["base64"], validate=True)
    if hashlib.sha256(header).hexdigest() != value["sha256"] or len(header) != int(value["bytes"]):
        raise ValueError("transported original NPY header digest or byte count changed")
    stream = io.BytesIO(header)
    if np.lib.format.read_magic(stream) != (1, 0):
        raise ValueError("portable cache requires the original NPY v1 header")
    shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
    if tuple(shape) != expected_shape or fortran or dtype != np.dtype("<f4"):
        raise ValueError("original NPY header differs from the expected cache contract")
    if stream.tell() != len(header):
        raise ValueError("original NPY header contains trailing or missing bytes")
    return header


def read_npy_header(path: Path) -> tuple[bytes, tuple[int, ...], np.dtype[Any]]:
    """Read a bounded NPY header and return its exact bytes, shape and dtype."""
    with path.open("rb") as stream:
        version = np.lib.format.read_magic(stream)
        if version != (1, 0):
            raise ValueError(f"transport array must use NPY v1, found {version}")
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
        header_bytes = stream.tell()
        stream.seek(0)
        header = stream.read(header_bytes)
    if fortran:
        raise ValueError("transport arrays must be C-order")
    return header, tuple(shape), dtype


def read_npy_sequence(
    stream: io.BufferedReader,
    index: int,
    header_bytes: int,
    elements_per_sequence: int,
    digest: Any,
) -> np.ndarray:
    """Read one contiguous transport sequence and update its ordered full-file hash."""
    byte_count = elements_per_sequence * np.dtype("<f4").itemsize
    stream.seek(header_bytes + index * byte_count)
    block = stream.read(byte_count)
    if len(block) != byte_count:
        raise ValueError("transport suffix ended before its declared sequence count")
    digest.update(block)
    return np.frombuffer(block, dtype="<f4").reshape(ROWS, -1).copy()


def verify_parquet_source(path: Path, identity: dict[str, Any]) -> None:
    """Check frozen source size and footer digest without hashing the full Parquet file."""
    if path.stat().st_size != int(identity["bytes"]):
        raise ValueError(f"portable Parquet size differs from the campaign pin: {path}")
    footer_bytes = int(identity["footer_bytes"])
    if footer_bytes <= 0 or footer_bytes + 8 > path.stat().st_size:
        raise ValueError("portable Parquet footer length is invalid")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        stream.seek(-8, os.SEEK_END)
        trailer = stream.read(8)
        if trailer[4:] != b"PAR1" or int.from_bytes(trailer[:4], "little") != footer_bytes:
            raise ValueError("portable Parquet trailer differs from the source pin")
        stream.seek(-8 - footer_bytes, os.SEEK_END)
        remaining = footer_bytes + 8
        while remaining:
            block = stream.read(min(remaining, 1 << 20))
            if not block:
                raise ValueError("portable Parquet footer is truncated")
            digest.update(block)
            remaining -= len(block)
    if digest.hexdigest() != identity["footer_sha256"]:
        raise ValueError(f"portable Parquet footer hash differs from the source pin: {path}")


def rebuild_cache_file(
    destination: Path,
    original_header: bytes,
    row_refs: Sequence[dict[str, Any]],
    read_raw: Reader,
    suffix_path: Path,
    suffix_sha256: str,
    suffix_shape: tuple[int, int, int],
    suffix_header_bytes: int,
    original_cache_sha256: str,
    *,
    evaluation: bool,
) -> dict[str, object]:
    """Stream one complete cache to ordinary disk writes and compare its full SHA."""
    expected_shape = (len(row_refs), ROWS, WIDTH)
    _, original_shape, original_dtype = _header_identity(original_header)
    if original_shape != expected_shape or original_dtype != np.dtype("<f4"):
        raise ValueError("attested original cache header does not match the frozen row order")
    expected_suffix_width = PARENT_WIDTH if evaluation else FIT_SUFFIX_WIDTH
    if suffix_shape != (len(row_refs), ROWS, expected_suffix_width):
        raise ValueError("portable suffix header differs from the complete frozen membership")
    suffix_header, observed_shape, observed_dtype = read_npy_header(suffix_path)
    if (
        observed_shape != suffix_shape
        or observed_dtype != np.dtype("<f4")
        or len(suffix_header) != suffix_header_bytes
    ):
        raise ValueError("transport suffix NPY header differs from its manifest")

    suffix_digest, output_digest = hashlib.sha256(), hashlib.sha256()
    suffix_digest.update(suffix_header)
    output_digest.update(original_header)
    suffix_width = expected_suffix_width
    with suffix_path.open("rb") as suffix, destination.open("xb") as output:
        output.write(original_header)
        for index, ref in enumerate(row_refs):
            suffix_values = read_npy_sequence(
                suffix,
                index,
                suffix_header_bytes,
                ROWS * suffix_width,
                suffix_digest,
            )
            raw = read_raw(str(ref["split"]), int(ref["row_group"]))
            if raw.shape != (ROWS, RAW_WIDTH):
                raise ValueError("portable Parquet reader returned a noncanonical raw sequence")
            record = np.empty((ROWS, WIDTH), dtype="<f4")
            record[:, :RAW_WIDTH] = raw
            if evaluation:
                record[:, RAW_WIDTH : RAW_WIDTH + 3] = 0.0
                record[:, RAW_WIDTH + 3 :] = suffix_values
            else:
                record[:, RAW_WIDTH:] = suffix_values
            block = record.tobytes(order="C")
            output.write(block)
            output_digest.update(block)
            del suffix_values, raw, record, block
        output.flush()
        os.fsync(output.fileno())
    if suffix_digest.hexdigest() != suffix_sha256:
        destination.unlink(missing_ok=True)
        raise ValueError("transport suffix full-file hash differs from its manifest")
    output_sha = output_digest.hexdigest()
    if output_sha != original_cache_sha256:
        destination.unlink(missing_ok=True)
        raise ValueError("rebuilt full cache SHA differs from the original pinned payload")
    return {"path": str(destination), "sha256": output_sha, "shape": list(expected_shape)}


def reader_for_sources(source_paths: dict[str, Path]) -> tuple[Reader, Callable[[], None]]:
    """Open canonical row-group readers and return a close callback."""
    files: dict[str, SequenceFile] = {}
    try:
        for split, path in source_paths.items():
            files[split] = SequenceFile(path)
    except BaseException:
        _close_sources(files)
        raise

    def _read_raw(split: str, row_group: int) -> np.ndarray:
        if split not in files:
            raise ValueError(f"portable raw source map omits {split!r}")
        sequence = files[split].read(row_group, targets=False, mask=False)
        return np.asarray(sequence.features, dtype=np.float32)

    def _close() -> None:
        _close_sources(files)

    return _read_raw, _close


def _close_sources(files: dict[str, SequenceFile]) -> None:
    for source in files.values():
        handle = getattr(source._pf, "close", None)
        if handle is not None:
            handle()


def copy_pinned_file(source: Path, destination: Path, expected_sha256: str) -> None:
    """Copy a small pinned sidecar or parent array without changing its bytes."""
    shutil.copyfile(source, destination)
    if _file_sha256(destination) != expected_sha256:
        destination.unlink(missing_ok=True)
        raise ValueError(f"copied portable input changed its pinned SHA: {source.name}")


def release_sequence_pages(array: np.ndarray, index: int) -> None:
    """Drop one fully consumed memmap sequence from process RSS on Linux."""
    if not isinstance(array, np.memmap):
        return
    mapping = array._mmap
    if not hasattr(mapping, "madvise") or not hasattr(mmap, "MADV_DONTNEED"):
        raise RuntimeError("portable Kaggle mode requires mmap MADV_DONTNEED support")
    page = mmap.PAGESIZE
    sequence_bytes = int(np.prod(array.shape[1:])) * array.dtype.itemsize
    start = int(array.offset) + index * sequence_bytes
    end = start + sequence_bytes
    page_start = start - start % page
    page_end = min(len(mapping), ((end + page - 1) // page) * page)
    if page_end > page_start:
        mapping.madvise(mmap.MADV_DONTNEED, page_start, page_end - page_start)


def _header_identity(header: bytes) -> tuple[int, tuple[int, ...], np.dtype[Any]]:
    stream = io.BytesIO(header)
    version = np.lib.format.read_magic(stream)
    shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
    if version != (1, 0) or fortran or stream.tell() != len(header):
        raise ValueError("portable full-cache header must be exact C-order NPY v1 bytes")
    return len(header), tuple(shape), dtype


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
