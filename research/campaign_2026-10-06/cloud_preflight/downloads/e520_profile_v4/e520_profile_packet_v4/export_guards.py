"""Small fail-closed guards shared by a future staged E520 prediction exporter."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


class ExportContractError(ValueError):
    """Raised when export provenance, disk, or chunk dimensions are inconsistent."""


def prediction_storage_bytes(rows: int, sequence_rows: int = 20_000, outputs: int = 2) -> int:
    """Return exact float32 output bytes, rejecting non-positive dimensions."""
    if min(rows, sequence_rows, outputs) <= 0:
        raise ExportContractError("prediction dimensions must be positive")
    return rows * sequence_rows * outputs * 4


def validate_streaming_export_preconditions(
    *,
    rows: int,
    free_disk_bytes: int,
    reserve_bytes: int,
    split_sha256: str,
    expected_split_sha256: str,
    row_groups_sha256: str,
    expected_row_groups_sha256: str,
    provenance: dict[str, Any],
    checkpoint_sha256: str,
    target_order_sha256: str,
) -> int:
    """Verify immutable ancestry and disk budget before opening a memmap staging file."""
    expected_hex = (
        split_sha256,
        expected_split_sha256,
        row_groups_sha256,
        expected_row_groups_sha256,
        checkpoint_sha256,
        target_order_sha256,
    )
    if any(
        len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
        for value in expected_hex
    ):
        raise ExportContractError("export provenance digests must be lowercase SHA-256 hex")
    if split_sha256 != expected_split_sha256 or row_groups_sha256 != expected_row_groups_sha256:
        raise ExportContractError("export split or target row-group digest changed")
    if provenance.get("split_manifest_sha256") != split_sha256:
        raise ExportContractError("checkpoint provenance uses another split")
    if provenance.get("checkpoint_sha256") != checkpoint_sha256:
        raise ExportContractError("checkpoint provenance hash differs from the loaded member")
    if provenance.get("target_order_sha256") != target_order_sha256:
        raise ExportContractError("checkpoint provenance uses another target row order")
    needed = prediction_storage_bytes(rows)
    if free_disk_bytes < needed + reserve_bytes:
        raise ExportContractError(
            "disk cannot hold the staged output and required free-space reserve"
        )
    return needed


def chunk_ranges(total_rows: int, chunk_rows: int = 4_000) -> list[tuple[int, int]]:
    """Return bounded contiguous exporter slices; avoid full-output materialization."""
    if total_rows <= 0 or chunk_rows != 4_000:
        raise ExportContractError("E520 exports require positive rows and fixed 4,000-row chunks")
    return [
        (start, min(start + chunk_rows, total_rows)) for start in range(0, total_rows, chunk_rows)
    ]


def validate_chunk_shape(actual_shape: tuple[int, ...], expected_shape: tuple[int, ...]) -> None:
    """Require one bounded float32 prediction chunk with the exact expected shape."""
    if actual_shape != expected_shape or len(actual_shape) != 3 or actual_shape[2] != 2:
        raise ExportContractError(
            f"prediction chunk shape mismatch: {actual_shape} != {expected_shape}"
        )


def streaming_sha256(path: Path, block_bytes: int = 1 << 20) -> str:
    """Hash a finalized output incrementally without loading it into memory."""
    if block_bytes <= 0:
        raise ExportContractError("hash block size must be positive")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_bytes), b""):
            digest.update(block)
    return digest.hexdigest()
