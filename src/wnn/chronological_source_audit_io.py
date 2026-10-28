"""Bounded file hashing and source-mutation checks for chronological audits."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable
from pathlib import Path

HASH_CHUNK_BYTES = 8 << 20
HASH_PROGRESS_BYTES = 512 << 20


def file_sha256_with_progress(path: Path, progress: Callable[[int], None]) -> tuple[str, int]:
    """Hash file bytes in bounded chunks and return digest plus bytes actually read."""
    digest = hashlib.sha256()
    bytes_hashed = 0
    next_report = HASH_PROGRESS_BYTES
    with path.open("rb") as stream:
        while chunk := stream.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
            bytes_hashed += len(chunk)
            if bytes_hashed >= next_report:
                progress(bytes_hashed)
                next_report += HASH_PROGRESS_BYTES
    progress(bytes_hashed)
    return digest.hexdigest(), bytes_hashed


def assert_stat_unchanged(path: Path, before: os.stat_result, phase: str) -> None:
    """Reject source replacement or mutation using only size and modification time."""
    after = path.stat()
    if after.st_size != before.st_size or after.st_mtime_ns != before.st_mtime_ns:
        raise ValueError(
            f"source size or mtime changed during {phase}: {path} "
            f"({before.st_size}, {before.st_mtime_ns}) -> ({after.st_size}, {after.st_mtime_ns})"
        )
