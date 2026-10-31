from __future__ import annotations

import json
import os
import re
import stat
from pathlib import Path
from typing import BinaryIO

from wnn.dashboard.schema import sanitize_config, sanitize_json

MAX_METADATA_BYTES = 16 * 1_048_576
MAX_DISCOVERED_RUNS = 64
MAX_LISTED_RUNS = 48
RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\Z")
ALLOWED_FILES = {
    "run_metadata.json",
    "monitor_events.jsonl",
    "monitor_evaluations.jsonl",
}


class MissingFileError(FileNotFoundError):
    """Raised when an optional file is absent from a run directory."""


class UnsafePathError(OSError):
    """Raised when a run file or directory is not a regular local path."""


def _root_stat(root: Path) -> os.stat_result:
    """Verify the resolved run root is still the same direct directory path."""
    try:
        resolved = root.resolve(strict=True)
        root_stat = os.lstat(root)
    except (OSError, RuntimeError) as exc:
        raise UnsafePathError("run root is unavailable") from exc
    if resolved != root or stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise UnsafePathError("run root is not a resolved directory")
    return root_stat


def validate_run_id(run_id: str) -> bool:
    """Return whether an identifier can name one direct child run directory."""
    return (
        bool(RUN_ID.fullmatch(run_id))
        and run_id not in {".", ".."}
        and ".." not in run_id
        and Path(run_id).name == run_id
    )


def open_run_file(root: Path, run_id: str, filename: str) -> BinaryIO:
    """Open one allowlisted regular file without following final symlinks."""
    if not validate_run_id(run_id) or filename not in ALLOWED_FILES:
        raise UnsafePathError("invalid run file")
    root_stat = _root_stat(root)
    run_path = root / run_id
    file_path = run_path / filename
    try:
        run_stat = os.lstat(run_path)
    except FileNotFoundError as exc:
        raise MissingFileError("run directory is missing") from exc
    if stat.S_ISLNK(run_stat.st_mode) or not stat.S_ISDIR(run_stat.st_mode):
        raise UnsafePathError("run directory is not a direct directory")
    try:
        path_stat = os.lstat(file_path)
    except FileNotFoundError as exc:
        raise MissingFileError("run file is missing") from exc
    if stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISREG(path_stat.st_mode):
        raise UnsafePathError("run file is not a regular file")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(file_path, flags)
    try:
        opened_stat = os.fstat(descriptor)
        current_dir_stat = os.lstat(run_path)
        current_root_stat = _root_stat(root)
        if (
            not stat.S_ISREG(opened_stat.st_mode)
            or opened_stat.st_dev != path_stat.st_dev
            or opened_stat.st_ino != path_stat.st_ino
            or stat.S_ISLNK(current_dir_stat.st_mode)
            or current_dir_stat.st_dev != run_stat.st_dev
            or current_dir_stat.st_ino != run_stat.st_ino
            or current_root_stat.st_dev != root_stat.st_dev
            or current_root_stat.st_ino != root_stat.st_ino
        ):
            raise UnsafePathError("run file changed during open")
        return os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise


def discover_run_ids(root: Path) -> list[tuple[str, int]]:
    """List recent direct run directories without traversing symlinks."""
    _root_stat(root)
    found: list[tuple[str, int]] = []
    with os.scandir(root) as entries:
        for entry in entries:
            if len(found) >= MAX_DISCOVERED_RUNS:
                break
            if not validate_run_id(entry.name) or entry.is_symlink():
                continue
            try:
                entry_stat = entry.stat(follow_symlinks=False)
            except OSError:
                continue
            if stat.S_ISDIR(entry_stat.st_mode):
                found.append((entry.name, entry_stat.st_mtime_ns))
    found.sort(key=lambda item: item[1], reverse=True)
    # One extra entry lets the reader report whether the visible list was capped.
    return found[: MAX_LISTED_RUNS + 1]


def metadata_signature(handle: BinaryIO) -> tuple[int, int, int, int]:
    """Return a cache key for the metadata file currently held open."""
    metadata_stat = os.fstat(handle.fileno())
    return (
        metadata_stat.st_dev,
        metadata_stat.st_ino,
        metadata_stat.st_size,
        metadata_stat.st_mtime_ns,
    )


def read_run_metadata(
    root: Path, run_id: str, cached_signature: tuple[int, ...] | None = None
) -> tuple[dict[str, object] | None, tuple[int, ...]]:
    """Read only source, config, and provenance from bounded run metadata."""
    with open_run_file(root, run_id, "run_metadata.json") as handle:
        signature = metadata_signature(handle)
        if signature == cached_signature:
            return None, signature
        if signature[2] > MAX_METADATA_BYTES:
            raise ValueError("metadata is too large")
        payload = handle.read(MAX_METADATA_BYTES + 1)
    decoded: object = json.loads(payload)
    if not isinstance(decoded, dict):
        raise ValueError("metadata must be a JSON object")
    return (
        {
            "source": sanitize_json(decoded.get("source", "unknown")),
            "config": sanitize_config(decoded.get("config", {})),
            "provenance": sanitize_json(decoded.get("provenance", {})),
            "synthetic": bool(decoded.get("synthetic", False)),
        },
        signature,
    )
