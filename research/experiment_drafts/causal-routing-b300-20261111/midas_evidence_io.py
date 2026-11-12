"""Bounded, duplicate-key-safe file helpers for Signal Routes's MIDAS evidence exporter."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any


def _pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in rows:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _object(raw: bytes, label: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        """Reject JavaScript-style non-finite values from strict JSON input."""
        raise ValueError(f"non-finite JSON constant: {value}")

    value = json.loads(raw, object_pairs_hook=_pairs, parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _stat(path: Path, limit: int) -> os.stat_result:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"unsafe evidence: {path.name}")
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
        raise ValueError(f"evidence exceeds bound: {path.name}")
    return info


def _unchanged(before: os.stat_result, after: os.stat_result, count: int, label: str) -> None:
    if count != before.st_size or (before.st_size, before.st_mtime_ns, before.st_ino) != (
        after.st_size,
        after.st_mtime_ns,
        after.st_ino,
    ):
        raise ValueError(f"evidence changed while reading: {label}")


def _read(path: Path, limit: int = 64 << 20) -> tuple[dict[str, Any], bytes]:
    before = _stat(path, limit)
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    after = path.lstat()
    if len(raw) > limit:
        raise ValueError(f"evidence exceeds bound: {path.name}")
    _unchanged(before, after, len(raw), path.name)
    return _object(raw, path.name), raw


def _read_bytes(path: Path, limit: int) -> bytes:
    before = _stat(path, limit)
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    after = path.lstat()
    if len(raw) > limit:
        raise ValueError(f"evidence exceeds bound: {path.name}")
    _unchanged(before, after, len(raw), path.name)
    return raw


def _hash_file(path: Path, limit: int) -> tuple[str, int]:
    before = _stat(path, limit)
    digest = hashlib.sha256()
    count = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            count += len(chunk)
            if count > limit:
                raise ValueError(f"evidence exceeds bound: {path.name}")
            digest.update(chunk)
    _unchanged(before, path.lstat(), count, path.name)
    return digest.hexdigest(), count


def _write(stage: Path, name: str, raw: bytes, files: dict[str, dict[str, Any]]) -> dict[str, Any]:
    path = stage / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)
    item = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    files[name] = item
    return {"path": name, **item}


def _copy(
    stage: Path, source: Path, name: str, limit: int, files: dict[str, dict[str, Any]]
) -> None:
    before = _stat(source, limit)
    target = stage / name
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    count = 0
    with source.open("rb") as src, target.open("xb") as dst:
        while chunk := src.read(1 << 20):
            count += len(chunk)
            if count > limit:
                raise ValueError(f"evidence exceeds bound: {source.name}")
            digest.update(chunk)
            dst.write(chunk)
    _unchanged(before, source.lstat(), count, source.name)
    files[name] = {"sha256": digest.hexdigest(), "bytes": count}
