"""Bounded, no-follow file operations for the F1 result collector."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import stat
import struct
import sys
from pathlib import Path
from typing import Any

COPY_CHUNK = 1024 * 1024
MAX_JSON_BYTES = 4 * 1024 * 1024
MAX_SIDECAR_BYTES = 128 * 1024
MAX_EVIDENCE_BYTES = 16 * 1024 * 1024
MAX_PREDICTION_BYTES = 128 * 1024 * 1024

Json = dict[str, Any]


class CollectionError(RuntimeError):
    """Raised when collection input or provenance fails a check."""


def require(condition: bool, message: str) -> None:
    """Raise a descriptive error when a collection invariant is false."""
    if not condition:
        raise CollectionError(message)


def project_path(path: Path, root: Path) -> Path:
    """Normalize a project-local path and refuse symlinked parents or targets."""
    absolute = Path(os.path.abspath(path))
    require(absolute.is_relative_to(root), f"path escapes project root: {absolute}")
    current = root
    for part in absolute.relative_to(root).parts:
        current /= part
        try:
            info = current.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise CollectionError(f"cannot inspect path component {current}: {exc}") from exc
        require(not stat.S_ISLNK(info.st_mode), f"symlinked project path is forbidden: {current}")
    return absolute


def identity(path: Path) -> dict[str, int]:
    """Return a no-follow identity for a regular input file."""
    try:
        info = path.lstat()
    except OSError as exc:
        raise CollectionError(f"cannot stat input {path}: {exc}") from exc
    require(stat.S_ISREG(info.st_mode), f"input is not a regular non-symlink file: {path}")
    return {
        "device": info.st_dev,
        "inode": info.st_ino,
        "size": info.st_size,
        "mtime_ns": info.st_mtime_ns,
        "ctime_ns": info.st_ctime_ns,
    }


def _open(path: Path, expected: dict[str, int]) -> int:
    """Open a file without following symlinks and verify its inode identity."""
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise CollectionError(f"cannot open input without following links {path}: {exc}") from exc
    opened = os.fstat(fd)
    actual = {
        "device": opened.st_dev,
        "inode": opened.st_ino,
        "size": opened.st_size,
        "mtime_ns": opened.st_mtime_ns,
        "ctime_ns": opened.st_ctime_ns,
    }
    if actual != expected:
        os.close(fd)
        raise CollectionError(f"input changed between stat and open: {path}")
    return fd


def read_json(path: Path, limit: int = MAX_JSON_BYTES) -> tuple[Json, str, dict[str, int]]:
    """Read bounded JSON and bind its exact bytes and stable file identity."""
    before = identity(path)
    require(before["size"] <= limit, f"JSON input exceeds {limit} bytes: {path}")
    fd = _open(path, before)
    try:
        with os.fdopen(fd, "rb") as stream:
            raw = stream.read(limit + 1)
            descriptor = os.fstat(stream.fileno())
    except OSError as exc:
        raise CollectionError(f"cannot read JSON input {path}: {exc}") from exc
    require(
        before == identity(path) and len(raw) == before["size"],
        f"JSON input changed while reading: {path}",
    )
    require(
        descriptor.st_ino == before["inode"] and len(raw) <= limit,
        f"JSON descriptor or length changed: {path}",
    )
    try:
        value = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as exc:
        raise CollectionError(f"invalid JSON at {path}: {exc}") from exc
    require(isinstance(value, dict), f"expected JSON object at {path}")
    return value, hashlib.sha256(raw).hexdigest(), before


def hash_stable(path: Path, limit: int) -> tuple[str, dict[str, int]]:
    """Hash a bounded file in chunks and reject replacement or concurrent writes."""
    before = identity(path)
    require(before["size"] <= limit, f"input exceeds {limit} bytes: {path}")
    fd = _open(path, before)
    digest = hashlib.sha256()
    size = 0
    with os.fdopen(fd, "rb") as stream:
        while block := stream.read(COPY_CHUNK):
            digest.update(block)
            size += len(block)
        descriptor = os.fstat(stream.fileno())
    require(
        before == identity(path) and size == before["size"],
        f"input changed or truncated while hashing: {path}",
    )
    require(descriptor.st_ino == before["inode"], f"input descriptor changed while hashing: {path}")
    return digest.hexdigest(), before


def membership_sha256(groups: list[int]) -> str:
    """Hash ordered signed 64-bit group IDs as canonical export code does."""
    digest = hashlib.sha256()
    for group in groups:
        digest.update(struct.pack("<q", group))
    return digest.hexdigest()


def inspect_npy(
    path: Path, expected_shape: list[int]
) -> tuple[dict[str, Any], str, dict[str, int]]:
    """Validate and hash a C-order float32 NPY while streaming finite-value checks."""
    before = identity(path)
    require(before["size"] <= MAX_PREDICTION_BYTES, "prediction exceeds the configured file cap")
    fd = _open(path, before)
    digest = hashlib.sha256()
    with os.fdopen(fd, "rb") as stream:
        magic = stream.read(8)
        digest.update(magic)
        require(magic[:6] == b"\x93NUMPY", f"not an NPY file: {path}")
        if magic[6:8] == b"\x01\x00":
            length_raw, encoding = stream.read(2), "latin1"
            require(len(length_raw) == 2, "truncated NPY v1 header length")
            header_len = struct.unpack("<H", length_raw)[0]
        elif magic[6:8] in {b"\x02\x00", b"\x03\x00"}:
            length_raw = stream.read(4)
            require(len(length_raw) == 4, "truncated NPY v2/v3 header length")
            header_len = struct.unpack("<I", length_raw)[0]
            encoding = "utf-8" if magic[6:8] == b"\x03\x00" else "latin1"
        else:
            raise CollectionError(f"unsupported NPY version {magic[6:8]!r}")
        digest.update(length_raw)
        require(0 < header_len <= 16_384, "invalid NPY header length")
        raw = stream.read(header_len)
        digest.update(raw)
        require(len(raw) == header_len, "truncated NPY header")
        try:
            header = ast.literal_eval(raw.decode(encoding).strip())
        except (SyntaxError, ValueError, UnicodeDecodeError) as exc:
            raise CollectionError(f"invalid NPY header: {exc}") from exc
        require(isinstance(header, dict), "NPY header is not a dict")
        shape, dtype = header.get("shape"), header.get("descr")
        require(
            isinstance(shape, tuple) and all(type(x) is int for x in shape), "invalid NPY shape"
        )
        require(list(shape) == expected_shape, f"NPY shape differs: {shape} != {expected_shape}")
        require(
            dtype == "<f4" or dtype == "=f4" and sys.byteorder == "little",
            "prediction must be little-endian float32",
        )
        require(header.get("fortran_order") is False, "prediction must use C-order")
        offset = stream.tell()
        expected_bytes = offset + math.prod(expected_shape) * 4
        require(
            before["size"] == expected_bytes, "NPY payload length does not match its float32 shape"
        )
        remaining = before["size"] - offset
        while remaining:
            block = stream.read(min(COPY_CHUNK, remaining))
            require(bool(block) and len(block) % 4 == 0, "truncated or unaligned NPY payload")
            digest.update(block)
            require(
                all(math.isfinite(value[0]) for value in struct.iter_unpack("<f", block)),
                "prediction contains nonfinite float32 values",
            )
            remaining -= len(block)
        descriptor = os.fstat(stream.fileno())
    require(
        before == identity(path) and descriptor.st_ino == before["inode"],
        "prediction changed while scanning",
    )
    return (
        {
            "shape": expected_shape,
            "dtype": dtype,
            "fortran_order": False,
            "header_bytes": offset,
            "bytes": before["size"],
        },
        digest.hexdigest(),
        before,
    )


def copy_stable(
    source: Path, destination: Path, expected_sha: str, expected_identity: dict[str, int]
) -> dict[str, Any]:
    """Copy a prevalidated input create-only, rechecking bytes and identity."""
    before = identity(source)
    require(before == expected_identity, f"source changed after preflight: {source}")
    fd = _open(source, before)
    digest = hashlib.sha256()
    size = 0
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        with os.fdopen(fd, "rb") as src, destination.open("xb") as dst:
            os.chmod(destination, 0o600)
            while block := src.read(COPY_CHUNK):
                dst.write(block)
                digest.update(block)
                size += len(block)
            dst.flush()
            os.fsync(dst.fileno())
            descriptor = os.fstat(src.fileno())
    except FileExistsError as exc:
        raise CollectionError(f"refusing to overwrite archived file {destination}") from exc
    after = identity(source)
    copied_sha = digest.hexdigest()
    require(
        before == after and size == before["size"] and descriptor.st_ino == before["inode"],
        f"source changed while copying: {source}",
    )
    require(copied_sha == expected_sha, f"source hash changed after preflight: {source}")
    return {
        "source": str(source),
        "snapshot": str(destination),
        "bytes": size,
        "sha256": copied_sha,
        "source_stat_before": before,
        "source_stat_after": after,
    }
