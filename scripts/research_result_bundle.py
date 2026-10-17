#!/usr/bin/env python3
"""Archive explicitly selected run evidence and model weights with SHA-256s."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import tarfile
import tempfile
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import BinaryIO

CHUNK_SIZE = 1024 * 1024
ARCHIVE_NAMES = ("evidence.tar", "weights.tar")
MANIFEST_NAME = "manifest.json"


class _HashingReader:
    """Hash the exact bytes read by tarfile and reject files truncated in flight."""

    def __init__(self, source: BinaryIO, expected_size: int) -> None:
        self._source = source
        self._expected_size = expected_size
        self._remaining = expected_size
        self._digest = hashlib.sha256()

    def read(self, size: int = -1) -> bytes:
        if size == 0:
            return b""
        if self._remaining == 0:
            return b""
        requested = self._remaining if size < 0 else min(size, self._remaining)
        block = self._source.read(requested)
        if not block:
            raise OSError("Input file was truncated while it was being archived")
        self._remaining -= len(block)
        self._digest.update(block)
        return block

    @property
    def sha256(self) -> str:
        return self._digest.hexdigest()

    @property
    def complete(self) -> bool:
        return self._remaining == 0


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def _parse_relative_path(value: str | Path) -> PurePosixPath:
    """Reject absolute, traversing, and cross-platform ambiguous archive paths."""
    raw = str(value)
    path = PurePosixPath(raw)
    if (
        not raw
        or not path.parts
        or path.is_absolute()
        or "\\" in raw
        or "\x00" in raw
        or ":" in path.parts[0]
        or any(part in ("", ".", "..") for part in path.parts)
    ):
        raise ValueError(f"Expected a safe root-relative file path: {value}")
    return path


def _open_source(root_fd: int, relative_path: PurePosixPath) -> tuple[BinaryIO, os.stat_result]:
    """Open a root-relative regular file without following any symlink component."""
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise OSError("Safe bundle creation requires O_NOFOLLOW and O_DIRECTORY support")
    directory_fd = os.dup(root_fd)
    try:
        for component in relative_path.parts[:-1]:
            next_fd = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=directory_fd,
            )
            os.close(directory_fd)
            directory_fd = next_fd
        file_fd = os.open(
            relative_path.parts[-1],
            os.O_RDONLY | os.O_NOFOLLOW,
            dir_fd=directory_fd,
        )
    finally:
        os.close(directory_fd)
    file_stat = os.fstat(file_fd)
    if not stat.S_ISREG(file_stat.st_mode):
        os.close(file_fd)
        raise ValueError(f"Bundle inputs must be regular files: {relative_path}")
    return os.fdopen(file_fd, "rb"), file_stat


def _validated_paths(
    root_fd: int, values: Sequence[str | Path], category: str
) -> tuple[PurePosixPath, ...]:
    if not values:
        raise ValueError(f"At least one --{category} file is required")
    paths = tuple(_parse_relative_path(value) for value in values)
    if len({path.as_posix() for path in paths}) != len(paths):
        raise ValueError(f"Duplicate --{category} paths are not allowed")
    for path in paths:
        source, _ = _open_source(root_fd, path)
        source.close()
    return paths


def _archive_category(
    root_fd: int, paths: Sequence[PurePosixPath], staged_archive: Path
) -> list[dict[str, object]]:
    members: list[dict[str, object]] = []
    descriptor = os.open(
        staged_archive,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    with os.fdopen(descriptor, "wb") as output:
        with tarfile.open(fileobj=output, mode="w", format=tarfile.PAX_FORMAT) as archive:
            for relative_path in paths:
                source, file_stat = _open_source(root_fd, relative_path)
                with source:
                    reader = _HashingReader(source, file_stat.st_size)
                    member = tarfile.TarInfo(relative_path.as_posix())
                    member.size = file_stat.st_size
                    member.mode = stat.S_IMODE(file_stat.st_mode) & 0o777
                    member.mtime = int(file_stat.st_mtime)
                    member.uid = 0
                    member.gid = 0
                    member.uname = ""
                    member.gname = ""
                    archive.addfile(member, reader)
                    if not reader.complete:
                        raise OSError(f"Input file changed while archiving: {relative_path}")
                    members.append(
                        {
                            "path": relative_path.as_posix(),
                            "size_bytes": file_stat.st_size,
                            "mtime_unix": int(file_stat.st_mtime),
                            "mode": member.mode,
                            "sha256": reader.sha256,
                        }
                    )
        output.flush()
        os.fsync(output.fileno())
    return members


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        while block := source.read(CHUNK_SIZE):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def _write_staged_file(path: Path, content: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    with os.fdopen(descriptor, "wb") as output:
        output.write(content)
        output.flush()
        os.fsync(output.fileno())


def _ensure_targets_absent(output_dir: Path) -> None:
    for name in (*ARCHIVE_NAMES, MANIFEST_NAME):
        try:
            os.lstat(output_dir / name)
        except FileNotFoundError:
            continue
        raise FileExistsError(f"Refusing to overwrite bundle output: {output_dir / name}")


def _commit_staged_files(output_dir: Path, staged_dir: Path) -> None:
    staged_names = (*ARCHIVE_NAMES, MANIFEST_NAME)
    created: list[tuple[Path, int, int]] = []
    try:
        for name in staged_names:
            staged = staged_dir / name
            target = output_dir / name
            staged_stat = os.stat(staged, follow_symlinks=False)
            os.link(staged, target, follow_symlinks=False)
            created.append((target, staged_stat.st_dev, staged_stat.st_ino))
    except OSError:
        for path, device, inode in reversed(created):
            try:
                current = os.lstat(path)
            except FileNotFoundError:
                continue
            if current.st_dev == device and current.st_ino == inode:
                path.unlink()
        raise


def build_result_bundle(
    root: str | Path,
    output_dir: str | Path,
    evidence_files: Sequence[str | Path],
    weight_files: Sequence[str | Path],
) -> Path:
    """Create evidence and weight tarballs plus a hash manifest.

    Every input is an explicitly named regular file relative to ``root``.
    Inputs are never discovered recursively, and the three named outputs must
    not already exist in ``output_dir``.

    Args:
        root: Directory against which all input paths are resolved.
        output_dir: Destination for the two tarballs and their manifest.
        evidence_files: Explicit root-relative paths for logs and other evidence.
        weight_files: Explicit root-relative paths for model weights/checkpoints.

    Returns:
        Path to the created manifest.

    Raises:
        FileExistsError: If any named output already exists.
        OSError: If an input cannot be safely read or an output cannot be written.
        ValueError: If an input path is unsafe or not a regular file.
    """
    root_path = Path(root).resolve(strict=True)
    if not root_path.is_dir():
        raise ValueError(f"--root must be a directory: {root}")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    destination = destination.resolve(strict=True)
    if not destination.is_dir():
        raise ValueError(f"--output-dir must be a directory: {output_dir}")
    _ensure_targets_absent(destination)

    root_fd = os.open(root_path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        evidence_paths = _validated_paths(root_fd, evidence_files, "evidence")
        weight_paths = _validated_paths(root_fd, weight_files, "weight")
        overlap = {path.as_posix() for path in evidence_paths} & {
            path.as_posix() for path in weight_paths
        }
        if overlap:
            raise ValueError(f"Evidence and weight input lists overlap: {sorted(overlap)}")

        with tempfile.TemporaryDirectory(prefix=".research-bundle-", dir=destination) as temp_dir:
            staged_dir = Path(temp_dir)
            evidence_members = _archive_category(
                root_fd, evidence_paths, staged_dir / ARCHIVE_NAMES[0]
            )
            weight_members = _archive_category(root_fd, weight_paths, staged_dir / ARCHIVE_NAMES[1])
            evidence_sha, evidence_size = _sha256_file(staged_dir / ARCHIVE_NAMES[0])
            weight_sha, weight_size = _sha256_file(staged_dir / ARCHIVE_NAMES[1])
            manifest = {
                "schema_version": 1,
                "created_at_utc": datetime.now(UTC).isoformat(),
                "archives": {
                    "evidence": {
                        "file": ARCHIVE_NAMES[0],
                        "size_bytes": evidence_size,
                        "sha256": evidence_sha,
                        "members": evidence_members,
                    },
                    "weights": {
                        "file": ARCHIVE_NAMES[1],
                        "size_bytes": weight_size,
                        "sha256": weight_sha,
                        "members": weight_members,
                    },
                },
            }
            _write_staged_file(staged_dir / MANIFEST_NAME, _json_bytes(manifest))
            _commit_staged_files(destination, staged_dir)
    finally:
        os.close(root_fd)
    return destination / MANIFEST_NAME


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Root for relative input paths")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--evidence", action="append", required=True, metavar="REL_PATH", help="Evidence file"
    )
    parser.add_argument(
        "--weight", action="append", required=True, metavar="REL_PATH", help="Model weight file"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Build the requested archives and print their manifest path."""
    args = _build_parser().parse_args(argv)
    try:
        manifest_path = build_result_bundle(args.root, args.output_dir, args.evidence, args.weight)
    except (OSError, TypeError, ValueError, tarfile.TarError) as error:
        print(f"research_result_bundle: {error}", file=sys.stderr)
        return 2
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
