"""Hash-checked extraction and input staging for the DataSphere profile."""

from __future__ import annotations

import hashlib
import os
import stat
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

MAX_MOUNT_ENTRIES = 250_000
MAX_MOUNT_DEPTH = 16
MAX_TARGET_INDEX_BYTES = int(2.1 * (1 << 30))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"unsafe relative input path: {value!r}")
    return path


def sealed_hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"unsealed SHA-256 for {label}")
    try:
        bytes.fromhex(value)
    except ValueError as error:
        raise ValueError(f"invalid SHA-256 for {label}") from error
    if value.lower() != value:
        raise ValueError(f"SHA-256 must be lowercase for {label}")
    return value


def safe_extract(
    archive_path: Path, destination: Path, expected: dict[str, str] | None = None
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    seen: set[str] = set()
    with tarfile.open(archive_path, "r:*") as archive:
        for member in archive.getmembers():
            raw_name = member.name.rstrip("/")
            relative = relative_path(raw_name) if raw_name else None
            if relative is None:
                if not member.isdir():
                    raise ValueError("archive has an invalid root member")
                continue
            target = destination.joinpath(*relative.parts)
            if not target.resolve().is_relative_to(root):
                raise ValueError(f"archive member escapes destination: {member.name}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise ValueError(f"non-regular archive member rejected: {member.name}")
            name = relative.as_posix()
            if expected is not None and name not in expected:
                raise ValueError(f"unlisted supplemental archive file: {name}")
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"archive member has no payload: {name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            with source, target.open("xb") as output:
                while block := source.read(8 << 20):
                    digest.update(block)
                    output.write(block)
            if expected is not None and digest.hexdigest() != sealed_hash(expected[name], name):
                raise ValueError(f"supplemental member hash mismatch: {name}")
            seen.add(name)
    if expected is not None and seen != set(expected):
        missing = sorted(set(expected) - seen)
        raise ValueError(f"supplemental archive member set differs: missing {missing}")


def discover_mounted_files(
    dataset_roots: dict[str, Path], expected_names: set[str]
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, int]]]:
    """Resolve exact basenames across mounts using bounded directory metadata only.

    Symlinks are ignored, and a name is accepted only when it occurs exactly
    once across all mounts. No file payload is opened during this pass.
    """
    if not expected_names or any("/" in name or "\\" in name for name in expected_names):
        raise ValueError("mounted input discovery requires exact basenames")
    resolved: dict[str, dict[str, Any]] = {}
    scans: dict[str, dict[str, int]] = {}
    entries_seen = 0
    for slot, root_arg in dataset_roots.items():
        root = root_arg.resolve(strict=True)
        if not root.is_dir():
            raise NotADirectoryError(f"dataset {slot} root is not a directory")
        stack: list[tuple[Path, PurePosixPath]] = [(root, PurePosixPath())]
        directories = 0
        files = 0
        skipped_symlinks = 0
        mount_entries = 0
        while stack:
            directory, relative_directory = stack.pop()
            directories += 1
            with os.scandir(directory) as iterator:
                for item in iterator:
                    entries_seen += 1
                    mount_entries += 1
                    if entries_seen > MAX_MOUNT_ENTRIES:
                        raise ValueError("mounted dataset metadata scan exceeded its entry bound")
                    relative = relative_directory / item.name
                    if item.is_symlink():
                        skipped_symlinks += 1
                        if item.name in expected_names:
                            raise ValueError(
                                f"expected mounted input is a symlink: {slot}/{relative}"
                            )
                        continue
                    if item.is_dir(follow_symlinks=False):
                        if len(relative.parts) > MAX_MOUNT_DEPTH:
                            raise ValueError(
                                "mounted dataset metadata scan exceeded its depth bound"
                            )
                        stack.append((Path(item.path), relative))
                        continue
                    if not item.is_file(follow_symlinks=False):
                        continue
                    files += 1
                    if item.name not in expected_names:
                        continue
                    info = item.stat(follow_symlinks=False)
                    if not stat.S_ISREG(info.st_mode):
                        raise ValueError(
                            f"expected mounted input is not a regular file: {slot}/{relative}"
                        )
                    path = Path(item.path)
                    if not path.resolve(strict=True).is_relative_to(root):
                        raise ValueError(f"mounted input escapes dataset {slot}: {relative}")
                    name = item.name
                    if name in resolved:
                        previous = resolved[name]
                        raise ValueError(
                            f"duplicate mounted basename {name!r} in {previous['dataset_slot']} and {slot}"
                        )
                    resolved[name] = {
                        "dataset_slot": slot,
                        "relative_path": relative.as_posix(),
                        "bytes": info.st_size,
                        "path": str(path),
                        "root": str(root),
                    }
        scans[slot] = {
            "entries": mount_entries,
            "directories": directories,
            "regular_files": files,
            "skipped_symlinks": skipped_symlinks,
        }
    missing = sorted(expected_names - set(resolved))
    if missing:
        raise FileNotFoundError(f"mounted inputs were not found by exact basename: {missing}")
    return resolved, scans


def _source_file(discovered: dict[str, dict[str, Any]], filename: str) -> Path:
    if filename not in discovered:
        raise FileNotFoundError(f"mounted input was not discovered: {filename}")
    item = discovered[filename]
    path = Path(item["path"])
    base = Path(item["root"]).resolve(strict=True)
    relative = relative_path(item["relative_path"])
    expected_path = base.joinpath(*relative.parts)
    if path != expected_path or not path.resolve(strict=True).is_relative_to(base):
        raise ValueError(f"mounted input escapes its dataset: {filename}")
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != int(item["bytes"]):
        raise ValueError(f"mounted input metadata changed after discovery: {filename}")
    if path.name != relative.name:
        raise ValueError(f"mounted input basename changed after discovery: {filename}")
    return path


def assemble(
    split: str, entry: dict[str, Any], root: Path, discovered: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    target = root.joinpath(*relative_path(entry["assembled_path"]).parts)
    target.parent.mkdir(parents=True, exist_ok=True)
    total, total_bytes, parts = hashlib.sha256(), 0, []
    with target.open("xb") as output:
        for part in entry["parts"]:
            filename = part["filename"]
            source = _source_file(discovered, filename)
            digest, count = hashlib.sha256(), 0
            with source.open("rb") as stream:
                for block in iter(lambda: stream.read(8 << 20), b""):
                    digest.update(block)
                    total.update(block)
                    output.write(block)
                    count += len(block)
                    total_bytes += len(block)
            if count != int(discovered[filename]["bytes"]):
                raise ValueError(f"mounted part size changed during assembly: {source.name}")
            parts.append(
                {
                    "filename": filename,
                    "dataset_slot": discovered[filename]["dataset_slot"],
                    "relative_path": discovered[filename]["relative_path"],
                    "bytes": count,
                    "sha256": digest.hexdigest(),
                }
            )
    want_hash = sealed_hash(entry.get("sha256"), f"assembled {split}")
    expected_bytes = sum(int(discovered[part["filename"]]["bytes"]) for part in entry["parts"])
    if (
        total_bytes != expected_bytes
        or total_bytes != int(entry["bytes"])
        or total.hexdigest() != want_hash
    ):
        raise ValueError(f"assembled {split} parquet identity mismatch")
    return {
        "path": str(target),
        "bytes": total_bytes,
        "sha256": total.hexdigest(),
        "parts": parts,
    }


def copy_mounted(
    logical: str, entry: dict[str, Any], root: Path, discovered: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    filename = entry["source_filename"]
    source = _source_file(discovered, filename)
    expected_bytes = int(entry["bytes"])
    if expected_bytes > MAX_TARGET_INDEX_BYTES:
        raise ValueError(f"target index exceeds the bounded staging cap: {logical}")
    if int(discovered[filename]["bytes"]) != expected_bytes:
        raise ValueError(f"mounted target index size mismatch: {logical}")
    target = root.joinpath(*relative_path(logical).parts)
    target.parent.mkdir(parents=True, exist_ok=True)
    digest, count = hashlib.sha256(), 0
    with source.open("rb") as src, target.open("xb") as dst:
        for block in iter(lambda: src.read(8 << 20), b""):
            digest.update(block)
            dst.write(block)
            count += len(block)
    want = sealed_hash(entry.get("sha256"), logical)
    if count != expected_bytes or digest.hexdigest() != want:
        raise ValueError(f"mounted target index identity mismatch: {logical}")
    return {
        "source": str(source),
        "source_filename": filename,
        "dataset_slot": discovered[filename]["dataset_slot"],
        "relative_path": discovered[filename]["relative_path"],
        "path": str(target),
        "bytes": count,
        "sha256": digest.hexdigest(),
    }


def write_bundle(out_path: Path, output: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out_path, "w:gz") as archive:
        archive.add(output, arcname="profile-results")
