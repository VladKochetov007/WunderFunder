"""Hash-checked extraction and input staging for the DataSphere profile."""

from __future__ import annotations

import hashlib
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any


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


def _source_file(dataset_roots: dict[str, Path], entry: dict[str, Any]) -> Path:
    slot = entry.get("dataset_slot")
    if not isinstance(slot, str) or slot not in dataset_roots:
        raise ValueError(f"unknown dataset slot: {slot!r}")
    relative = relative_path(entry["relative_path"])
    base = dataset_roots[slot].resolve()
    path = base.joinpath(*relative.parts)
    if not path.resolve().is_relative_to(base) or not path.is_file():
        raise FileNotFoundError(f"mounted input missing or outside dataset {slot}: {relative}")
    if path.stat().st_size != int(entry["bytes"]):
        raise ValueError(f"mounted input size mismatch: {slot}/{relative}")
    return path


def assemble(
    split: str, entry: dict[str, Any], root: Path, dataset_roots: dict[str, Path]
) -> dict[str, Any]:
    target = root.joinpath(*relative_path(entry["assembled_path"]).parts)
    target.parent.mkdir(parents=True, exist_ok=True)
    total, total_bytes, parts = hashlib.sha256(), 0, []
    with target.open("xb") as output:
        for part in entry["parts"]:
            source = _source_file(dataset_roots, part)
            want_part = sealed_hash(part.get("sha256"), f"{split} part")
            digest, count = hashlib.sha256(), 0
            with source.open("rb") as stream:
                for block in iter(lambda: stream.read(8 << 20), b""):
                    digest.update(block)
                    total.update(block)
                    output.write(block)
                    count += len(block)
                    total_bytes += len(block)
            if count != int(part["bytes"]) or digest.hexdigest() != want_part:
                raise ValueError(f"mounted part content mismatch: {source.name}")
            parts.append({"path": str(source), "bytes": count, "sha256": digest.hexdigest()})
    want_hash = sealed_hash(entry.get("sha256"), f"assembled {split}")
    if total_bytes != int(entry["bytes"]) or total.hexdigest() != want_hash:
        raise ValueError(f"assembled {split} parquet identity mismatch")
    return {"path": str(target), "bytes": total_bytes, "sha256": total.hexdigest(), "parts": parts}


def copy_mounted(
    logical: str, entry: dict[str, Any], root: Path, dataset_roots: dict[str, Path]
) -> dict[str, Any]:
    source = _source_file(dataset_roots, entry)
    if source.stat().st_size > 512 * (1 << 20):
        raise ValueError(f"declared target index is unexpectedly large: {logical}")
    target = root.joinpath(*relative_path(logical).parts)
    target.parent.mkdir(parents=True, exist_ok=True)
    digest, count = hashlib.sha256(), 0
    with source.open("rb") as src, target.open("xb") as dst:
        for block in iter(lambda: src.read(8 << 20), b""):
            digest.update(block)
            dst.write(block)
            count += len(block)
    want = sealed_hash(entry.get("sha256"), logical)
    if count != int(entry["bytes"]) or digest.hexdigest() != want:
        raise ValueError(f"mounted target index identity mismatch: {logical}")
    return {
        "source": str(source),
        "path": str(target),
        "bytes": count,
        "sha256": digest.hexdigest(),
    }


def write_bundle(out_path: Path, output: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out_path, "w:gz") as archive:
        archive.add(output, arcname="profile-results")
