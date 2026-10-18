"""Integrity manifest and bounded archive helpers for the F1 output packet."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

from profile_staging import sha256_file

# Manifest data uses Any because it is parsed from producer-generated JSON.
MANIFEST_NAME = "fit-artifact-manifest.json"
MAX_MANIFEST_BYTES = 16 << 20
MAX_PROVIDER_OUTPUT_BYTES = 1 << 30
MAX_BUNDLE_CONTENT_BYTES = 4 << 30


def write_bundle(out_path: Path, output: Path) -> None:
    """Hash and verify every artifact before bounded terminal-output packaging."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() or out_path.resolve().is_relative_to(output.resolve()):
        raise FileExistsError("bundle output already exists or would archive itself")
    manifest = write_hash_manifest(output)
    if manifest["total_bytes"] > MAX_BUNDLE_CONTENT_BYTES:
        raise ValueError("bundle contents exceed the 4 GiB collection safety bound")
    verify_hash_manifest(output)
    try:
        with tarfile.open(out_path, "w:gz", compresslevel=1) as archive:
            archive.add(output, arcname="fit-results")
        if out_path.stat().st_size > MAX_PROVIDER_OUTPUT_BYTES:
            raise ValueError(
                "compressed fit bundle exceeds the DataSphere 1 GiB aggregate output cap"
            )
        verify_bundle_archive(out_path)
    except BaseException:
        out_path.unlink(missing_ok=True)
        raise


def write_hash_manifest(root: Path) -> dict[str, Any]:
    """Record each regular bundle file by relative path, size, and SHA-256."""
    manifest_path = root / MANIFEST_NAME
    files = []
    for path in _regular_files(root):
        if path == manifest_path:
            continue
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    total_bytes = sum(entry["bytes"] for entry in files)
    if total_bytes > MAX_BUNDLE_CONTENT_BYTES:
        raise ValueError("bundle contents exceed the 4 GiB collection safety bound")
    manifest = {
        "schema_version": 1,
        "hash_algorithm": "sha256",
        "root": "fit-results",
        "manifest_excluded_from_file_list": True,
        "file_count": len(files),
        "total_bytes": total_bytes,
        "files": files,
    }
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if temporary.stat().st_size > MAX_MANIFEST_BYTES:
        temporary.unlink()
        raise ValueError("bundle hash manifest exceeds the 16 MiB metadata bound")
    temporary.replace(manifest_path)
    return manifest


def verify_hash_manifest(root: Path) -> dict[str, Any]:
    """Verify a downloaded/extracted bundle tree against its embedded file hashes."""
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError(f"bundle is missing regular {MANIFEST_NAME}")
    if manifest_path.stat().st_size > MAX_MANIFEST_BYTES:
        raise ValueError("bundle hash manifest exceeds the 16 MiB metadata bound")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = _validate_manifest(manifest)
    actual_files = {
        path.relative_to(root).as_posix(): path
        for path in _regular_files(root)
        if path != manifest_path
    }
    if set(actual_files) != set(entries):
        raise ValueError("bundle tree paths differ from its hash manifest")
    for relative, expected in entries.items():
        path = actual_files[relative]
        if path.stat().st_size != expected["bytes"] or sha256_file(path) != expected["sha256"]:
            raise ValueError(f"bundle file hash or size mismatch: {relative}")
    if sum(int(item["bytes"]) for item in entries.values()) != manifest["total_bytes"]:
        raise ValueError("bundle manifest total byte count is inconsistent")
    return manifest


def verify_bundle_archive(archive_path: Path) -> dict[str, Any]:
    """Verify every regular archive file against the embedded manifest."""
    if archive_path.stat().st_size > MAX_PROVIDER_OUTPUT_BYTES:
        raise ValueError("bundle archive exceeds the DataSphere 1 GiB aggregate output cap")
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        if sum(member.size for member in members if member.isfile()) > MAX_BUNDLE_CONTENT_BYTES:
            raise ValueError("bundle archive contents exceed the 4 GiB collection safety bound")
        by_name: dict[str, tarfile.TarInfo] = {}
        for member in members:
            if not (member.isfile() or member.isdir()):
                raise ValueError(f"bundle archive contains a non-regular entry: {member.name}")
            normalized = PurePosixPath(member.name)
            if (
                "\\" in member.name
                or ".." in normalized.parts
                or not normalized.parts
                or normalized.parts[0] != "fit-results"
            ):
                raise ValueError(f"bundle archive path is unsafe: {member.name}")
            if member.isfile():
                if member.name in by_name:
                    raise ValueError(f"duplicate bundle archive member: {member.name}")
                by_name[member.name] = member
        manifest_name = f"fit-results/{MANIFEST_NAME}"
        manifest_member = by_name.get(manifest_name)
        if manifest_member is None:
            raise ValueError(f"bundle archive is missing {MANIFEST_NAME}")
        if manifest_member.size > MAX_MANIFEST_BYTES:
            raise ValueError("bundle hash manifest exceeds the 16 MiB metadata bound")
        source = archive.extractfile(manifest_member)
        if source is None:
            raise ValueError("bundle manifest could not be read from archive")
        with source:
            manifest = json.load(source)
        entries = _validate_manifest(manifest)
        expected_names = {manifest_name, *(f"fit-results/{name}" for name in entries)}
        if set(by_name) != expected_names:
            raise ValueError("bundle archive paths differ from its hash manifest")
        for relative, expected in entries.items():
            member = by_name[f"fit-results/{relative}"]
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"bundle member could not be read: {relative}")
            with source:
                digest, byte_count = _hash_stream(source)
            if byte_count != expected["bytes"] or digest != expected["sha256"]:
                raise ValueError(f"bundle archive hash or size mismatch: {relative}")
    if archive_path.stat().st_size > MAX_PROVIDER_OUTPUT_BYTES:
        raise ValueError("bundle archive exceeds the DataSphere 1 GiB aggregate output cap")
    return manifest


def _validate_manifest(value: object) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict) or (
        type(value.get("schema_version")) is not int
        or value.get("schema_version") != 1
        or value.get("hash_algorithm") != "sha256"
        or value.get("root") != "fit-results"
        or value.get("manifest_excluded_from_file_list") is not True
        or not isinstance(value.get("files"), list)
    ):
        raise ValueError("bundle hash manifest has an unsupported schema")
    entries: dict[str, dict[str, Any]] = {}
    for item in value["files"]:
        if not isinstance(item, dict):
            raise ValueError("bundle hash manifest entries must be objects")
        relative = item.get("path")
        size = item.get("bytes")
        digest = item.get("sha256")
        path = PurePosixPath(relative) if isinstance(relative, str) else None
        if (
            path is None
            or not relative
            or "\\" in relative
            or path.is_absolute()
            or path.as_posix() != relative
            or any(part in ("", ".", "..") for part in path.parts)
            or relative == MANIFEST_NAME
            or type(size) is not int
            or size < 0
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or relative in entries
        ):
            raise ValueError("bundle hash manifest contains an invalid or duplicate file entry")
        entries[relative] = {"bytes": size, "sha256": digest}
    if type(value.get("file_count")) is not int or value["file_count"] != len(entries):
        raise ValueError("bundle manifest file count is inconsistent")
    if (
        type(value.get("total_bytes")) is not int
        or value["total_bytes"] < 0
        or value["total_bytes"] > MAX_BUNDLE_CONTENT_BYTES
    ):
        raise ValueError("bundle manifest total byte count is invalid")
    if sum(item["bytes"] for item in entries.values()) != value["total_bytes"]:
        raise ValueError("bundle manifest total byte count is inconsistent")
    return entries


def _regular_files(root: Path) -> list[Path]:
    result = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"bundle tree contains a symlink: {path.relative_to(root)}")
        if path.is_file():
            result.append(path)
    return result


def _hash_stream(stream: BinaryIO) -> tuple[str, int]:
    digest = hashlib.sha256()
    byte_count = 0
    for block in iter(lambda: stream.read(8 << 20), b""):
        digest.update(block)
        byte_count += len(block)
    return digest.hexdigest(), byte_count


def _verify_cli() -> int:
    parser = argparse.ArgumentParser(description="Verify a collected F1 fit bundle.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--verify-tree", type=Path)
    group.add_argument("--verify-bundle", type=Path)
    args = parser.parse_args()
    result = (
        verify_hash_manifest(args.verify_tree)
        if args.verify_tree
        else verify_bundle_archive(args.verify_bundle)
    )
    print(
        json.dumps(
            {
                "verified": True,
                "file_count": result["file_count"],
                "total_bytes": result["total_bytes"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_verify_cli())
