"""Create separate immutable evidence and weights archives on approved storage."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Any

from report_state import read_json


def file_sha(path: Path) -> str:
    """Hash a file with bounded memory rather than reading tensor payloads at once."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def allowed_inputs(root: Path, paths: list[str], *, weights: bool) -> list[Path]:
    """Allow only explicit result files; exclude supplied data and private credentials."""
    result = []
    root = root.resolve(strict=True)
    suffixes = {".pt"} if weights else {".json", ".jsonl", ".log", ".txt"}
    for value in sorted(set(paths)):
        supplied = Path(value)
        if supplied.is_symlink():
            raise ValueError("Archive input cannot be a symbolic link")
        path = supplied.resolve(strict=True)
        if not path.is_file() or not path.is_relative_to(root):
            raise ValueError("Archive input is not an approved project file")
        relative = path.relative_to(root)
        if relative.parts[0] not in {"state", "runs", "exports", "grades", "runtime"}:
            raise ValueError("Archive input belongs to data, credentials or another project")
        if path.suffix not in suffixes:
            raise ValueError("Archive input has a payload type outside this archive's scope")
        if any(part in {"private", ".ssh", ".git", "datasets"} for part in relative.parts):
            raise ValueError("Archive input enters a private or supplied-data directory")
        result.append(path)
    if not result or len(result) > 100:
        raise ValueError("Archive needs a bounded nonempty explicit file list")
    return result


def _sync_directory(directory: Path) -> None:
    """Durably publish a newly linked file on the report storage filesystem."""
    if os.name == "nt":
        return
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_json_noreplace(path: Path, value: dict[str, Any]) -> None:
    """Publish a JSON receipt atomically without replacing any previous attempt."""
    descriptor, temporary_name = tempfile.mkstemp(prefix=".manifest-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        _sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _tar_index(path: Path) -> dict[str, dict[str, Any]]:
    """Read regular archive members and hash their stored bytes without extraction."""
    index: dict[str, dict[str, Any]] = {}
    with tarfile.open(path, "r") as archive:
        for member in archive.getmembers():
            name = Path(member.name)
            if (
                not member.isfile()
                or name.is_absolute()
                or ".." in name.parts
                or member.name in index
            ):
                raise ValueError("Evidence archive contains an unsafe or duplicate member")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError("Evidence archive member cannot be read")
            digest = hashlib.sha256()
            size = 0
            with stream:
                while block := stream.read(4 * 1024**2):
                    digest.update(block)
                    size += len(block)
            index[member.name] = {"bytes": size, "sha256": digest.hexdigest()}
    return index


def _input_index(root: Path, paths: list[Path]) -> dict[str, dict[str, Any]]:
    """Bind every current archive member to a relative path, size, and digest."""
    return {
        path.relative_to(root).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": file_sha(path),
        }
        for path in paths
    }


def _archive_record(root: Path, target: Path, paths: list[Path]) -> dict[str, Any]:
    """Verify an archive exactly matches its explicit current input list."""
    expected = _input_index(root, paths)
    if _tar_index(target) != expected:
        raise ValueError("Retained archive members differ from the explicit evidence list")
    return {
        "path": str(target),
        "bytes": target.stat().st_size,
        "sha256": file_sha(target),
        "files": [
            {"path": str(path), **expected[path.relative_to(root).as_posix()]} for path in paths
        ],
    }


def _create_archive(root: Path, directory: Path, kind: str, paths: list[Path]) -> dict[str, Any]:
    """Write one archive once and reconcile a crash after its no-replace publish."""
    target = directory / f"{kind}.tar"
    temporary = directory / f"{kind}.partial.tar"
    if target.is_symlink():
        raise ValueError("Archive target cannot be a symbolic link")
    if temporary.exists() or temporary.is_symlink():
        if (
            target.is_file()
            and not temporary.is_symlink()
            and os.path.samefile(target, temporary)
            and _tar_index(target) == _input_index(root, paths)
        ):
            temporary.unlink()
            _sync_directory(directory)
            return _archive_record(root, target, paths)
        raise FileExistsError(
            "Partial archive attempt is retained; manual reconciliation is required"
        )
    if target.exists():
        return _archive_record(root, target, paths)
    limit = (5 if kind == "weights" else 1) * 1024**3
    total = sum(path.stat().st_size for path in paths)
    if total > limit or shutil.disk_usage(root).free < total + 8 * 1024**3:
        raise OSError("Archive exceeds its per-result bound or physical disk reserve")
    members = []
    with tarfile.open(temporary, "x") as archive:
        for path in paths:
            before = path.stat()
            archive.add(path, arcname=str(path.relative_to(root)), recursive=False)
            digest = file_sha(path)
            after = path.stat()
            if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ):
                raise ValueError("Archive input changed during packaging")
            members.append(dict(path=str(path), bytes=after.st_size, sha256=digest))
    with temporary.open("r+b") as stream:
        os.fsync(stream.fileno())
    os.link(temporary, target)
    _sync_directory(directory)
    temporary.unlink()
    _sync_directory(directory)
    record = dict(
        path=str(target), bytes=target.stat().st_size, sha256=file_sha(target), files=members
    )
    if _tar_index(target) != _input_index(root, paths):
        raise ValueError("Published archive does not contain the exact source files")
    return record


def _ensure_kind(
    root: Path,
    directory: Path,
    kind: str,
    identity: dict[str, str],
    paths: list[Path],
) -> dict[str, Any]:
    """Reconcile one independent archive before a sibling archive is attempted."""
    target = directory / f"{kind}.tar"
    manifest_path = directory / f"{kind}.json"
    if manifest_path.exists() or manifest_path.is_symlink():
        manifest = read_json(manifest_path)
        if manifest.get("schema") != "tabdata-contract-result-archive-kind.v1":
            raise ValueError(f"Retained {kind} archive manifest has an unknown schema")
        if any(manifest.get(key) != value for key, value in identity.items()):
            raise ValueError(f"Retained {kind} archive manifest identity differs")
        record = manifest.get("archive")
        if (
            not isinstance(record, dict)
            or Path(str(record.get("path"))) != target
            or not target.is_file()
            or target.stat().st_size != record.get("bytes")
            or file_sha(target) != record.get("sha256")
            or record.get("files") != _archive_record(root, target, paths).get("files")
        ):
            raise ValueError(f"Retained {kind} archive no longer matches its receipt")
        return record

    record = _create_archive(root, directory, kind, paths)
    manifest = {
        "schema": "tabdata-contract-result-archive-kind.v1",
        **identity,
        "archive": record,
    }
    _publish_json_noreplace(manifest_path, manifest)
    return record


def ensure_archives(root: Path, row: dict) -> dict:
    """Bind one completed endpoint to durable separate archive manifests."""
    from render_report import canonical_sha

    root = root.resolve(strict=True)
    identity = canonical_sha(dict(id=row["id"], receipt_sha256=row["receipt_sha256"]))
    directory = root / "artifacts/github-reporting-v1" / identity
    manifest_path = directory / "MANIFEST.json"
    if not directory.resolve(strict=False).is_relative_to(root):
        raise ValueError("Archive directory resolves outside the approved project root")
    if directory.is_symlink():
        raise ValueError("Archive directory cannot be a symbolic link")
    directory.mkdir(parents=True, exist_ok=True)
    identity_fields = {"id": row["id"], "receipt_sha256": row["receipt_sha256"]}
    evidence = allowed_inputs(root, row.get("evidence_paths", []), weights=False)
    weights = allowed_inputs(root, row.get("weights_paths", []), weights=True)
    evidence_record = _ensure_kind(root, directory, "evidence", identity_fields, evidence)
    weights_record = _ensure_kind(root, directory, "weights", identity_fields, weights)
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if (
            manifest.get("schema") != "tabdata-contract-result-archives.v1"
            or manifest.get("receipt_sha256") != row["receipt_sha256"]
            or manifest.get("id") != row["id"]
            or manifest.get("evidence") != evidence_record
            or manifest.get("weights") != weights_record
        ):
            raise ValueError("Archived endpoint identity differs")
    else:
        manifest = dict(
            schema="tabdata-contract-result-archives.v1",
            id=row["id"],
            receipt_sha256=row["receipt_sha256"],
            evidence=evidence_record,
            weights=weights_record,
        )
        try:
            _publish_json_noreplace(manifest_path, manifest)
        except FileExistsError:
            if read_json(manifest_path) != manifest:
                raise ValueError(
                    "Competing archive manifest differs; retained for inspection"
                ) from None
    return dict(
        archive_manifest_path=str(manifest_path),
        archive_manifest_sha256=file_sha(manifest_path),
        evidence_archive={k: v for k, v in manifest["evidence"].items() if k != "files"},
        weights_archive={k: v for k, v in manifest["weights"].items() if k != "files"},
    )
