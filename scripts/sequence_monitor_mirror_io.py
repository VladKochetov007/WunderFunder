from __future__ import annotations

import base64
import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path

from scripts.collect_sequence_monitor import CHUNK_BYTES, FILES, RUN_ID


def atomic_json(path: Path, value: object) -> None:
    """Publish a local controller observation without exposing partial JSON."""
    atomic_bytes(path, (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode())


def atomic_bytes(path: Path, payload: bytes) -> None:
    """Replace one direct local mirror file through a same-directory temporary."""
    if path.is_symlink():
        raise ValueError("mirror file cannot be a symlink")
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _run_directory(root: Path, run_id: str, owner: dict[str, str]) -> Path:
    if not RUN_ID.fullmatch(run_id) or ".." in run_id or root.is_symlink():
        raise ValueError("unsafe host-qualified mirror run identifier")
    root.mkdir(parents=True, exist_ok=True)
    directory = root / run_id
    if directory.is_symlink():
        raise ValueError("mirror run cannot be a symlink")
    directory.mkdir(exist_ok=True)
    ownership = directory / ".sequence_mirror_owner.json"
    if ownership.exists():
        if ownership.is_symlink() or json.loads(ownership.read_bytes()) != owner:
            raise ValueError("existing run directory belongs to another source")
    else:
        atomic_json(ownership, owner)
    return directory


def _archive_journal(path: Path) -> None:
    """Retain an old nonempty local prefix before an actual producer reset."""
    if not path.exists() or not path.stat().st_size:
        return
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK_BYTES), b""):
            digest.update(block)
    archived = path.with_name(f"{path.name}.previous-{digest.hexdigest()[:16]}")
    if archived.is_symlink():
        raise ValueError("journal archive cannot be a symlink")
    if not archived.exists():
        os.replace(path, archived)


def _apply_chunk(directory: Path, chunk: dict[str, object], previous: dict[str, object]) -> bool:
    name, offset, cursor = chunk.get("name"), chunk.get("offset"), chunk.get("cursor")
    if name not in FILES or type(offset) is not int or offset < 0 or not isinstance(cursor, dict):
        raise ValueError("malformed mirror chunk identity")
    payload = base64.b64decode(str(chunk["data_base64"]), validate=True)
    if len(payload) > CHUNK_BYTES or len(payload) != chunk["bytes"]:
        raise ValueError("mirror payload exceeds or differs from its declared bound")
    if hashlib.sha256(payload).hexdigest() != chunk["sha256"]:
        raise ValueError("mirror payload digest mismatch")
    if cursor.get("offset") != offset + len(payload):
        raise ValueError("mirror cursor does not match its exact source-byte range")
    journal = name.endswith(".jsonl")
    path = directory / name
    target = path if journal else directory / f".{name}.partial"
    if path.is_symlink() or target.is_symlink():
        raise ValueError("mirror target cannot be a symlink")
    replay = previous == cursor
    if chunk["reset"] and not replay:
        if journal:
            _archive_journal(path)
        atomic_bytes(target, b"")
    if not journal and not target.exists() and not payload and chunk["eof"] and path.exists():
        return True
    if not target.exists():
        if offset:
            raise ValueError("mirror is missing its source prefix")
        atomic_bytes(target, b"")
    if not stat.S_ISREG(target.lstat().st_mode):
        raise ValueError("mirror target must be regular")
    with target.open("r+b") as stream:
        stream.seek(0, 2)
        size = stream.tell()
        if size == offset:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        elif size >= offset + len(payload):
            stream.seek(offset)
            if stream.read(len(payload)) != payload:
                raise ValueError("mirror replay differs from already copied source bytes")
        else:
            raise ValueError("mirror byte position differs from the source cursor")
    if not journal and chunk["eof"]:
        if target.stat().st_size > 16 * 1_048_576:
            raise ValueError("mirror JSON exceeds its 16 MiB bound")
        json.loads(target.read_bytes())
        os.replace(target, path)
    return journal or bool(chunk["eof"])


def apply_snapshot(
    snapshot: dict[str, object],
    dashboard_root: Path,
    native_root: Path,
    cursors: dict[str, object],
    expected_host: str,
) -> dict[str, object]:
    """Apply verified source deltas, preserving native bytes and producer timestamps."""
    if (
        snapshot.get("schema") != "wnn.sequence-monitor-rpc.v1"
        or snapshot.get("host") != expected_host
    ):
        raise ValueError("unexpected monitor snapshot schema or host")
    updated = dict(cursors)
    for run in snapshot["runs"]:
        run_id = run["run_id"]
        if not run_id.startswith(f"{expected_host}-") or run.get("host") != expected_host:
            raise ValueError("run is not qualified by its declared source host")
        owner = {"host": expected_host, "source_directory": run["source_directory"]}
        native = _run_directory(native_root, run_id, owner)
        view = _run_directory(dashboard_root, run_id, owner)
        for chunk in run["files"]:
            key = f"{run_id}/{chunk['name']}"
            previous = updated.get(key, {})
            previous = previous if isinstance(previous, dict) else {}
            complete = _apply_chunk(native, chunk, previous)
            _apply_chunk(view, chunk, previous)
            updated[key] = chunk["cursor"]
            if complete:
                source = native / chunk["name"]
                timestamp = chunk["source_mtime_ns"]
                os.utime(source, ns=(timestamp, timestamp))
                mirrored = view / chunk["name"]
                if mirrored.exists():
                    os.utime(mirrored, ns=(timestamp, timestamp))
        atomic_json(
            native / "collection_receipt.json",
            {
                "host": expected_host,
                "source_directory": run["source_directory"],
                "collected_at_unix": snapshot["collected_at_unix"],
                "files": {
                    key: value for key, value in updated.items() if key.startswith(f"{run_id}/")
                },
            },
        )
    return updated
