"""Content validation for unchanged research inputs copied to another worker."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from functools import lru_cache
from pathlib import Path

PROGRESS_EVERY_BYTES = 1 << 30


def _startup_event(event: str, **fields: object) -> None:
    """Emit opt-in startup telemetry without changing checked input bytes."""
    if os.environ.get("WNN_STARTUP_DIAGNOSTICS") != "1":
        return
    print(json.dumps({"event": event, **fields}, sort_keys=True), file=sys.stderr, flush=True)


def sha256_file(path: Path, *, startup_progress: bool = False) -> str:
    """Hash a file without materializing it; optionally report bounded progress."""
    digest = hashlib.sha256()
    total = path.stat().st_size if startup_progress else 0
    started = time.monotonic()
    bytes_read = 0
    next_progress = PROGRESS_EVERY_BYTES
    if startup_progress:
        _startup_event("portable_sha256_begin", file=path.name, bytes_total=total)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
            bytes_read += len(block)
            if startup_progress and bytes_read >= next_progress:
                _startup_event(
                    "portable_sha256_progress",
                    file=path.name,
                    bytes_total=total,
                    bytes_read=bytes_read,
                    elapsed_seconds=round(time.monotonic() - started, 3),
                )
                next_progress += PROGRESS_EVERY_BYTES
    result = digest.hexdigest()
    if startup_progress:
        _startup_event(
            "portable_sha256_complete",
            file=path.name,
            bytes_total=total,
            bytes_read=bytes_read,
            elapsed_seconds=round(time.monotonic() - started, 3),
            sha256=result,
        )
    return result


@lru_cache(maxsize=64)
def _content_digest(
    path: str, device: int, inode: int, size: int, modified_ns: int, changed_ns: int
) -> str:
    return sha256_file(Path(path), startup_progress=True)


def verify_portable_source(
    source: Path, name: str, original_identity: dict, registry: Path
) -> None:
    """Accept relocated bytes only against the controller's sealed source registry.

    The original split manifest stays unchanged. Its metadata must be reproduced
    exactly in the registry, whose own hash belongs in the experiment input spec.
    A copied file's mtime is never used as a substitute for its content digest.
    """
    payload = json.loads(registry.read_text())
    if payload.get("version") != 1:
        raise ValueError("unsupported portable input registry version")
    entry = payload["sources"].get(name)
    if entry is None or entry.get("original_identity") != original_identity:
        raise ValueError(f"portable identity disagrees with frozen split: {name}")
    stat = source.stat()
    if stat.st_size != original_identity["bytes"]:
        raise ValueError(f"portable source size changed: {name}")
    digest = _content_digest(
        str(source.resolve()),
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )
    if digest != entry.get("sha256"):
        raise ValueError(f"portable source content changed: {name}")
