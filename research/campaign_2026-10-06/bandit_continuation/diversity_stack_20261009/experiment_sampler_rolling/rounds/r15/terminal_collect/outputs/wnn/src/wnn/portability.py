"""Content validation for unchanged research inputs copied to another worker."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Callable


def sha256_file(path: Path) -> str:
    """Hash a file without materializing its contents in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=64)
def _content_digest(
    path: str, device: int, inode: int, size: int, modified_ns: int, changed_ns: int
) -> str:
    return sha256_file(Path(path))


def verify_portable_source(
    source: Path,
    name: str,
    original_identity: dict,
    registry: Path,
    *,
    observation_callback: Callable[[int, str | None], None] | None = None,
) -> str:
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
        if observation_callback is not None:
            observation_callback(stat.st_size, None)
        raise ValueError(f"portable source size changed: {name}")
    digest = _content_digest(
        str(source.resolve()),
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )
    if observation_callback is not None:
        observation_callback(stat.st_size, digest)
    if digest != entry.get("sha256"):
        raise ValueError(f"portable source content changed: {name}")
    return digest
