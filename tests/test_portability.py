from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from wnn.portability import verify_portable_source


def registry_for(tmp_path: Path, data: bytes) -> tuple[Path, Path, dict]:
    source = tmp_path / "data.bin"
    source.write_bytes(data)
    original = {"bytes": len(data), "mtime_ns": 12345}
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "version": 1,
                "sources": {
                    "data.bin": {
                        "original_identity": original,
                        "sha256": hashlib.sha256(data).hexdigest(),
                    }
                },
            }
        )
    )
    return source, registry, original


def test_relocated_content_ignores_only_mtime(tmp_path: Path) -> None:
    source, registry, original = registry_for(tmp_path, b"same data")
    verify_portable_source(source, "data.bin", original, registry)
    os.utime(source, ns=(99999, 99999))
    verify_portable_source(source, "data.bin", original, registry)


def test_same_size_tampering_is_rejected(tmp_path: Path) -> None:
    source, registry, original = registry_for(tmp_path, b"same data")
    verify_portable_source(source, "data.bin", original, registry)
    timestamp = source.stat().st_mtime_ns
    source.write_bytes(b"evil data")
    os.utime(source, ns=(timestamp, timestamp))
    with pytest.raises(ValueError, match="content changed"):
        verify_portable_source(source, "data.bin", original, registry)


def test_registry_cannot_relabel_frozen_identity(tmp_path: Path) -> None:
    source, registry, original = registry_for(tmp_path, b"same data")
    with pytest.raises(ValueError, match="disagrees with frozen split"):
        verify_portable_source(source, "data.bin", original | {"mtime_ns": 99}, registry)
