from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from wnn.chronological_splits import _footer_identity, _verify_source


def _parquet_like_bytes(payload: bytes, footer: bytes) -> bytes:
    """Build the bounded trailer shape inspected by the source identity checker."""
    return payload + footer + len(footer).to_bytes(4, "little") + b"PAR1"


def _write_registry(
    registry: Path, source_name: str, source: Path, identity: dict[str, int | str]
) -> None:
    registry.write_text(
        json.dumps(
            {
                "version": 1,
                "sources": {
                    source_name: {
                        "original_identity": identity,
                        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    }
                },
            }
        )
    )


def test_portable_parquet_copy_ignores_mtime_only_after_full_hash(
    tmp_path: Path, monkeypatch
) -> None:
    source_name = "datasets/train.parquet"
    source = tmp_path / source_name
    source.parent.mkdir()
    source.write_bytes(_parquet_like_bytes(b"row payload", b"original footer"))
    identity = _footer_identity(source)
    registry = tmp_path / "portable.json"
    _write_registry(registry, source_name, source, identity)
    monkeypatch.setenv("WNN_PORTABLE_DATA_MANIFEST", str(registry))

    original_mtime = source.stat().st_mtime_ns
    os.utime(source, ns=(original_mtime + 10_000, original_mtime + 10_000))
    _verify_source(tmp_path, source_name, identity)

    source.write_bytes(_parquet_like_bytes(b"bad payload", b"original footer"))
    with pytest.raises(ValueError, match="portable source content changed"):
        _verify_source(tmp_path, source_name, identity)


def test_nonportable_parquet_identity_still_requires_original_mtime(
    tmp_path: Path, monkeypatch
) -> None:
    source_name = "datasets/train.parquet"
    source = tmp_path / source_name
    source.parent.mkdir()
    source.write_bytes(_parquet_like_bytes(b"row payload", b"original footer"))
    identity = _footer_identity(source)
    monkeypatch.delenv("WNN_PORTABLE_DATA_MANIFEST", raising=False)

    original_mtime = source.stat().st_mtime_ns
    os.utime(source, ns=(original_mtime + 10_000, original_mtime + 10_000))
    with pytest.raises(ValueError, match="chronological primary source changed"):
        _verify_source(tmp_path, source_name, identity)
