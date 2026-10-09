"""Synthetic tests for advised hashes and cgroup memory.stat phase receipts."""

from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
import os
import sys
import tarfile
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from profile_worker import (
    MEMORY_STAT_MAX_BYTES,
    ProfileAbort,
    _hash_with_memory_stat,
    _read_memory_stat_file,
    install_loader_portable_hash_advice,
)

PACKET_ROOT = Path(__file__).resolve().parents[1]


def test_oversized_memory_stat_reader_is_bounded_and_returns_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stat_path = tmp_path / "memory.stat"
    stat_path.write_bytes(b"x" * (MEMORY_STAT_MAX_BYTES + 4096))
    original_open = Path.open
    read_sizes: list[int] = []

    class ReadSpy:
        def __init__(self, stream: Any) -> None:
            self.stream = stream

        def __enter__(self) -> ReadSpy:
            return self

        def __exit__(self, *_exc: object) -> None:
            self.stream.close()

        def read(self, size: int = -1) -> bytes:
            read_sizes.append(size)
            return self.stream.read(size)

    def tracked_open(path: Path, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        stream = original_open(path, mode, *args, **kwargs)
        return ReadSpy(stream) if path == stat_path and mode == "rb" else stream

    monkeypatch.setattr(Path, "open", tracked_open)
    result = _read_memory_stat_file(stat_path)

    assert read_sizes == [MEMORY_STAT_MAX_BYTES + 1]
    assert result["status"] == "unavailable"
    assert result["bytes_observed"] == MEMORY_STAT_MAX_BYTES + 1
    assert result["diagnostic"] == f"memory.stat exceeds {MEMORY_STAT_MAX_BYTES} bytes"


def _load_pinned_portability(source_root: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Extract and import only the pinned standard-library verifier module."""
    module_path = source_root / "src" / "wnn" / "portability.py"
    module_path.parent.mkdir(parents=True)
    with tarfile.open(PACKET_ROOT / "source_archive.tar", "r") as archive:
        source = archive.extractfile("src/wnn/portability.py").read()
    module_path.write_bytes(source)
    spec = importlib.util.spec_from_file_location("e520_v6_test_portability", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not import pinned portability module")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def _memory_snapshot(counter: itertools.count) -> dict[str, Any]:
    value = next(counter)
    return {
        "status": "available",
        "memory_current_bytes": 100 + value,
        "memory_limit_bytes": 1_000,
        "memory_available_bytes": 900 - value,
        "memory_stat": {"file": value, "anon": 200},
    }


def test_portable_hash_advice_preserves_pinned_digest_and_identity_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "source"
    raw_root = source_root / "datasets"
    raw_root.mkdir(parents=True)
    hash_receipt = tmp_path / "receipts" / "hash_io_advice.json"
    memory_receipt = tmp_path / "receipts" / "memory_stat_phases.json"
    snapshots = itertools.count()
    monkeypatch.setattr(
        "profile_worker.cgroup_memory_stat_snapshot", lambda: _memory_snapshot(snapshots)
    )

    advice_calls: list[tuple[int, int, int, int]] = []
    page_size = os.sysconf("SC_PAGE_SIZE")

    def record_advice(fd: int, offset: int, length: int, advice: int) -> None:
        assert offset % page_size == 0
        assert length > 0 and length % page_size == 0
        advice_calls.append((fd, offset, length, advice))

    monkeypatch.setattr("profile_worker.os.posix_fadvise", record_advice, raising=False)
    monkeypatch.setattr("profile_worker.os.POSIX_FADV_DONTNEED", 4, raising=False)

    contents: dict[str, bytes] = {}
    identities: dict[str, dict[str, Any]] = {}
    registry_sources: dict[str, dict[str, Any]] = {}
    for split_name in ("train", "valid"):
        logical = f"datasets/{split_name}.parquet"
        content = (f"synthetic-{split_name}-".encode() * 90_000) + b"tail"
        source = raw_root / f"{split_name}.parquet"
        source.write_bytes(content)
        stat = source.stat()
        original_identity = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        identities[logical] = original_identity
        registry_sources[logical] = {
            "original_identity": original_identity,
            "sha256": hashlib.sha256(content).hexdigest(),
        }
        contents[logical] = content

        observed = _hash_with_memory_stat(
            source,
            hash_receipt_path=hash_receipt,
            memory_stat_receipt_path=memory_receipt,
            logical_name=f"initial:{logical}",
            diagnostic_phase=f"initial_raw_hash:{logical}",
        )
        assert observed == hashlib.sha256(content).hexdigest()

    portability = _load_pinned_portability(source_root, monkeypatch)
    original_file_sha256 = portability.sha256_file
    patch_identity = install_loader_portable_hash_advice(
        source_root,
        hash_receipt,
        memory_receipt,
        portability_module=portability,
    )
    assert patch_identity["patched_symbol"] == "wnn.portability._content_digest"
    assert patch_identity["raw_source_names"] == [
        "datasets/train.parquet",
        "datasets/valid.parquet",
    ]
    assert portability.sha256_file is original_file_sha256

    registry = tmp_path / "portable_inputs.json"
    registry.write_text(json.dumps({"version": 1, "sources": registry_sources}), encoding="utf-8")
    for logical, original_identity in identities.items():
        source = source_root / logical
        portability.verify_portable_source(source, logical, original_identity, registry)

    hash_payload = json.loads(hash_receipt.read_text(encoding="utf-8"))
    assert set(hash_payload["files"]) == {
        "initial:datasets/train.parquet",
        "initial:datasets/valid.parquet",
        "loader_portable:datasets/train.parquet",
        "loader_portable:datasets/valid.parquet",
    }
    for logical, content in contents.items():
        record = hash_payload["files"][f"loader_portable:{logical}"]
        assert record["status"] == "complete"
        assert record["sha256"] == hashlib.sha256(content).hexdigest()
        assert record["bytes_hashed"] == len(content)
        assert record["bytes_advised"] <= len(content)
    assert advice_calls

    memory_payload = json.loads(memory_receipt.read_text(encoding="utf-8"))
    for logical in identities:
        for phase_prefix in ("initial_raw_hash", "loader_portable_content_digest"):
            record = memory_payload["phases"][f"{phase_prefix}:{logical}"]
            assert record["before"]["snapshot"]["status"] == "available"
            assert record["after"]["snapshot"]["status"] == "available"
            assert (
                record["after"]["snapshot"]["memory_stat"]["file"]
                > record["before"]["snapshot"]["memory_stat"]["file"]
            )

    bad_registry = dict(registry_sources)
    bad_registry["datasets/train.parquet"] = {
        **bad_registry["datasets/train.parquet"],
        "sha256": "0" * 64,
    }
    registry.write_text(json.dumps({"version": 1, "sources": bad_registry}), encoding="utf-8")
    with pytest.raises(ValueError, match="portable source content changed"):
        portability.verify_portable_source(
            source_root / "datasets/train.parquet",
            "datasets/train.parquet",
            identities["datasets/train.parquet"],
            registry,
        )

    bad_identity = {**identities["datasets/valid.parquet"], "bytes": -1}
    with pytest.raises(ValueError, match="portable identity disagrees"):
        portability.verify_portable_source(
            source_root / "datasets/valid.parquet",
            "datasets/valid.parquet",
            bad_identity,
            registry,
        )


def test_portable_hash_advice_failure_aborts_without_plain_hash_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "source"
    raw_root = source_root / "datasets"
    raw_root.mkdir(parents=True)
    source = raw_root / "train.parquet"
    source.write_bytes(b"synthetic raw parquet" * 100_000)
    valid_source = raw_root / "valid.parquet"
    valid_source.write_bytes(b"synthetic valid parquet")
    portability = _load_pinned_portability(source_root, monkeypatch)
    install_loader_portable_hash_advice(
        source_root,
        tmp_path / "receipts" / "hash_io_advice.json",
        tmp_path / "receipts" / "memory_stat_phases.json",
        portability_module=portability,
    )
    snapshots = itertools.count()
    monkeypatch.setattr(
        "profile_worker.cgroup_memory_stat_snapshot", lambda: _memory_snapshot(snapshots)
    )

    def fail_advice(_fd: int, _offset: int, _length: int, _advice: int) -> None:
        raise OSError("synthetic fadvise failure")

    monkeypatch.setattr("profile_worker.os.posix_fadvise", fail_advice, raising=False)
    monkeypatch.setattr("profile_worker.os.POSIX_FADV_DONTNEED", 4, raising=False)
    stat = source.stat()
    registry = tmp_path / "portable_inputs.json"
    registry.write_text(
        json.dumps(
            {
                "version": 1,
                "sources": {
                    "datasets/train.parquet": {
                        "original_identity": {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns},
                        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ProfileAbort, match="page-cache advice/hash failed"):
        portability.verify_portable_source(
            source,
            "datasets/train.parquet",
            {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns},
            registry,
        )
    io_record = json.loads(
        (tmp_path / "receipts" / "hash_io_advice.json").read_text(encoding="utf-8")
    )["files"]["loader_portable:datasets/train.parquet"]
    assert io_record["status"] == "failed"
    assert io_record["advice_failures"] == 1
    memory_record = json.loads(
        (tmp_path / "receipts" / "memory_stat_phases.json").read_text(encoding="utf-8")
    )["phases"]["loader_portable_content_digest:datasets/train.parquet"]
    assert memory_record["details"]["status"] == "failed"
    assert "synthetic fadvise failure" in memory_record["details"]["diagnostic"]
