"""Focused safety checks for E520 raw-source digest reuse."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import ModuleType

import pytest
from support import input_identity
from support.verified_digest_handoff import (
    DigestMismatchError,
    VerifiedDigestHandoff,
)

from wnn import portability


def _counting_portability_module() -> tuple[ModuleType, dict[str, int]]:
    """Return a small verifier substitute that counts fallback bytes."""
    module = ModuleType("synthetic_wnn_portability")
    counters = {"calls": 0, "bytes_read": 0}

    def content_digest(
        path: str,
        _device: int,
        _inode: int,
        _size: int,
        _modified_ns: int,
        _changed_ns: int,
    ) -> str:
        counters["calls"] += 1
        digest = hashlib.sha256()
        with Path(path).open("rb") as stream:
            while block := stream.read(1 << 20):
                counters["bytes_read"] += len(block)
                digest.update(block)
        return digest.hexdigest()

    module._content_digest = content_digest
    return module, counters


def _verify(module: ModuleType, path: Path, expected_sha256: str) -> str:
    """Exercise the exact portability callback signature and digest comparison."""
    stat_result = path.stat()
    digest = module._content_digest(
        str(path.resolve()),
        stat_result.st_dev,
        stat_result.st_ino,
        stat_result.st_size,
        stat_result.st_mtime_ns,
        stat_result.st_ctime_ns,
    )
    if digest != expected_sha256:
        raise ValueError("portable source content changed")
    return digest


def test_raw_hash_advice_and_portable_reuse_preserve_exact_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The initial pass advises pages and later verification reuses those bytes."""
    path = tmp_path / "train.parquet"
    payload = (b"verified-parquet-row-group\0" * 80_000) + b"tail"
    path.write_bytes(payload)
    advised: list[tuple[int, int, int]] = []

    def fake_fadvise(file_descriptor: int, offset: int, length: int, advice: int) -> None:
        advised.append((offset, length, advice))

    monkeypatch.setattr(input_identity.os, "posix_fadvise", fake_fadvise, raising=False)
    monkeypatch.setattr(input_identity.os, "POSIX_FADV_DONTNEED", 4, raising=False)
    expected = hashlib.sha256(payload).hexdigest()
    handoff = VerifiedDigestHandoff((path,))
    receipt_path = tmp_path / "hash_io_advice.json"

    digest_receipt = input_identity.hash_raw_with_handoff(
        path, receipt_path, "datasets/train.parquet", handoff
    )
    handoff.authorize(
        digest_receipt,
        expected_sha256=expected,
        registry_sha256=expected,
    )
    module, fallback = _counting_portability_module()
    handoff.install_portable_verifier(module)

    assert _verify(module, path, expected) == expected
    first_metrics = handoff.metrics_for(path)
    assert _verify(module, path, expected) == expected
    advice = json.loads(receipt_path.read_text())["files"]["datasets/train.parquet"]
    page_size = os.sysconf("SC_PAGE_SIZE")
    assert digest_receipt.sha256 == expected
    assert advice["status"] == "complete"
    assert advice["bytes_hashed"] == len(payload)
    assert advice["bytes_advised"] == len(payload) - (len(payload) % page_size)
    assert advised
    assert all(offset % page_size == 0 and length % page_size == 0 for offset, length, _ in advised)
    assert handoff.metrics_for(path)["digest_reuse_hits"] == 2
    assert fallback == {"calls": 0, "bytes_read": 0}
    runtime = input_identity.digest_handoff_runtime_report(
        {
            "verified_digest_handoff": {
                "datasets/train.parquet": {
                    "loader_verification": {
                        **first_metrics,
                        "first_split_load_comparable_bytes_avoided": len(payload),
                    }
                }
            }
        },
        handoff,
        {"train": path},
        {"train": digest_receipt},
        phase="synthetic_after_seed3",
    )
    train_runtime = runtime["sources"]["datasets/train.parquet"]
    assert train_runtime["first_load_baseline_duplicate_bytes_avoided"] == len(payload)
    assert train_runtime["post_initial_split_load_counters"]["digest_reuse_hits"] == 1
    assert train_runtime["all_handoff_hit_bytes"] == 2 * len(payload)


def test_digest_authorization_requires_campaign_and_registry_agreement(tmp_path: Path) -> None:
    """A correct byte hash is still unusable if either frozen digest disagrees."""
    path = tmp_path / "valid.parquet"
    path.write_bytes(b"exact source identity")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    handoff = VerifiedDigestHandoff((path,))
    receipt = handoff.hash_file(path)

    with pytest.raises(DigestMismatchError, match="disagrees"):
        handoff.authorize(
            receipt,
            expected_sha256=digest,
            registry_sha256="0" * 64,
        )
    assert (
        handoff.lookup(
            str(path),
            device=receipt.identity.device,
            inode=receipt.identity.inode,
            size=receipt.identity.size,
            modified_ns=receipt.identity.modified_ns,
            changed_ns=receipt.identity.changed_ns,
        )
        is None
    )


def test_stat_mutation_forces_pinned_verifier_fallback(tmp_path: Path) -> None:
    """Changed input bytes cannot reuse a stale digest, even with restored mtime."""
    path = tmp_path / "valid.parquet"
    original_bytes = b"a" * 1_100_000
    changed_bytes = b"b" * len(original_bytes)
    path.write_bytes(original_bytes)
    expected = hashlib.sha256(original_bytes).hexdigest()
    handoff = VerifiedDigestHandoff((path,))
    receipt = handoff.hash_file(path)
    handoff.authorize(receipt, expected_sha256=expected, registry_sha256=expected)
    module, fallback = _counting_portability_module()
    handoff.install_portable_verifier(module)

    before = path.stat()
    path.write_bytes(changed_bytes)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    after = path.stat()
    assert after.st_size == before.st_size
    assert after.st_mtime_ns == before.st_mtime_ns
    assert after.st_ctime_ns != receipt.identity.changed_ns

    with pytest.raises(ValueError, match="content changed"):
        _verify(module, path, expected)
    assert handoff.metrics_for(path)["digest_reuse_misses"] == 1
    assert handoff.metrics_for(path)["fallback_digest_calls"] == 1
    assert fallback == {"calls": 1, "bytes_read": len(changed_bytes)}


def test_pinned_portability_registry_checks_remain_before_digest_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real pinned verifier retains metadata checks around the wrapped digest."""
    path = tmp_path / "train.parquet"
    payload = b"portable raw bytes" * 100
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    handoff = VerifiedDigestHandoff((path,))
    receipt = handoff.hash_file(path)
    handoff.authorize(receipt, expected_sha256=digest, registry_sha256=digest)
    original_digest = portability._content_digest
    monkeypatch.setattr(portability, "_content_digest", original_digest)
    handoff.install_portable_verifier(portability)
    stat_result = path.stat()
    original_identity = {"bytes": len(payload), "mtime_ns": stat_result.st_mtime_ns}
    registry = tmp_path / "portable_inputs.json"
    registry.write_text(
        json.dumps(
            {
                "version": 1,
                "sources": {
                    "datasets/train.parquet": {
                        "original_identity": original_identity,
                        "sha256": digest,
                    }
                },
            }
        )
    )

    portability.verify_portable_source(path, "datasets/train.parquet", original_identity, registry)
    assert handoff.metrics_for(path)["digest_reuse_hits"] == 1

    with pytest.raises(ValueError, match="portable identity disagrees"):
        portability.verify_portable_source(
            path,
            "datasets/train.parquet",
            {"bytes": len(payload), "mtime_ns": stat_result.st_mtime_ns + 1},
            registry,
        )
