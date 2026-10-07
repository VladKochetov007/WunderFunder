"""Safety tests for exact-stat, worker-local source-digest reuse."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from types import ModuleType

import pytest
from verified_digest_handoff import (
    DigestHandoffError,
    DigestMismatchError,
    SourceChangedError,
    VerifiedDigestHandoff,
)


def _install_counting_verifier() -> tuple[ModuleType, dict[str, int]]:
    module = ModuleType("synthetic_portability")
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


def _verified_consumer(module: ModuleType, path: Path, expected: str) -> str:
    stat_result = path.stat()
    observed = module._content_digest(
        str(path.resolve()),
        stat_result.st_dev,
        stat_result.st_ino,
        stat_result.st_size,
        stat_result.st_mtime_ns,
        stat_result.st_ctime_ns,
    )
    if observed != expected:
        raise ValueError("portable source content changed")
    return observed


def _new_handoff(path: Path) -> VerifiedDigestHandoff:
    return VerifiedDigestHandoff((path,))


def test_authorized_exact_digest_skips_one_redundant_full_read(tmp_path: Path) -> None:
    path = tmp_path / "train.parquet"
    payload = (b"row-group-payload\0" * 200_000) + b"tail"
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    progress_bytes: list[int] = []
    handoff = _new_handoff(path)
    receipt = handoff.hash_file(path, lambda _fd, size: progress_bytes.append(size))
    assert receipt.bytes_hashed == len(payload)
    assert progress_bytes[-1] == len(payload)
    handoff.authorize(receipt, expected_sha256=digest, registry_sha256=digest)
    module, fallback = _install_counting_verifier()
    handoff.install_portable_verifier(module)

    assert _verified_consumer(module, path, digest) == digest
    assert _verified_consumer(module, path, digest) == digest
    assert fallback == {"calls": 0, "bytes_read": 0}


def test_mutation_invalidates_handoff_and_registry_checks_fresh_bytes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "valid.parquet"
    original_bytes = b"a" * 1_400_000
    changed_bytes = b"b" * len(original_bytes)
    path.write_bytes(original_bytes)
    expected = hashlib.sha256(original_bytes).hexdigest()
    handoff = _new_handoff(path)
    receipt = handoff.hash_file(path)
    handoff.authorize(receipt, expected_sha256=expected, registry_sha256=expected)
    module, fallback = _install_counting_verifier()
    handoff.install_portable_verifier(module)

    before = path.stat()
    path.write_bytes(changed_bytes)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    after = path.stat()
    assert after.st_size == before.st_size
    assert after.st_mtime_ns == before.st_mtime_ns
    assert after.st_ctime_ns != receipt.identity.changed_ns

    with pytest.raises(ValueError, match="content changed"):
        _verified_consumer(module, path, expected)
    assert fallback == {"calls": 1, "bytes_read": len(changed_bytes)}


def test_authorization_requires_expected_and_registry_digest_agreement(
    tmp_path: Path,
) -> None:
    path = tmp_path / "train.parquet"
    path.write_bytes(b"identity-bound bytes")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    handoff = _new_handoff(path)
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


def test_authorization_fails_closed_if_source_disappears_after_hash(tmp_path: Path) -> None:
    path = tmp_path / "valid.parquet"
    path.write_bytes(b"verified before unlink")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    handoff = _new_handoff(path)
    receipt = handoff.hash_file(path)
    path.unlink()

    with pytest.raises(SourceChangedError, match="became unavailable"):
        handoff.authorize(
            receipt,
            expected_sha256=digest,
            registry_sha256=digest,
        )


def test_hash_rejects_mutation_during_the_initial_read(tmp_path: Path) -> None:
    path = tmp_path / "train.parquet"
    path.write_bytes(b"x" * 1_500_000)
    handoff = _new_handoff(path)
    changed = False

    def mutate_after_first_chunk(_fd: int, bytes_hashed: int) -> None:
        nonlocal changed
        if not changed and bytes_hashed >= 1 << 20:
            changed = True
            path.write_bytes(b"y" * 1_500_000)

    with pytest.raises(SourceChangedError, match="changed"):
        handoff.hash_file(path, mutate_after_first_chunk)
    assert changed


def test_paths_outside_exact_allowlist_are_not_cached(tmp_path: Path) -> None:
    allowed = tmp_path / "train.parquet"
    other = tmp_path / "valid.parquet"
    allowed.write_bytes(b"train")
    other.write_bytes(b"valid")
    handoff = _new_handoff(allowed)

    with pytest.raises(DigestHandoffError, match="outside handoff scope"):
        handoff.hash_file(other)
