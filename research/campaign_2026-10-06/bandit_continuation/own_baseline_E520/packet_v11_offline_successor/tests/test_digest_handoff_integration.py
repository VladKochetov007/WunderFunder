"""Focused integration check for the profile worker's pinned-loader handoff."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import ModuleType

import profile_worker
from pytest import MonkeyPatch
from verified_digest_handoff import VerifiedDigestHandoff


def _portable_args(path: Path) -> tuple[str, int, int, int, int, int]:
    current = path.stat()
    return (
        str(path.resolve()),
        current.st_dev,
        current.st_ino,
        current.st_size,
        current.st_mtime_ns,
        current.st_ctime_ns,
    )


def test_worker_installs_exact_digest_handoff_and_falls_back_on_mutation(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    source_root = tmp_path / "source"
    module_path = source_root / "src" / "wnn" / "portability.py"
    module_path.parent.mkdir(parents=True)
    module_path.write_text("# synthetic pinned module identity\n", encoding="utf-8")
    module = ModuleType("wnn.portability")
    module.__file__ = str(module_path)
    fallback = {"calls": 0, "bytes": 0}

    def original_digest(
        path: str,
        _device: int,
        _inode: int,
        _size: int,
        _modified_ns: int,
        _changed_ns: int,
    ) -> str:
        fallback["calls"] += 1
        payload = Path(path).read_bytes()
        fallback["bytes"] += len(payload)
        return hashlib.sha256(payload).hexdigest()

    module._content_digest = original_digest
    module_hash = hashlib.sha256(module_path.read_bytes()).hexdigest()
    monkeypatch.setattr(profile_worker, "PINNED_PORTABILITY_SOURCE_SHA256", module_hash)

    raw_path = tmp_path / "mount" / "train.parquet"
    raw_path.parent.mkdir()
    initial = b"verified raw input"
    raw_path.write_bytes(initial)
    digest = hashlib.sha256(initial).hexdigest()
    handoff = VerifiedDigestHandoff((raw_path,))
    receipt = handoff.hash_file(raw_path)
    handoff.authorize(receipt, expected_sha256=digest, registry_sha256=digest)

    installed = profile_worker.install_loader_verified_digest_handoff(
        source_root,
        handoff,
        portability_module=module,
    )
    assert installed["module_sha256"] == module_hash
    assert module._content_digest(*_portable_args(raw_path)) == digest
    assert handoff.metrics_for(raw_path) == {
        "verifier_lookups": 1,
        "digest_reuse_hits": 1,
        "digest_reuse_misses": 0,
        "fallback_digest_calls": 0,
    }
    assert fallback == {"calls": 0, "bytes": 0}

    raw_path.write_bytes(b"mutated raw input")
    changed_digest = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    assert module._content_digest(*_portable_args(raw_path)) == changed_digest
    assert fallback == {"calls": 1, "bytes": len(b"mutated raw input")}
    assert handoff.metrics_for(raw_path)["fallback_digest_calls"] == 1
