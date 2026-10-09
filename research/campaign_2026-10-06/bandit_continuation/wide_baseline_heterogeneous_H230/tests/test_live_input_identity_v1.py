"""Focused checks for the bounded H230 live-input hasher."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "verify_h230_live_inputs_v1.py"
MODULE_SPEC = importlib.util.spec_from_file_location("h230_live_identity", MODULE_PATH)
if MODULE_SPEC is None or MODULE_SPEC.loader is None:
    raise RuntimeError(f"cannot load H230 live identity module: {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = MODULE
MODULE_SPEC.loader.exec_module(MODULE)
IdentityError = MODULE.IdentityError
hash_with_page_advice = MODULE.hash_with_page_advice


def test_streamed_hash_matches_exact_bytes_and_advises_aligned_ranges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "payload.bin"
    payload = bytes(range(251)) * 431
    path.write_bytes(payload)
    page_size = int(os.sysconf("SC_PAGE_SIZE"))
    advice: list[tuple[int, int, int]] = []

    def record_advice(fd: int, offset: int, length: int, advice_kind: int) -> None:
        assert fd >= 0
        advice.append((offset, length, advice_kind))

    monkeypatch.setattr(os, "posix_fadvise", record_advice)
    progress: dict[str, object] = {}
    checkpoints: list[dict[str, object]] = []

    actual = hash_with_page_advice(path, progress, lambda: checkpoints.append(progress.copy()))

    assert actual == hashlib.sha256(payload).hexdigest()
    assert progress["bytes_hashed"] == len(payload)
    assert progress["bytes_advised"] == len(payload) - len(payload) % page_size
    assert progress["status"] == "hashed"
    assert checkpoints
    assert sum(length for _, length, _ in advice) == progress["bytes_advised"]
    assert all(offset % page_size == 0 and length % page_size == 0 for offset, length, _ in advice)


def test_page_advice_failure_aborts_without_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "payload.bin"
    path.write_bytes(b"identity" * 10_000)
    progress: dict[str, object] = {}

    def fail_advice(*args: object) -> None:
        raise OSError("injected fadvise error")

    monkeypatch.setattr(os, "posix_fadvise", fail_advice)

    with pytest.raises(IdentityError, match="hash/advice failed"):
        hash_with_page_advice(path, progress, lambda: None)

    assert progress["status"] == "hash_failed"
    assert progress["bytes_hashed"] == path.stat().st_size


def test_file_mutation_during_hash_is_reported_with_observed_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "payload.bin"
    original = b"before" * 20_000
    path.write_bytes(original)
    mutated = False

    def mutate_after_read(fd: int, offset: int, length: int, advice_kind: int) -> None:
        nonlocal mutated
        assert fd >= 0 and offset >= 0 and length >= 0 and advice_kind >= 0
        if not mutated:
            with path.open("ab") as stream:
                stream.write(b"after")
            mutated = True

    monkeypatch.setattr(os, "posix_fadvise", mutate_after_read)
    progress: dict[str, object] = {}
    checkpoints: list[dict[str, object]] = []

    with pytest.raises(IdentityError, match="changed during hash"):
        hash_with_page_advice(path, progress, lambda: checkpoints.append(progress.copy()))

    assert progress["observed_sha256"] == hashlib.sha256(original + b"after").hexdigest()
    assert progress["file_before"] != progress["file_after"]
    assert checkpoints
