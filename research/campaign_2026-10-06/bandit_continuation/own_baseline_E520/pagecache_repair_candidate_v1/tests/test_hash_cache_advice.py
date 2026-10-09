"""Synthetic tests for fail-closed page-cache-safe input hashing."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
from profile_worker_candidate import CAPS, ProfileAbort, sha256_file


def _write_synthetic_file(path: Path, size_bytes: int) -> bytes:
    """Write repeatable content in small bounded chunks and return no file-sized buffer."""
    digest = hashlib.sha256()
    seed = bytes(range(256)) * 256
    remaining = size_bytes
    with path.open("wb") as stream:
        while remaining:
            block = seed[: min(len(seed), remaining)]
            stream.write(block)
            digest.update(block)
            remaining -= len(block)
    return digest.digest()


def test_profile_resource_caps_remain_unchanged() -> None:
    assert CAPS == {
        "rss_bytes": 8_000_000_000,
        "cuda_allocated_bytes": 2_000_000_000,
        "cuda_reserved_bytes": 2_500_000_000,
        "gpu_process_bytes": 3_000_000_000,
        "headroom_bytes": 4_294_967_296,
        "headroom_fraction": 0.20,
        "disk_free_bytes": 8_000_000_000,
        "visible_gpus": 1,
    }


def test_streamed_hash_matches_reference_and_records_page_aligned_advice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "synthetic.parquet"
    size_bytes = (2 << 20) + 123
    expected = _write_synthetic_file(source, size_bytes)
    calls: list[tuple[int, int, int, int]] = []

    def record_advice(fd: int, offset: int, length: int, advice: int) -> None:
        calls.append((fd, offset, length, advice))

    monkeypatch.setattr(os, "posix_fadvise", record_advice)
    receipt_path = tmp_path / "hash_io_advice.json"
    observed = sha256_file(
        source,
        pagecache_receipt_path=receipt_path,
        logical_name="datasets/synthetic.parquet",
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    record = receipt["files"]["datasets/synthetic.parquet"]
    page_size = int(os.sysconf("SC_PAGE_SIZE"))

    assert observed == expected.hex()
    assert record["status"] == "complete"
    assert record["bytes_hashed"] == size_bytes
    assert record["sha256"] == expected.hex()
    assert record["bytes_advised"] == size_bytes - (size_bytes % page_size)
    assert len(calls) == record["advice_calls"]
    assert all(
        offset % page_size == 0 and length % page_size == 0 for _, offset, length, _ in calls
    )
    assert all(advice == os.POSIX_FADV_DONTNEED for _, _, _, advice in calls)
    assert not receipt_path.with_name(receipt_path.name + ".partial").exists()


def test_advice_error_is_durably_recorded_and_aborts_without_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "synthetic.parquet"
    _write_synthetic_file(source, 2 << 20)

    def fail_advice(fd: int, offset: int, length: int, advice: int) -> None:
        raise OSError("injected fadvise failure")

    monkeypatch.setattr(os, "posix_fadvise", fail_advice)
    receipt_path = tmp_path / "hash_io_advice.json"
    with pytest.raises(ProfileAbort, match="page-cache advice/hash failed"):
        sha256_file(
            source,
            pagecache_receipt_path=receipt_path,
            logical_name="datasets/synthetic.parquet",
        )

    record = json.loads(receipt_path.read_text(encoding="utf-8"))["files"][
        "datasets/synthetic.parquet"
    ]
    assert record["status"] == "failed"
    assert record["advice_failures"] == 1
    assert record["bytes_hashed"] == 1 << 20
    assert "injected fadvise failure" in record["diagnostic"]


def test_missing_advice_support_fails_before_read_and_records_capability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "synthetic.parquet"
    _write_synthetic_file(source, 8_192)
    monkeypatch.delattr(os, "posix_fadvise", raising=False)
    receipt_path = tmp_path / "hash_io_advice.json"

    with pytest.raises(ProfileAbort, match="advice is unavailable"):
        sha256_file(
            source,
            pagecache_receipt_path=receipt_path,
            logical_name="datasets/synthetic.parquet",
        )

    record = json.loads(receipt_path.read_text(encoding="utf-8"))["files"][
        "datasets/synthetic.parquet"
    ]
    assert record["status"] == "unsupported"
    assert record["bytes_hashed"] == 0
    assert record["posix_fadvise_available"] is False
