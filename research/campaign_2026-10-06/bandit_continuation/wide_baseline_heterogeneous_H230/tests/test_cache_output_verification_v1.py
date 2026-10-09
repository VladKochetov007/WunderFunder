"""Focused checks for bounded H230 cache output verification."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "verify_h230_cache_outputs_v1.py"
SPEC = importlib.util.spec_from_file_location("h230_cache_output_verifier", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load the H230 cache output verifier")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
inspect_npy_file = MODULE.inspect_npy_file


def test_npy_stream_reports_exact_hash_shape_and_finite_count(tmp_path: Path) -> None:
    """The streaming verifier checks payload bytes without materializing a copy."""
    path = tmp_path / "sample.npy"
    expected = np.arange(30, dtype=np.float32).reshape(3, 5, 2)
    np.save(path, expected)

    result = inspect_npy_file(path, (3, 5, 2))

    assert result["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert result["shape"] == [3, 5, 2]
    assert result["dtype"] == "float32"
    assert result["finite_values"] == expected.size
    assert result["fadvise_calls"] > 0


def test_npy_stream_rejects_wrong_shape(tmp_path: Path) -> None:
    """Expected output shape is enforced before scanning payload values."""
    path = tmp_path / "sample.npy"
    np.save(path, np.zeros((3, 5, 2), dtype=np.float32))

    with pytest.raises(ValueError, match="shape mismatch"):
        inspect_npy_file(path, (4, 5, 2))


def test_npy_stream_rejects_nonfinite_values(tmp_path: Path) -> None:
    """Nonfinite target values fail the cache gate."""
    path = tmp_path / "sample.npy"
    values = np.zeros((3, 5, 2), dtype=np.float32)
    values[1, 2, 1] = np.nan
    np.save(path, values)

    with pytest.raises(ValueError, match="nonfinite"):
        inspect_npy_file(path, (3, 5, 2))


def test_npy_stream_fails_closed_when_pagecache_advice_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Advice failure is surfaced instead of continuing without resource control."""
    path = tmp_path / "sample.npy"
    np.save(path, np.zeros((3, 5, 2), dtype=np.float32))

    def fail_advice(*args: object) -> None:
        raise OSError("synthetic fadvise failure")

    monkeypatch.setattr("os.posix_fadvise", fail_advice)
    with pytest.raises(OSError, match="synthetic fadvise failure"):
        inspect_npy_file(path, (3, 5, 2))
