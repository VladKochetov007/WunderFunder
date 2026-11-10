"""Focused CPU-only gates for the causal-stem readout receiver."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

MODULE_PATH = Path(__file__).with_name("readout_stem_r1.py")
SPEC = importlib.util.spec_from_file_location("readout_stem_r1_fixture", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
receiver = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = receiver
SPEC.loader.exec_module(receiver)


def test_interpreter_path_keeps_venv_spelling(tmp_path: Path) -> None:
    """The command path stays lexical even when its filesystem entry is a symlink."""
    venv = tmp_path / "venv" / "bin"
    venv.mkdir(parents=True)
    candidate = venv / "python"
    with patch.object(Path, "is_file", return_value=True), patch("os.access", return_value=True), patch.object(
        Path, "resolve", side_effect=AssertionError("must not dereference venv executable")
    ):
        actual = receiver.lexical_executable({}, candidate)
    assert str(actual) == os.path.abspath(os.fspath(candidate))


def test_fit_result_cutoff_checks_timestamp_and_atomic_file_mtime(tmp_path: Path) -> None:
    result_path = tmp_path / "cell-result.json"
    result_path.write_text("{}", encoding="utf-8")
    os.utime(result_path, (99.0, 99.0))
    receiver.validate_fit_result_time(result_path, {"at": 99.0}, 100.0)
    with pytest.raises(TimeoutError, match="after the original fit cutoff"):
        receiver.validate_fit_result_time(result_path, {"at": 100.1}, 100.0)
    os.utime(result_path, (101.0, 101.0))
    with pytest.raises(TimeoutError, match="after the original fit cutoff"):
        receiver.validate_fit_result_time(result_path, {"at": 99.0}, 100.0)


def test_fit_result_rejects_nonfinite_timestamp(tmp_path: Path) -> None:
    result_path = tmp_path / "cell-result.json"
    result_path.write_text("{}", encoding="utf-8")
    with pytest.raises(TimeoutError):
        receiver.validate_fit_result_time(result_path, {"at": float("nan")}, 100.0)


def test_contrasts_pair_only_matching_seeds_and_expected_directions() -> None:
    scores = {}
    for cell, (seed, kernel, _) in receiver.SPEC.items():
        arm_delta = {0: 0.0, 1: 0.01, 16: 0.04}[kernel]
        scores[cell] = {
            "within_tolerance": True,
            "t0": seed / 1_000_000 + arm_delta,
            "t1": seed / 2_000_000 + arm_delta * 2,
            "wp": seed / 1_500_000 + arm_delta * 1.5,
        }
    rows = receiver.derive_contrasts(scores)
    assert [(row["seed"], row["contrast"]) for row in rows] == [
        (203301, "K16 minus k1"),
        (203301, "K16 minus plain"),
        (203302, "K16 minus k1"),
        (203302, "K16 minus plain"),
    ]
    for row in rows:
        scale = 1 if row["contrast"] == "K16 minus k1" else 4 / 3
        assert row["t0"] == pytest.approx(0.03 * scale)
        assert row["t1"] == pytest.approx(0.06 * scale)
        assert row["wp"] == pytest.approx(0.045 * scale)


def test_contrasts_withhold_incomplete_or_mismatched_scores() -> None:
    scores = {cell: {"within_tolerance": True, "t0": 0.1, "t1": 0.2, "wp": 0.15} for cell in receiver.CELLS}
    scores["k16_s203302"]["within_tolerance"] = False
    with pytest.raises(ValueError, match="all six"):
        receiver.derive_contrasts(scores)
