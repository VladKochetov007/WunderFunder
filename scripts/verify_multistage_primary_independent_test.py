from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_multistage_primary_independent as independent  # noqa: E402


def test_independent_clipped_covariance_and_raw_prediction_wmse() -> None:
    target = np.asarray([[1.0, -1.0], [2.0, -2.0], [-1.0, 1.0]], np.float32)
    prediction = np.asarray([[3.0, -3.0], [1.0, -1.0], [-3.0, 3.0]], np.float32)

    result = independent.independent_metrics(target, prediction)

    assert result["wp"] == pytest.approx(6.0 / np.sqrt(54.0), abs=1e-15)
    assert result["t0"] == result["t1"]
    assert result["wmse"] == result["wmse_t0"] == result["wmse_t1"] == 2.5


def test_independent_metric_zero_weight_constant_and_nonfinite_cases() -> None:
    zeros = np.zeros((3, 2), np.float32)
    assert independent.independent_metrics(zeros, np.ones_like(zeros))["wp"] == 0.0
    assert independent.independent_metrics(np.ones_like(zeros), zeros)["wp"] == 0.0
    bad = zeros.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        independent.independent_metrics(zeros, bad)


def test_independent_blend_casts_once_after_raw_prediction_weighting() -> None:
    members = [np.asarray([[3.0, -3.0]], np.float32), np.asarray([[1.0, 1.0]], np.float32)]
    actual = independent.reconstruct_block(members, np.asarray([0.5, 0.5]))
    np.testing.assert_array_equal(actual, np.asarray([[2.0, -1.0]], np.float32))
    with pytest.raises(ValueError, match="simplex"):
        independent.reconstruct_block(members, np.asarray([0.3, 0.3]))


def test_independent_prediction_reader_rejects_changed_bytes_shape_and_dtype(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(independent, "ROWS", 2)
    path = tmp_path / "prediction.npy"
    np.save(path, np.zeros((2, 2), np.float32))
    assert independent._array(path, independent.sha256(path)).shape == (2, 2)
    with pytest.raises(ValueError, match="receipt"):
        independent._array(path, "wrong")
    np.save(path, np.zeros((2, 2), np.float64))
    with pytest.raises(ValueError, match="dtype, shape"):
        independent._array(path, independent.sha256(path))


def test_independent_comparison_rejects_error_above_declared_tolerance() -> None:
    actual = {"wp": 0.5, "selected_rows": independent.ROWS}
    assert independent._compare(actual, {"wp": 0.5 + 1e-12}) < 1e-10
    with pytest.raises(ValueError, match="beyond1e-10"):
        independent._compare(actual, {"wp": 0.5 + 1e-9})
