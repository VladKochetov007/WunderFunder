from __future__ import annotations

import io
import zipfile

import numpy as np
import pytest

from scripts.check_submission_zip_native_parity import (
    compare_required_rows,
    same_float32_bits,
    validate_archive,
    validate_prediction,
)


def test_callback_contract_requires_none_on_warmup_and_float32_scores() -> None:
    assert validate_prediction(None, need_prediction=False, row=0) is None
    expected = np.array([0.25, -1.5], dtype=np.float32)
    assert validate_prediction(expected, need_prediction=True, row=99) is expected

    with pytest.raises(ValueError, match="warmup row"):
        validate_prediction(expected, need_prediction=False, row=0)
    with pytest.raises(TypeError, match="NumPy array"):
        validate_prediction([0.25, -1.5], need_prediction=True, row=99)
    with pytest.raises(TypeError, match="dtype must be float32"):
        validate_prediction(expected.astype(np.float64), need_prediction=True, row=99)
    with pytest.raises(ValueError, match=r"shape must be \(2,\)"):
        validate_prediction(expected.reshape(1, 2), need_prediction=True, row=99)
    with pytest.raises(ValueError, match="non-finite"):
        validate_prediction(np.array([np.nan, 0.0], dtype=np.float32), need_prediction=True, row=99)


def test_bitwise_comparison_detects_signed_zero_even_when_max_abs_is_zero() -> None:
    expected = np.array([[0.0, 1.0], [2.0, -3.0]], dtype=np.float32)
    actual = expected.copy()
    actual[0, 0] = -0.0

    result = compare_required_rows(actual, expected)

    assert result["exact_bitwise"] is False
    assert result["different_rows"] == 1
    assert result["different_values"] == 1
    assert result["max_abs_difference"] == 0.0
    assert not same_float32_bits(actual, expected)
    assert same_float32_bits(expected, expected.copy())


def test_bitwise_comparison_accepts_identical_float32_arrays() -> None:
    expected = np.array([[0.25, -1.5]], dtype=np.float32)
    result = compare_required_rows(expected.copy(), expected)
    assert result["exact_bitwise"] is True
    assert result["different_rows"] == 0
    assert result["different_values"] == 0
    assert result["actual_sha256"] == result["expected_sha256"]


def test_archive_validation_rejects_traversal_and_symlink_entries() -> None:
    traversal = io.BytesIO()
    with zipfile.ZipFile(traversal, "w") as archive:
        archive.writestr("solution.py", "class PredictionModel: pass\n")
        archive.writestr("../escape.txt", "bad")
    with zipfile.ZipFile(traversal) as archive, pytest.raises(
        ValueError, match="unsafe ZIP entry path"
    ):
        validate_archive(archive)

    symlink = io.BytesIO()
    with zipfile.ZipFile(symlink, "w") as archive:
        archive.writestr("solution.py", "class PredictionModel: pass\n")
        info = zipfile.ZipInfo("link")
        info.create_system = 3
        info.external_attr = (0o120777 << 16)
        archive.writestr(info, "solution.py")
    with zipfile.ZipFile(symlink) as archive, pytest.raises(
        ValueError, match="not a regular file or directory"
    ):
        validate_archive(archive)


def test_bitwise_comparison_rejects_dtype_and_shape_drift() -> None:
    expected = np.zeros((3, 2), dtype=np.float32)
    with pytest.raises(TypeError, match="both be float32"):
        compare_required_rows(expected.astype(np.float64), expected)
    with pytest.raises(ValueError, match="equal shape"):
        compare_required_rows(expected[:2], expected)
