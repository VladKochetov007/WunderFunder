from __future__ import annotations

import pytest
import torch

from scripts.predict_run import family_output_slice


def test_family_slice_excludes_trailing_mean_variance_rows() -> None:
    outputs = torch.arange(49)
    columns = family_output_slice({"distill_family_weight": 0.3, "mean_variance_head": 1}, 49)
    assert columns == slice(39, 47)
    torch.testing.assert_close(outputs[columns], torch.arange(39, 47))


def test_family_slice_preserves_legacy_layout_without_variance_head() -> None:
    columns = family_output_slice({"distill_family_weight": 0.3}, 47)
    assert columns == slice(39, 47)


def test_family_slice_rejects_missing_or_malformed_family_head() -> None:
    with pytest.raises(ValueError, match="requires a checkpoint"):
        family_output_slice({"distill_family_weight": 0.0}, 47)
    with pytest.raises(ValueError, match="cannot contain"):
        family_output_slice({"distill_family_weight": 0.3, "mean_variance_head": 1}, 10)
