from __future__ import annotations

import numpy as np
import pytest

from scripts.l4_teacher_capacity_20261104.stage2_stream_common import (
    fit_source_positions,
    positions_sha256,
)


def test_full_source_prediction_selects_only_official_fit_groups_in_split_order() -> None:
    source_order = [90, 10, 50, 30, 70]
    fit_order = [30, 90, 10]
    bank = np.asarray([[group, -group] for group in source_order], dtype=np.float32)

    positions = fit_source_positions(source_order, fit_order)
    selected = bank[positions]

    assert positions.tolist() == [3, 0, 1]
    np.testing.assert_array_equal(selected, [[30, -30], [90, -90], [10, -10]])
    assert positions_sha256(positions) == positions_sha256(np.asarray([3, 0, 1], dtype="<i8"))


@pytest.mark.parametrize(
    ("source_order", "fit_order"),
    [([1, 2, 2], [1]), ([1, 2, 3], [4])],
)
def test_source_mapping_rejects_duplicate_or_missing_row_groups(
    source_order: list[int], fit_order: list[int]
) -> None:
    with pytest.raises(ValueError):
        fit_source_positions(source_order, fit_order)
