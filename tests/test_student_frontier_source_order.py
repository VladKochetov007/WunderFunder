from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wnn.student_frontier_order import (  # noqa: E402
    seeded_epoch_positions,
    source_qualified_members,
    source_qualified_target_positions,
)


ACTUAL_F0_FIRST16_POSITIONS = [
    10562, 8725, 6305, 1586, 2134, 11477, 8128, 1560,
    2922, 8340, 4913, 2805, 4517, 9259, 2632, 2203,
]
ACTUAL_F0_FIRST16_MEMBERS = [
    ("valid", 37),
    ("train", 9839),
    ("train", 1639),
    ("train", 7977),
    ("train", 7548),
    ("valid", 1263),
    ("train", 3788),
    ("train", 9480),
    ("train", 2478),
    ("train", 1094),
    ("train", 7529),
    ("train", 10227),
    ("train", 2131),
    ("train", 9134),
    ("train", 1832),
    ("train", 5224),
]


def test_frozen_f0_sampler_prefix_is_source_qualified_despite_overlapping_37() -> None:
    positions = seeded_epoch_positions(11_940, seed=3, batch_size=16, epoch=0)
    assert positions[:16] == ACTUAL_F0_FIRST16_POSITIONS

    # The observed first batch has an actual valid row group 37. Add the same
    # numeric ID to train to ensure the mapper uses the dataset offset, not an
    # ambiguous membership lookup.
    train_order = [-index - 1 for index in range(10_535)]
    valid_groups = [-index - 20_000 for index in range(1_405)]
    train_order[0] = 37
    for position, (source, group) in zip(positions[:16], ACTUAL_F0_FIRST16_MEMBERS, strict=True):
        if source == "train":
            train_order[position] = group
        else:
            valid_groups[position - len(train_order)] = group

    actual = source_qualified_members(train_order, valid_groups, positions[:16])
    assert actual == ACTUAL_F0_FIRST16_MEMBERS


def test_every_cache_row_maps_to_its_source_even_when_ids_overlap() -> None:
    positions = source_qualified_target_positions(
        train_target_groups=[37, 5],
        valid_target_groups=[37, 9],
        expected_train_groups=[37, 5],
        expected_valid_groups=[37, 9],
    )
    assert positions == {("train", 37): 0, ("train", 5): 1, ("valid", 37): 0, ("valid", 9): 1}


def test_cache_row_mapping_rejects_missing_or_duplicate_source_rows() -> None:
    with pytest.raises(ValueError, match="preserve the full frozen fit order"):
        source_qualified_target_positions([5], [37], [37, 5], [37])
    with pytest.raises(ValueError, match="duplicate source-qualified target row"):
        source_qualified_target_positions([37, 37], [37], [37, 37], [37])
