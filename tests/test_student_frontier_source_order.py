from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wnn.student_frontier import teacher_source_matches_contract  # noqa: E402
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

# Extracted from the real clean-F0 producer metadata.json, whose pinned SHA256
# is 1e4c5c4915b643352494558efc17996a11900224231405fa4ab05698f2be3dc8.
# The producer stores absolute checkpoint_path/provenance_path fields; these
# relative paths are the exact values after removing the project-root prefix.
ACTUAL_PRODUCER_TEACHER_SOURCE = {
    "family": "strong",
    "run": "clean_t3_f0_muonall_h384_e3_s3_r5",
    "seed": 3,
    "epoch": 2,
    "coefficient": 0.1,
    "checkpoint_path": "artifacts/runs/clean_t3_f0_muonall_h384_e3_s3_r5/model_ep2.pt",
    "checkpoint_sha256": "201221c6b4140dda70b078db2a9d7a7246ec40b813f054570d195cecefc61582",
    "provenance_path": "artifacts/runs/clean_t3_f0_muonall_h384_e3_s3_r5/provenance.json",
    "provenance_sha256": "740b0d96ee311f215153db62272262e0153ce11f8f91560b03efae8e739f9ed0",
}

CONTRACT_TEACHER_SOURCE = {
    "family": "strong",
    "run": "clean_t3_f0_muonall_h384_e3_s3_r5",
    "seed": 3,
    "epoch": 2,
    "coefficient": 0.1,
    "checkpoint": "artifacts/runs/clean_t3_f0_muonall_h384_e3_s3_r5/model_ep2.pt",
    "checkpoint_sha256": "201221c6b4140dda70b078db2a9d7a7246ec40b813f054570d195cecefc61582",
    "provenance": "artifacts/runs/clean_t3_f0_muonall_h384_e3_s3_r5/provenance.json",
    "provenance_sha256": "740b0d96ee311f215153db62272262e0153ce11f8f91560b03efae8e739f9ed0",
}


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


def test_b8_keeps_the_same_first16_order_and_four_source_qualified_tail_omissions() -> None:
    positions = seeded_epoch_positions(11_940, seed=3, batch_size=8, epoch=0)
    positions_b16 = seeded_epoch_positions(11_940, seed=3, batch_size=16, epoch=0)
    assert positions[:16] == positions_b16[:16] == ACTUAL_F0_FIRST16_POSITIONS
    assert len(positions) == 11_940
    assert len(positions) // 8 == 1_492
    assert (len(positions) // 8) * 8 == 11_936
    assert positions[11_936:] == [5638, 4941, 3775, 2355]
    assert (len(positions) // 8) * 10 == 14_920


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


def test_actual_producer_teacher_paths_match_portable_consumer_contract(
    tmp_path: Path,
) -> None:
    for path_field in ("checkpoint_path", "provenance_path"):
        relative = ACTUAL_PRODUCER_TEACHER_SOURCE[path_field]
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.touch()

    source = {
        **ACTUAL_PRODUCER_TEACHER_SOURCE,
        "checkpoint_path": str(tmp_path / ACTUAL_PRODUCER_TEACHER_SOURCE["checkpoint_path"]),
        "provenance_path": str(tmp_path / ACTUAL_PRODUCER_TEACHER_SOURCE["provenance_path"]),
        "fit_group_memberships": {"producer-only": True},
        "predictions": {"producer-only": True},
    }
    assert teacher_source_matches_contract(source, CONTRACT_TEACHER_SOURCE, tmp_path)

    wrong_checkpoint = dict(source, checkpoint_path=str(tmp_path / "wrong-checkpoint.pt"))
    assert not teacher_source_matches_contract(wrong_checkpoint, CONTRACT_TEACHER_SOURCE, tmp_path)
    wrong_provenance_hash = dict(source, provenance_sha256="0" * 64)
    assert not teacher_source_matches_contract(
        wrong_provenance_hash, CONTRACT_TEACHER_SOURCE, tmp_path
    )
