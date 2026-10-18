"""Focused guards for source, graph, fold, and target-index identity checks."""

from __future__ import annotations

import numpy as np
import pytest
from canonical_manifest_checks import assert_memberships_equal, validate_manifest_sequence_ids
from derive_l4_folds import GRAPH_BYTES, GRAPH_SHA256, fold_partition, validate_graph_identity
from validate_l4_targets import validate_group_order


def test_graph_identity_rejects_wrong_digest_or_size() -> None:
    with pytest.raises(ValueError, match="graph identity mismatch"):
        validate_graph_identity("0" * 64, GRAPH_BYTES)
    with pytest.raises(ValueError, match="graph identity mismatch"):
        validate_graph_identity(GRAPH_SHA256, GRAPH_BYTES + 1)


def test_fold_partition_rejects_component_split_across_folds() -> None:
    component_by_node = {("valid", 10): 4, ("valid", 11): 4}
    evaluation_by_fold = {0: {10}, 1: {11}, 2: set(), 3: set()}
    with pytest.raises(ValueError, match="component crosses folds"):
        fold_partition(component_by_node, evaluation_by_fold, {10, 11})


def test_target_row_group_order_rejects_permutation() -> None:
    with pytest.raises(ValueError, match="canonical parquet order"):
        validate_group_order("valid", np.array([0, 2, 1]), 3)


def test_target_row_group_order_excludes_corrupt_train_group() -> None:
    groups = np.array([group for group in range(10_607) if group != 528])
    assert validate_group_order("train", groups, 10_607) == groups.tolist()
    assert 528 not in groups


def test_target_index_must_match_all_canonical_sequence_maps() -> None:
    manifests = [{"sequences": {"train": [{"row_group": 0, "seq_ix": 123}]}} for _ in range(4)]
    validate_manifest_sequence_ids(manifests, "train", [0], [123])
    with pytest.raises(ValueError, match="seq_ix at row group 0"):
        validate_manifest_sequence_ids(manifests, "train", [0], [124])


def test_canonical_sequence_maps_must_agree_across_folds() -> None:
    manifests = [{"sequences": {"valid": [{"row_group": 2, "seq_ix": 50}]}} for _ in range(4)]
    manifests[3]["sequences"]["valid"][0]["seq_ix"] = 51
    with pytest.raises(ValueError, match="differ across folds"):
        validate_manifest_sequence_ids(manifests, "valid", [2], [50])


def test_membership_comparison_covers_fit_evaluation_and_exclusions() -> None:
    canonical = {
        "fit": {"train": [0, 1], "valid": [3]},
        "evaluation": {"valid": [4]},
        "excluded": {"train": [2], "valid": [4]},
        "sequences": {"valid": [{"row_group": 4, "seq_ix": 99, "component": 1}]},
    }
    fresh = {**canonical}
    fresh["fit"] = {"train": [0], "valid": [3]}
    with pytest.raises(ValueError, match="fit membership differs"):
        assert_memberships_equal(canonical, fresh, 0)
