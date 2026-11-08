from __future__ import annotations

import pytest

from wnn.training.sequence_complement_monitor import (
    select_train_eval_members,
    validate_complement_monitor,
)


def test_hash_rank_cohort_is_repeatable_source_qualified_and_fit_only() -> None:
    """Hash selection is independent of caller order and returns whole supplied members."""
    members = [("train", index) for index in range(12)]
    first = select_train_eval_members(members, 5, "hash_rank_v1")
    reversed_input = select_train_eval_members(list(reversed(members)), 5, "hash_rank_v1")

    assert first == reversed_input
    assert len(first) == 5
    assert len(set(first)) == 5
    assert set(first) <= set(members)
    assert select_train_eval_members(members, 3, "prefix") == members[:3]


def test_hash_rank_rejects_invalid_cohort_contracts() -> None:
    """Selection fails closed on empty, oversized, duplicate, or unknown cohorts."""
    members = [("train", 0), ("train", 1)]
    with pytest.raises(ValueError, match="cohort size"):
        select_train_eval_members(members, 0, "hash_rank_v1")
    with pytest.raises(ValueError, match="duplicate"):
        select_train_eval_members([members[0], members[0]], 1, "hash_rank_v1")
    with pytest.raises(ValueError, match="official train"):
        select_train_eval_members([members[0], ("valid", 1)], 1, "hash_rank_v1")
    with pytest.raises(ValueError, match="unsupported"):
        select_train_eval_members(members, 1, "seq_ix_order")


def test_monitor_selection_is_optional_for_frozen_configs_and_validated_when_set() -> None:
    """Old configs keep prefix selection; new configs may explicitly request hash rank."""
    base = {
        "monitoring": {"train_eval_sequences": 32, "eval_batch_size": 4},
        "blend_objective_alpha": 0.1,
        "kd_weight": 0,
    }
    validate_complement_monitor(base)
    validate_complement_monitor(
        base | {"monitoring": base["monitoring"] | {"train_eval_selection": "hash_rank_v1"}}
    )
    with pytest.raises(ValueError, match="train_eval_selection"):
        validate_complement_monitor(
            base | {"monitoring": base["monitoring"] | {"train_eval_selection": "seq_ix_order"}}
        )
