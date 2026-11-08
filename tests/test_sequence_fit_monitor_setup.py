from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from wnn.splits import CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA
from wnn.training.sequence_config_validation import validate_monitoring_config
from wnn.training.sequence_distributed import DistributedContext
from wnn.training.sequence_fit_monitor_setup import (
    initialize_monitor,
    record_monitoring_plan,
    select_train_eval_members,
)
from wnn.training.sequence_monitor_identity import (
    TRAIN_EVAL_HASH_RANK_NAMESPACE,
    hash_rank_train_eval_members,
    mask_comparison,
    train_eval_comparability,
)


def test_train_eval_mask_must_match_actual_fitting_mask(tmp_path: Path) -> None:
    data = SimpleNamespace(fit=SimpleNamespace(training_mask="is_scored"))
    with pytest.raises(ValueError, match="match the actual fitting objective mask"):
        initialize_monitor(
            tmp_path,
            tmp_path,
            {"train_eval_mask": "all_after_warmup"},
            [],
            data,
            {},
            False,
            DistributedContext(0, 0, 1, torch.device("cpu"), False),
        )


def test_mask_comparison_is_accurate_for_legacy_and_extended_monitors() -> None:
    assert mask_comparison("is_scored") == (
        "train_eval and valid both use the official is_scored mask"
    )
    assert "masks differ" in mask_comparison("all_after_warmup")


def test_hash_rank_selection_is_fit_only_seed_independent_and_backwards_compatible() -> None:
    fit_members = [("train", group) for group in (5, 100, 2, 89, 14, 310, 1, 21)]
    valid_members = [("valid", group) for group in (1, 2)]
    data = SimpleNamespace(
        split=SimpleNamespace(schema=CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA),
        fit=SimpleNamespace(members=fit_members),
        evaluation=SimpleNamespace(members=valid_members),
    )

    prefix = select_train_eval_members({"train_eval_sequences": 3}, data)
    assert prefix == fit_members[:3]
    assert (
        select_train_eval_members(
            {"train_eval_sequences": 3, "train_eval_selection": "prefix"}, data
        )
        == prefix
    )

    config = {"train_eval_sequences": 5, "train_eval_selection": "hash_rank_v1"}
    selected = select_train_eval_members(config, data)
    expected = sorted(
        fit_members,
        key=lambda member: (
            hashlib.sha256(
                json.dumps(
                    [TRAIN_EVAL_HASH_RANK_NAMESPACE, member[0], member[1]],
                    separators=(",", ":"),
                ).encode()
            ).digest(),
            member[0],
            member[1],
        ),
    )[:5]
    assert selected == expected
    assert selected == select_train_eval_members(config, data)
    assert selected == hash_rank_train_eval_members(list(reversed(fit_members)))[:5]
    assert set(selected).issubset(set(data.fit.members))
    assert not set(selected).intersection(valid_members)


def test_hash_rank_diagnostic_never_claims_generalization_comparability() -> None:
    metadata = train_eval_comparability(
        "is_scored", "is_scored", "valid", "hash_rank_v1", 128, 1873
    )
    assert metadata["train_eval_mask_matches_evaluation_mask"] is True
    assert metadata["comparable_to_primary_valid"] is False
    assert metadata["gap_is_calibrated_generalization_estimate"] is False
    assert metadata["selection_claims_representativeness"] is False


def test_inner_control_is_not_described_as_official_valid() -> None:
    metadata = train_eval_comparability(
        "all_after_warmup", "all_after_warmup", "control", "hash_rank_v1", 128, 32
    )
    assert metadata["evaluation_role"] == "control"
    assert metadata["comparable_to_evaluation"] is False
    assert "comparable_to_primary_valid" not in metadata
    assert "official_valid_sequence_count" not in metadata
    assert metadata["evaluation_sequence_count"] == 32
    assert "inner control" in str(metadata["reason"])
    assert "official valid" not in str(metadata["reason"])


def test_hash_rank_rejects_duplicate_fit_members_and_config_is_explicit() -> None:
    duplicate = [("train", 1), ("train", 1)]
    with pytest.raises(ValueError, match="duplicate members"):
        hash_rank_train_eval_members(duplicate)

    validate_monitoring_config(
        {
            "train_eval_sequences": 128,
            "diagnostics_every_steps": 10,
            "train_eval_selection": "hash_rank_v1",
        },
        {"architecture": "gru"},
        4,
        "final_only",
    )


def test_report_only_plan_labels_exposed_best_and_keeps_control_role_distinct() -> None:
    config = {
        "train_eval_sequences": 128,
        "train_eval_selection": "hash_rank_v1",
        "train_eval_mask": "all_after_warmup",
        "early_stopping": {
            "min_epoch": 2,
            "patience": 2,
            "min_delta": 0.0001,
            "action": "report_only",
        },
    }
    valid_data = SimpleNamespace(
        split=SimpleNamespace(
            evaluation_role="valid", evaluation_source="valid.parquet", evaluation_mask="is_scored"
        ),
        evaluation=SimpleNamespace(members=[("valid", 2)]),
        provenance={},
    )
    record_monitoring_plan(valid_data, config, [("train", 1)], "raw112", None)
    valid_plan = valid_data.provenance["monitoring_plan"]
    assert "report-only" in valid_plan["early_stopping_semantics"]
    assert "exposed" in valid_plan["checkpoint_selection"]
    assert "not an independent or fixed inference endpoint" in valid_plan["checkpoint_selection"]

    control_data = SimpleNamespace(
        split=SimpleNamespace(
            evaluation_role="control",
            evaluation_source="train.parquet",
            evaluation_mask="is_scored",
        ),
        evaluation=SimpleNamespace(members=[("train", 2)]),
        provenance={},
    )
    record_monitoring_plan(control_data, config, [("train", 1)], "raw112", None)
    control_plan = control_data.provenance["monitoring_plan"]
    assert control_plan["evaluation_role"] == "control"
    assert "inner control" in control_plan["checkpoint_selection"]
    assert "official valid" not in control_plan["checkpoint_selection"]
