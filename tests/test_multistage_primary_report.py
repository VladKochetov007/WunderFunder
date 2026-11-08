from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from wnn.gmetric import GlobalAccumulator
from wnn.multistage_primary_report import (
    AUXILIARY_CONTROL_COUNT,
    AUXILIARY_FIT_COUNT,
    AUXILIARY_SPLIT_SHA256,
    PRIMARY_SPLIT_SHA256,
    PRIMARY_TRAIN_COUNT,
    PRIMARY_VALID_COUNT,
    _StreamingMetrics,
    validate_auxiliary_report_membership,
)
from wnn.splits import (
    CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA,
    MULTISTAGE_AUXILIARY_SPLIT_SCHEMA,
    CampaignSplit,
)
from wnn.training.evaluation import _weighted_mse_metrics
from wnn.training.sequence_monitor_identity import membership_sha256


def _splits() -> tuple[CampaignSplit, CampaignSplit, dict[str, dict[str, str]]]:
    """Build the exact frozen cohort sizes with simple source-qualified group IDs."""
    primary_train = tuple(range(PRIMARY_TRAIN_COUNT))
    primary_valid = tuple(range(PRIMARY_VALID_COUNT))
    auxiliary_fit = tuple(range(AUXILIARY_FIT_COUNT))
    auxiliary_control = tuple(range(AUXILIARY_FIT_COUNT, PRIMARY_TRAIN_COUNT))
    primary = CampaignSplit(
        Path("primary.json"),
        PRIMARY_SPLIT_SHA256,
        0,
        1,
        primary_train,
        (),
        primary_valid,
        (),
        CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA,
        "valid",
        "valid",
        "is_scored",
    )
    auxiliary = CampaignSplit(
        Path("auxiliary.json"),
        AUXILIARY_SPLIT_SHA256,
        0,
        1,
        auxiliary_fit,
        (),
        auxiliary_control,
        (),
        MULTISTAGE_AUXILIARY_SPLIT_SCHEMA,
        "train",
        "control",
        "all_after_warmup",
        PRIMARY_SPLIT_SHA256,
    )
    sources = {
        "datasets/train.parquet": {"sha256": "1" * 64},
        "datasets/valid.parquet": {"sha256": "2" * 64},
    }
    return primary, auxiliary, sources


def _valid_payload(
    primary: CampaignSplit, auxiliary: CampaignSplit, sources: dict[str, dict[str, str]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    fit = [["train", group] for group in auxiliary.train_groups]
    control = [["train", group] for group in auxiliary.evaluation_groups]
    data = {
        "split_sha256": auxiliary.sha256,
        "primary_split_sha256": primary.sha256,
        "fit_members": fit,
        "evaluation_members": control,
        "evaluation_role": "control",
        "evaluation_source": "train",
        "evaluation_mask": "all_after_warmup",
        "training_mask": "all_after_warmup",
        "mask": "all_after_warmup AND row_index>=99",
        "sources": dict(sources),
        "normalization": {
            "source_fingerprints": {"datasets/train.parquet": sources["datasets/train.parquet"]}
        },
        "teacher": None,
        "initializer": None,
        "hint_members": None,
    }
    config = {
        "precision_mode": "fp32",
        "kd_weight": 0.0,
        "hint_weight": 0.0,
        "model": {"hint_dim": 0},
    }
    return data, config


def test_auxiliary_report_requires_disjoint_train_only_label_ancestry() -> None:
    primary, auxiliary, sources = _splits()
    data, config = _valid_payload(primary, auxiliary, sources)

    fit, control = validate_auxiliary_report_membership(
        data, config, primary, auxiliary, dict(sources)
    )

    assert len(fit) == AUXILIARY_FIT_COUNT
    assert len(control) == AUXILIARY_CONTROL_COUNT
    assert set(fit).isdisjoint(control)
    assert set(fit) | set(control) == {("train", group) for group in primary.train_groups}
    assert membership_sha256(fit) == membership_sha256(
        [("train", group) for group in auxiliary.train_groups]
    )

    leaked = dict(data)
    leaked["fit_members"] = [["train", group] for group in auxiliary.train_groups[:-1]] + [
        ["valid", 0]
    ]
    with pytest.raises(ValueError, match="ancestry or fit/control membership"):
        validate_auxiliary_report_membership(leaked, config, primary, auxiliary, dict(sources))

    heldout_normalizer = dict(data)
    heldout_normalizer["normalization"] = {
        "source_fingerprints": {
            "datasets/train.parquet": sources["datasets/train.parquet"],
            "datasets/valid.parquet": sources["datasets/valid.parquet"],
        }
    }
    with pytest.raises(ValueError, match="ancestry or fit/control membership"):
        validate_auxiliary_report_membership(
            heldout_normalizer, config, primary, auxiliary, dict(sources)
        )

    teacher_parent = dict(data)
    teacher_parent["teacher"] = {"checkpoint_sha256": "a" * 64}
    with pytest.raises(ValueError, match="ancestry or fit/control membership"):
        validate_auxiliary_report_membership(
            teacher_parent, config, primary, auxiliary, dict(sources)
        )


def test_report_metric_accumulation_matches_canonical_wp_and_wmse() -> None:
    target = np.array(
        [
            [[0.2, -0.5], [1.1, 0.7], [-2.5, 1.8], [0.0, 0.1]],
            [[-0.9, 1.3], [0.4, -1.5], [2.2, -0.3], [1.7, 2.7]],
        ],
        dtype=np.float32,
    )
    prediction = np.array(
        [
            [[0.1, -0.2], [1.6, 0.5], [-1.2, 2.4], [0.3, -0.1]],
            [[-0.2, 1.8], [0.3, -1.7], [2.5, 0.2], [1.3, 1.9]],
        ],
        dtype=np.float32,
    )
    mask = np.array([[False, True, True, False], [True, True, False, True]])
    streamed = _StreamingMetrics()
    streamed.add(target[:1], prediction[:1], mask[:1])
    streamed.add(target[1:], prediction[1:], mask[1:])
    result = streamed.result()
    expected_wp = GlobalAccumulator()
    expected_wp.add(target, prediction, mask)
    expected_wmse = _weighted_mse_metrics(target, prediction, mask)

    assert result["metric_id"] == expected_wp.result()["metric_id"]
    assert result["selected_rows"] == int(mask.sum())
    assert result["wp"] == pytest.approx(expected_wp.result()["wp"], abs=1e-12)
    assert result["t0"] == pytest.approx(expected_wp.result()["t0"], abs=1e-12)
    assert result["t1"] == pytest.approx(expected_wp.result()["t1"], abs=1e-12)
    for key in ("wmse", "wmse_t0", "wmse_t1"):
        assert result[key] == pytest.approx(expected_wmse[key], abs=1e-12)
