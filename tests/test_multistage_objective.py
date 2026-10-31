from __future__ import annotations

import json
from pathlib import Path

import pytest

from wnn.splits import MULTISTAGE_AUXILIARY_SPLIT_SCHEMA, CampaignSplit
from wnn.training.sequence_monitor_identity import membership_sha256
from wnn.training.sequence_provenance import canonical_sha256
from wnn.training.sequence_score import _inner_objective_receipt


def _fixture(tmp_path: Path) -> tuple[CampaignSplit, dict, Path, list[tuple[str, int]]]:
    primary_hash = "a" * 64
    split = CampaignSplit(
        tmp_path / "aux.json",
        "b" * 64,
        0,
        1,
        (0,),
        (),
        (1,),
        (("train", 0, 0), ("train", 1, 1)),
        MULTISTAGE_AUXILIARY_SPLIT_SCHEMA,
        "train",
        "control",
        "all_after_warmup",
        primary_hash,
    )
    members = [("train", 1)]
    fit_members = [["train", 0]]
    data = {
        "split_sha256": split.sha256,
        "primary_split_sha256": primary_hash,
        "fit_members": fit_members,
        "evaluation_members": [list(member) for member in members],
    }
    config = {"precision_mode": "fp32", "model": {"architecture": "gru"}}
    identity = {
        "config_sha256": canonical_sha256(config),
        "data_sha256": canonical_sha256(data),
        "data": data,
    }
    producer = {
        "checkpoint_sha256": "c" * 64,
        "split_sha256": split.sha256,
        "fit_members": fit_members,
        "config": config,
        "provenance": {"identity": identity},
    }
    predictions = {"split_sha256": split.sha256, "producer": producer}
    manifest = tmp_path / "predictions.json"
    manifest.write_text(json.dumps({"producer": producer}))
    return split, predictions, manifest, members


def test_inner_objective_binds_producer_identity_and_memberships(tmp_path: Path) -> None:
    split, predictions, manifest, members = _fixture(tmp_path)

    receipt = _inner_objective_receipt(
        split,
        predictions,
        manifest,
        members,
        {"wp": 0.5, "t0": 0.4, "t1": 0.6, "metric_id": "wp-v1"},
        selected_rows=19_900,
    )

    identity = predictions["producer"]["provenance"]["identity"]
    assert receipt["producer"]["provenance"]["identity"] == identity
    assert receipt["producer"]["provenance"]["identity_sha256"] == canonical_sha256(identity)
    assert receipt["producer"]["split_sha256"] == split.sha256
    assert receipt["producer"]["fit_members_sha256"] == membership_sha256([("train", 0)])
    assert receipt["prediction_producer"] == predictions["producer"]


def test_inner_objective_rejects_unbound_control_membership(tmp_path: Path) -> None:
    split, predictions, manifest, members = _fixture(tmp_path)
    predictions["producer"]["provenance"]["identity"]["data"]["evaluation_members"] = [["valid", 1]]

    with pytest.raises(ValueError, match="producer run identity"):
        _inner_objective_receipt(
            split,
            predictions,
            manifest,
            members,
            {"wp": 0.5, "t0": 0.4, "t1": 0.6, "metric_id": "wp-v1"},
            selected_rows=19_900,
        )
