from __future__ import annotations

import json
from pathlib import Path

import torch

from wnn.multistage_search import canonical_json_sha256, file_sha256, read_json
from wnn.multistage_search_adopt_fit import load_config as load_adoption_config
from wnn.multistage_search_successor_parent import qualified_parent


def test_source3_adoption_config_matches_native_normalized_identity() -> None:
    """Load the real R5 config as the trainer does while keeping its raw-byte digest."""
    packet = (
        Path(__file__).resolve().parents[1]
        / "research/experiment_drafts/multistage-optuna-b300-20261031"
    )
    config_path = packet / "receiving/actual-profiles256-r1/requests/gru-b256-prefetch1-r5.json"
    profile_path = (
        packet / "receiving/actual-profiles256-r1/profiles/gru-b256-r5/profile_receipt.json"
    )
    review = read_json(packet / "INDEPENDENT_ACTUAL_PROFILE_REVIEW_R3.json")
    reviewed_profile = next(
        item for item in review["batch256_r5_profiles"] if item["family"] == "gru"
    )

    config = load_adoption_config(config_path)
    identity = read_json(profile_path)["identity"]

    assert file_sha256(config_path) == reviewed_profile["config_sha256_raw"]
    assert identity["config_sha256"] == canonical_json_sha256(config)
    assert file_sha256(config_path) != identity["config_sha256"]


def test_source3_qualified_parent_uses_native_normalized_config(tmp_path: Path) -> None:
    """A real R5 raw config qualifies against its native normalized checkpoint identity."""
    packet = (
        Path(__file__).resolve().parents[1]
        / "research/experiment_drafts/multistage-optuna-b300-20261031"
    )
    source_config = packet / "receiving/actual-profiles256-r1/requests/gru-b256-prefetch1-r5.json"
    profile_path = (
        packet / "receiving/actual-profiles256-r1/profiles/gru-b256-r5/profile_receipt.json"
    )
    native_identity = read_json(profile_path)["identity"]
    trial_dir = tmp_path / "trial"
    fit_dir = trial_dir / "fit"
    fit_dir.mkdir(parents=True)
    config_path = trial_dir / "config.json"
    config_path.write_bytes(source_config.read_bytes())
    config = load_adoption_config(config_path)
    assert native_identity["config_sha256"] == canonical_json_sha256(config)

    data = native_identity["data"]
    checkpoint = fit_dir / "model.pt"
    torch.save(
        {
            "checkpoint_schema": "wnn.sequence-fit.v1",
            "config": config,
            "data": data,
            "provenance": {"identity": native_identity},
            "completed_epochs": config["epochs"],
            "epoch": config["epochs"] - 1,
        },
        checkpoint,
    )
    result_path = trial_dir / "trial_result.json"
    result_path.write_text(
        json.dumps(
            {
                "status": "COMPLETE",
                "stage": 1,
                "family": "gru",
                "fit_output": str(fit_dir),
                "fit_model_sha256": file_sha256(checkpoint),
                "config_sha256": file_sha256(config_path),
                "objective": {"evaluation": {"scope": "inner_train_proxy"}},
            }
        ),
        encoding="utf-8",
    )

    parent, qualified_checkpoint, qualified_config = qualified_parent(
        result_path, allowed={1}
    )
    assert parent["status"] == "COMPLETE"
    assert qualified_checkpoint == checkpoint
    assert canonical_json_sha256(qualified_config) == native_identity["config_sha256"]
    assert file_sha256(config_path) == parent["config_sha256"]
