from __future__ import annotations

import json
from pathlib import Path

import pytest

from wnn.multistage_search import canonical_json_sha256, file_sha256
from wnn.multistage_search_metrics import validate_inner_objective
from wnn.training.sequence_provenance import validate_profile_receipt


def test_approved_profile_copy_passes_native_exact_identity_validator(tmp_path: Path) -> None:
    """Exercise the production profile gate on the reviewed native-compatible copy."""
    memory_bytes = 64 * (1 << 30)
    identity = {
        "schema": "wnn.sequence-run-identity.v1",
        "model_parameters": 7,
        "world_size": 1,
        "runtime": {
            "devices": [{"rank": 0, "total_memory_bytes": memory_bytes}],
        },
    }
    receipt = {
        "receipt_schema": "wnn.sequence-profile.v1",
        "status": "profile_complete",
        "approved": True,
        "identity": identity,
        "updates": 50,
        "epochs_traversed": 0,
        "elapsed_seconds": 12.5,
        "forwarded_rows": 200,
        "real_rows": 200,
        "valid_rows": 200,
        "losses": {"label": 0.2, "distillation": 0.0, "hint": 0.0, "total": 0.2},
        "model_parameters": 7,
        "memory_by_rank": [
            {
                "rank": 0,
                "peak_allocated_bytes": 4 * (1 << 30),
                "peak_reserved_bytes": 5 * (1 << 30),
                "device_total_memory_bytes": memory_bytes,
                "host_peak_rss_bytes": 1 << 30,
            }
        ],
        "peak_memory_bytes": 4 * (1 << 30),
        "peak_reserved_bytes": 5 * (1 << 30),
        "scientific_checkpoint_written": False,
        "errors": [],
        "error_count": 0,
    }
    path = tmp_path / "approved_profile_receipt.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")

    assert validate_profile_receipt(path, identity) == file_sha256(path)
    with pytest.raises(ValueError, match="does not match"):
        validate_profile_receipt(path, identity | {"world_size": 2})


def test_inner_objective_binds_producer_identity_and_checkpoint(tmp_path: Path) -> None:
    """Reject a score that substitutes the expected identity for producer evidence."""
    identity = {"schema": "wnn.sequence-run-identity.v1", "world_size": 1}
    identity_sha = canonical_json_sha256(identity)
    split_sha = "a" * 64
    primary_sha = "b" * 64
    checkpoint_sha = "c" * 64
    members_sha = "d" * 64
    objective = {
        "schema": "wnn.multistage-inner-objective.v1",
        "split_sha256": split_sha,
        "primary_split_sha256": primary_sha,
        "model_sha256": checkpoint_sha,
        "prediction_manifest_sha256": "e" * 64,
        "fit_members_sha256": members_sha,
        "evaluation_members_sha256": "f" * 64,
        "sequence_count": 3,
        "selected_rows": 60000,
        "objective": {
            "name": "wp",
            "value": 0.4,
            "t0": 0.3,
            "t1": 0.5,
            "reduction": "mean_of_two_pooled_clipped_weighted_Pearson_targets",
        },
        "evaluation": {
            "role": "control",
            "scope": "inner_train_proxy",
            "source": "train",
            "row_mask": "all_after_warmup AND row_index>=99",
            "is_scored_used": False,
        },
        "interpretation": "selection proxy only; not chronological validation or public score",
        "producer": {
            "checkpoint_sha256": checkpoint_sha,
            "split_sha256": split_sha,
            "fit_members_sha256": members_sha,
            "provenance": {"identity": identity},
        },
    }
    path = tmp_path / "objective.json"
    path.write_text(json.dumps(objective), encoding="utf-8")

    assert (
        validate_inner_objective(
            path,
            auxiliary_split_sha256=split_sha,
            primary_split_sha256=primary_sha,
            expected_checkpoint_sha256=checkpoint_sha,
            expected_identity_sha256=identity_sha,
        )["identity_sha256"]
        == identity_sha
    )

    objective["producer"]["provenance"]["identity"] = {"schema": "wrong"}
    path.write_text(json.dumps(objective), encoding="utf-8")
    with pytest.raises(ValueError, match="producer identity"):
        validate_inner_objective(
            path,
            auxiliary_split_sha256=split_sha,
            primary_split_sha256=primary_sha,
            expected_checkpoint_sha256=checkpoint_sha,
            expected_identity_sha256=identity_sha,
        )
