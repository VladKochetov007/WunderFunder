from __future__ import annotations

import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import readout_r1 as readout


def test_wait_does_not_admit_existing_result_after_fit_cutoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = tmp_path / "control/root-fits-r1/shared_control_s203201-result.json"
    result.parent.mkdir(parents=True)
    result.write_text(json.dumps({"status": "fit_complete"}), encoding="utf-8")
    monkeypatch.setattr(readout.time, "time", lambda: 100.001)
    assert readout._wait_for_fit(tmp_path, "shared_control_s203201", 100.0) is None


def _fit_fixture(cell: str = "shared_control_s203201") -> tuple[dict[str, object], ...]:
    gpu, uuid = readout.GPU_BINDINGS[cell]
    target = 0 if cell.startswith("t0_specialist_") else 1 if cell.startswith("t1_specialist_") else None
    seed = 203201 if cell.endswith("203201") else 203202
    config: dict[str, object] = {
        "seed": seed,
        "supervised_target": target,
        "epochs": 6,
        "objective_mode": "standalone_labels_wmse_v2",
        "required_device": "L4",
        "required_world_size": 1,
        "normalization_scope": "all_fit",
        "normalization_sequences": readout.FIT_SEQUENCES,
        "training_mask": "all_after_warmup",
        "checkpoint_endpoint_policy": "configured_final_epoch",
        "checkpoint_retention": "all_epochs",
        "model": {"architecture": "gru", "hidden": 256, "layers": 2},
    }
    fit_members = [["train", index] for index in range(readout.FIT_SEQUENCES)]
    valid_members = [["valid", index] for index in range(readout.VALID_SEQUENCES)]
    valid_hash = hashlib.sha256(json.dumps(valid_members, separators=(",", ":")).encode()).hexdigest()
    data: dict[str, object] = {
        "split_sha256": readout.SPLIT_SHA256,
        "fit_members": fit_members,
        "evaluation_members": valid_members,
        "sources": {
            "datasets/train.parquet": {
                "bytes": 100,
                "mtime_ns": 10,
                "footer_bytes": 20,
                "footer_sha256": "1" * 64,
            },
            "datasets/valid.parquet": {
                "bytes": 200,
                "mtime_ns": 20,
                "footer_bytes": 30,
                "footer_sha256": "2" * 64,
            },
        },
        "teacher": None,
        "normalization": {"scope": "all_fit", "sequence_count": readout.FIT_SEQUENCES},
        "training_mask": "all_after_warmup",
        "objective_contract": {
            "mode": "standalone_labels_wmse_v2",
            "reference_training_ancestry": "none",
        },
        "report_only_monitor": {"endpoint": "raw standalone z; no reference or blend"},
    }
    data_sha = readout.canonical_sha256(data)
    config_sha = readout.canonical_sha256(config)
    model_sha = "a" * 64
    profile = {"config_sha256": config_sha, "gpu": gpu, "gpu_uuid": uuid}
    metadata = {
        "config": config,
        "source": {"sha256": readout.SOURCE_SHA256},
        "provenance": {
            "data": data,
            "data_sha256": data_sha,
            "world_size": 1,
            "runtime": {"devices": [{"name": "NVIDIA L4", "uuid": uuid}]},
        },
    }
    events: list[dict[str, object]] = []
    for epoch in range(1, 7):
        events.append(
            {
                "schema": "wnn.monitor.v1",
                "kind": "evaluation",
                "epoch": epoch,
                "step": epoch * readout.UPDATES_PER_EPOCH,
                "metrics": {
                    "valid/sequence_count": readout.VALID_SEQUENCES,
                    "valid/selected_rows": readout.VALID_ROWS,
                    "valid/wp": 0.65,
                    "valid/wp/t0": 0.6,
                    "valid/wp/t1": 0.7,
                },
                "context": {
                    "split_sha256": readout.SPLIT_SHA256,
                    "evaluation_member_count": readout.VALID_SEQUENCES,
                    "evaluation_member_sha256": valid_hash,
                    "endpoint": "raw standalone z; no reference or blend",
                },
            }
        )
    final = events[-1]["metrics"]
    identity = {
        "source_sha256": readout.SOURCE_SHA256,
        "config_sha256": config_sha,
        "data_sha256": data_sha,
        "world_size": 1,
    }
    terminal = {
        "status": "completed",
        "completed_epochs": 6,
        "global_updates": readout.EXPECTED_UPDATES,
        "model_sha256": model_sha,
        "identity": identity,
    }
    fit_result = {
        "cell": cell,
        "exit_code": 0,
        "timed_out": False,
        "resource_errors": [],
        "status": "fit_complete",
        "model_sha256": model_sha,
        "final_metrics": final,
    }
    return profile, fit_result, metadata, terminal, model_sha, events


def test_fixed_endpoint_fit_receipt_binds_source_model_and_coverage() -> None:
    profile, result, metadata, terminal, model_sha, events = _fit_fixture()
    fit = readout.validate_fit_evidence(
        "shared_control_s203201", profile, result, metadata, terminal, model_sha, events
    )
    assert fit["model_sha256"] == model_sha
    assert fit["source_sha256"] == readout.SOURCE_SHA256
    assert len(fit["evaluation_members"]) == readout.VALID_SEQUENCES
    assert set(fit["data_source_fingerprints"]) == {
        "datasets/train.parquet",
        "datasets/valid.parquet",
    }


def test_fit_membership_matches_exact_official_split() -> None:
    profile, result, metadata, terminal, model_sha, events = _fit_fixture()
    fit = readout.validate_fit_evidence(
        "shared_control_s203201", profile, result, metadata, terminal, model_sha, events
    )
    split_manifest = {
        "schema": "wnn.chronological-primary.v1",
        "policy": "official_train_fit_full_official_valid_evaluation_v1",
        "chronology_basis": "official train/valid source boundary; calendar timestamps unavailable",
        "fit": {"train": list(range(readout.FIT_SEQUENCES)), "valid": []},
        "evaluation": {"valid": list(range(readout.VALID_SEQUENCES))},
    }
    readout.validate_fit_membership(fit, split_manifest)
    split_manifest["fit"]["train"][-1] = readout.FIT_SEQUENCES  # type: ignore[index]
    with pytest.raises(ValueError, match="exact official primary split"):
        readout.validate_fit_membership(fit, split_manifest)


def test_fit_receiver_rejects_wrong_model_source_and_partial_coverage() -> None:
    profile, result, metadata, terminal, model_sha, events = _fit_fixture()
    bad_result = deepcopy(result)
    bad_result["model_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="ROOT fit result"):
        readout.validate_fit_evidence(
            "shared_control_s203201", profile, bad_result, metadata, terminal, model_sha, events
        )

    bad_metadata = deepcopy(metadata)
    bad_metadata["source"]["sha256"] = "c" * 64  # type: ignore[index]
    with pytest.raises(ValueError, match="source/config"):
        readout.validate_fit_evidence(
            "shared_control_s203201", profile, result, bad_metadata, terminal, model_sha, events
        )

    partial = deepcopy(events)
    partial[-1]["metrics"]["valid/selected_rows"] -= 1  # type: ignore[index]
    with pytest.raises(ValueError, match="coverage mismatch"):
        readout.validate_fit_evidence(
            "shared_control_s203201", profile, result, metadata, terminal, model_sha, partial
        )


def test_prediction_manifest_rejects_wrong_producer_model_and_partial_membership() -> None:
    profile, result, metadata, terminal, model_sha, events = _fit_fixture()
    fit = readout.validate_fit_evidence(
        "shared_control_s203201", profile, result, metadata, terminal, model_sha, events
    )
    producer = {
        "checkpoint_sha256": model_sha,
        "split_sha256": readout.SPLIT_SHA256,
        "fit_members": fit["fit_members"],
        "teacher": None,
        "config": metadata["config"],
        "provenance": {
            "identity": {
                "source_sha256": readout.SOURCE_SHA256,
                "config_sha256": fit["config_sha256"],
                "data_sha256": fit["data_sha256"],
                "world_size": 1,
            },
            "completed_epochs": 6,
            "global_updates": readout.EXPECTED_UPDATES,
        },
    }
    sequences = [
        {"split": "valid", "row_group": i, "file": f"valid-{i}.npy", "sha256": "d" * 64}
        for i in range(readout.VALID_SEQUENCES)
    ]
    manifest = {
        "schema": "sequence-predictions-v1",
        "complete": True,
        "input_schema": "raw112-v1",
        "membership": "evaluation",
        "split_sha256": readout.SPLIT_SHA256,
        "hint_dim": 0,
        "sequence_length": readout.SEQUENCE_LENGTH,
        "world_size": 1,
        "producer": producer,
        "sequences": sequences,
    }
    readout.validate_prediction_manifest(manifest, fit, "e" * 64)
    wrong_model = deepcopy(manifest)
    wrong_model["producer"]["checkpoint_sha256"] = "f" * 64  # type: ignore[index]
    with pytest.raises(ValueError, match="producer"):
        readout.validate_prediction_manifest(wrong_model, fit, "e" * 64)
    partial = deepcopy(manifest)
    partial["sequences"] = partial["sequences"][:-1]  # type: ignore[index]
    with pytest.raises(ValueError, match="sequence list"):
        readout.validate_prediction_manifest(partial, fit, "e" * 64)


def test_score_receipt_compares_same_exposed_validation_and_four_rows_per_seed() -> None:
    scores: dict[str, dict[str, object]] = {}
    for i, cell in enumerate(readout.CELLS):
        target0, target1 = 0.60 + i / 100, 0.70 + i / 100
        score = {
            "metric_id": "global_wp_clip2_v1",
            "selected_rows": readout.VALID_ROWS,
            "t0": target0,
            "t1": target1,
            "wp": 0.5 * (target0 + target1),
        }
        receipt = {
            "split_sha256": readout.SPLIT_SHA256,
            "predictions_manifest_sha256": "1" * 64,
            "producer": {"checkpoint_sha256": "2" * 64},
            "sequences": readout.VALID_SEQUENCES,
            "interpretation": "Exposed fold; fixed endpoint, not checkpoint selection or public score.",
            "score": score,
        }
        manifest = {"producer": receipt["producer"]}
        monitor = {"final_metrics": {"valid/wp/t0": target0, "valid/wp/t1": target1, "valid/wp": score["wp"]}}
        compared = readout.validate_score_receipt(receipt, manifest, "1" * 64, monitor)
        scores[cell] = compared
    rows = readout.derive_output_rows(scores)
    assert len(rows) == 8
    assert rows[0]["output"].startswith("baseline")
    assert rows[4]["seed"] == 203202
    bad = deepcopy(scores)
    bad[readout.CELLS[0]]["within_tolerance"] = False
    with pytest.raises(ValueError, match="all six"):
        readout.derive_output_rows(bad)
