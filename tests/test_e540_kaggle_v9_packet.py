from __future__ import annotations

import copy
import hashlib
import json
import time
from pathlib import Path

import pytest

import scripts.build_e540_kaggle_v9_packet as e540_v9_builder
import scripts.e540_kaggle_v9_worker as e540_v9_worker

FIXTURE = Path(__file__).parent / "fixtures/E540_V8_LINEAR_CONTROL_PRODUCER_LAYOUT.json"
ORIGIN = Path(__file__).parent / "fixtures/E540_V8_LINEAR_CONTROL_PRODUCER_LAYOUT.origin.json"
SUBSET = e540_v9_worker.ROOT / e540_v9_worker.SUBSET_MANIFEST
V8_RESULT = Path(__file__).parent / "fixtures/E540_V8_TERMINAL_RESULT.json"
EXPECTED_ARGS_SHA256 = "931dccef9138bf3a41fc18c27b24af5202a3c842fc017f95af84d6b4f3e169d4"


def _fixture_and_manifest() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    origin = json.loads(ORIGIN.read_text(encoding="utf-8"))
    subset = json.loads(SUBSET.read_text(encoding="utf-8"))
    provenance = fixture["sibling_provenance"]
    row = provenance["update_diagnostics"][0]
    checkpoint_hashes = provenance["checkpoint_sha256"]
    manifest = {
        "schema": "wnn.e540-v8-linear-control.v1",
        "producer": {
            "packet_id": "E540_FP32_V8",
            "source_commit": e540_v9_worker.V8_ORIGIN["source_commit"],
            "source_review_sha256": e540_v9_worker.V8_ORIGIN["source_review_sha256"],
            "worker_sha256": e540_v9_worker.V8_ORIGIN["worker_sha256"],
            "code_manifest_sha256": e540_v9_worker.V8_ORIGIN["code_manifest_sha256"],
            "recipe_sha256": e540_v9_worker.V8_ORIGIN["recipe_sha256"],
            "conditional_authorization_sha256": e540_v9_worker.V8_ORIGIN[
                "conditional_authorization_sha256"
            ],
            "terminal_receipt_sha256": e540_v9_worker.V8_ORIGIN["terminal_receipt_sha256"],
            "result_sha256": e540_v9_worker.V8_ORIGIN["result_sha256"],
            "provenance_sha256": origin["producer_sha256"],
        },
        "run_name": "e540v8-linear-s540-fp32",
        "args_sha256": e540_v9_worker.canonical_json_sha256(provenance["args"]),
        "subset_manifest_sha256": provenance["fit_subset_manifest_sha256"],
        "fit_members_sha256": provenance["fit_subset_members_sha256"],
        "split_manifest_sha256": provenance["split_manifest_sha256"],
        "fit_group_counts": {
            key: len(value) for key, value in provenance["fit_row_groups"].items()
        },
        "evaluation_group_count": len(provenance["evaluation_row_groups"]),
        "training_source_hashes": e540_v9_worker.V8_TRAINING_SOURCE_HASHES,
        "files": {
            "model.pt": {"sha256": checkpoint_hashes["model.pt"]},
            "model_ep0.pt": {"sha256": checkpoint_hashes["model_ep0.pt"]},
            "provenance.json": {"sha256": origin["producer_sha256"]},
            "history.json": {"sha256": "a" * 64},
        },
        "update_diagnostics": {
            key: row[key]
            for key in (
                "epoch",
                "cumulative",
                "sample_order_identity",
                "normalization_identity",
                "grad_stats",
            )
        },
    }
    return fixture, subset, manifest


def test_actual_v8_producer_layout_places_checkpoint_hashes_in_sibling_provenance() -> None:
    fixture, subset, manifest = _fixture_and_manifest()
    origin = json.loads(ORIGIN.read_text(encoding="utf-8"))
    provenance = fixture["sibling_provenance"]

    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == origin["fixture_sha256"]
    assert origin["producer_sha256"] == e540_v9_worker.V8_ORIGIN["provenance_sha256"]
    assert "checkpoint_sha256" in provenance
    assert "checkpoint_sha256" not in fixture["checkpoint_snapshot"]["provenance"]
    assert manifest["files"]["model.pt"]["sha256"] == provenance["checkpoint_sha256"]["model.pt"]
    assert provenance["evaluation_row_groups"] != subset["evaluation_groups"]
    assert e540_v9_worker.same_group_membership(
        provenance["evaluation_row_groups"], subset["evaluation_groups"]
    )

    summary = e540_v9_worker.validate_control_provenance(
        provenance,
        manifest,
        subset,
        expected_manifest_sha256="b" * 64,
        expected_args_sha256=EXPECTED_ARGS_SHA256,
    )
    e540_v9_worker.validate_checkpoint_snapshot_metadata(fixture["checkpoint_snapshot"], provenance)

    assert summary["optimizer_completed_updates"] == 72_996
    assert summary["amp_skipped_updates"] == 0
    assert (
        summary["sample_order_identity"]
        == provenance["update_diagnostics"][0]["sample_order_identity"]
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda m: m["producer"].update(source_commit="0" * 40), "source, recipe"),
        (
            lambda m: m.update(subset_manifest_sha256="0" * 64),
            "arguments do not match|manifest differs",
        ),
        (lambda m: m.update(fit_members_sha256="0" * 64), "manifest differs"),
        (
            lambda m: m["training_source_hashes"].update({"scripts/train_rnn.py": "0" * 64}),
            "source map",
        ),
        (
            lambda m: m["update_diagnostics"]["epoch"].update(optimizer_completed_updates=72_995),
            "manifest differs",
        ),
    ],
)
def test_actual_v8_control_is_rejected_if_bound_manifest_identity_drifts(mutate, message) -> None:
    fixture, subset, manifest = _fixture_and_manifest()
    changed = copy.deepcopy(manifest)
    mutate(changed)

    with pytest.raises(ValueError, match=message):
        e540_v9_worker.validate_control_provenance(
            fixture["sibling_provenance"],
            changed,
            subset,
            expected_manifest_sha256="b" * 64,
            expected_args_sha256=EXPECTED_ARGS_SHA256,
        )


def test_v9_rejects_changed_evaluation_membership_but_not_its_source_order() -> None:
    fixture, subset, manifest = _fixture_and_manifest()
    provenance = copy.deepcopy(fixture["sibling_provenance"])
    assert e540_v9_worker.same_group_membership(
        provenance["evaluation_row_groups"], subset["evaluation_groups"]
    )

    provenance["evaluation_row_groups"][-1] = -1
    assert not e540_v9_worker.same_group_membership(
        provenance["evaluation_row_groups"], subset["evaluation_groups"]
    )
    with pytest.raises(ValueError, match="512/468 membership"):
        e540_v9_worker.validate_control_provenance(
            provenance,
            manifest,
            subset,
            expected_manifest_sha256="b" * 64,
            expected_args_sha256=EXPECTED_ARGS_SHA256,
        )


def test_v9_commands_parse_for_ple_profile_fit_and_both_feature_only_predictions() -> None:
    checks = e540_v9_worker.validate_recipe_cli()

    assert [(row["mode"], row["stage"]) for row in checks] == [
        ("ple", "profile"),
        ("ple", "fit"),
        ("linear", "prediction"),
        ("ple", "prediction"),
    ]
    assert all(row["status"] == "parsed" and row["script_token_removed"] for row in checks)


def test_v9_pending_auth_stops_before_mounted_data_hashing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    receipt_path = tmp_path / "pending-auth.json"
    receipt_path.write_text(
        json.dumps(
            {
                "schema": "wnn.e540-conditional-fit-authorization.v1",
                "owner": "root",
                "status": "pending_root_review",
                "authorized": False,
                "conditional_fit_after_profile_gate": True,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(e540_v9_worker, "RUNTIME", tmp_path / "runtime")
    monkeypatch.setattr(e540_v9_worker, "AUTHORIZATION_PATH", receipt_path)
    monkeypatch.setenv("WNN_KERNEL_DEADLINE_MONOTONIC", str(time.monotonic() + 6000))
    monkeypatch.setenv("WNN_E540_PACKET_ID", e540_v9_worker.PACKET_ID)
    monkeypatch.setenv("WNN_E540_AUTHORIZATION_PATH", receipt_path.as_posix())
    monkeypatch.setenv("WNN_E540_KERNEL_WALL_SECONDS", "6000")
    monkeypatch.setenv("WNN_E540_WORKER_WALL_SECONDS", "5200")
    for name, value in (
        ("WNN_E540_CODE_MANIFEST_SHA256", "a" * 64),
        ("WNN_E540_RECIPE_SHA256", "b" * 64),
        ("WNN_E540_SUBSET_SHA256", "c" * 64),
        ("WNN_E540_CONTROL_MANIFEST_SHA256", "d" * 64),
        ("WNN_E540_CONTROL_ARGS_SHA256", "e" * 64),
        ("WNN_E540_CONTROL_MODEL_SHA256", "f" * 64),
        ("WNN_E540_CONTROL_EPOCH0_SHA256", "1" * 64),
        ("WNN_E540_CONTROL_PROVENANCE_SHA256", "2" * 64),
    ):
        monkeypatch.setenv(name, value)

    def reject_data_hash() -> None:
        raise AssertionError("pending authorization must stop before mounted-input hashing")

    monkeypatch.setattr(e540_v9_worker, "verify_mounted_data", reject_data_hash)
    with pytest.raises(ValueError, match="missing valid status"):
        e540_v9_worker.run()

    preflight = json.loads((tmp_path / "runtime/startup_preflight.json").read_text())
    assert preflight["status"] == "failed_before_mounted_data_hashing"
    assert [item["status"] for item in preflight["cli_checks"]] == ["parsed"] * 4


def test_builder_uses_explicit_downloaded_result_path_for_v8_identity() -> None:
    result_origin = json.loads(
        (Path(__file__).parent / "fixtures/E540_V8_TERMINAL_RESULT.origin.json").read_text(
            encoding="utf-8"
        )
    )
    assert hashlib.sha256(V8_RESULT.read_bytes()).hexdigest() == result_origin["fixture_sha256"]
    assert result_origin["producer_sha256"] == e540_v9_worker.V8_ORIGIN["result_sha256"]
    assert V8_RESULT.is_file()
    review, authorization, terminal = e540_v9_builder._validate_v8_identity(V8_RESULT)

    assert review["source_commit"] == e540_v9_worker.V8_ORIGIN["source_commit"]
    assert authorization["status"] == "approved"
    assert terminal["kernel"]["provider_terminal_status"] == "KernelWorkerStatus.ERROR"
