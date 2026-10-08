from __future__ import annotations

import ast
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

STAGE_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(STAGE_ROOT / "src"))

from wnn_jax import e465_gate as gate  # noqa: E402

FOLD_PATH = "research/campaign_2026-09-29/artifacts/splits/fold0.json"
PORTABLE_PATH = "research/campaign_2026-10-04/portable_inputs.json"
ACTUAL_V6_DELIVERY = (
    ROOT
    / "research/campaign_2026-10-06/cloud_preflight/"
    "E463_TPU_V6_OPERATOR_DELIVERY_20261008_R1.json"
)
ACTUAL_V6_PACKET = (
    ROOT / "research/campaign_2026-10-06/cloud_preflight/tpu_packet_v6_realdata_reload_candidate_v1"
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _write_bundle(
    root: Path,
    *,
    provider_status: str = "COMPLETE",
    canonical_kernel: str = gate.V6_KERNEL,
) -> tuple[Path, Path, Path]:
    bundle = root / "bundle"
    outputs = bundle / "outputs"
    outputs.mkdir(parents=True)
    project = root / "project"
    fold_path = project / FOLD_PATH
    fold_path.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / FOLD_PATH, fold_path)
    fold = json.loads(fold_path.read_text(encoding="utf-8"))
    selected = list(fold["fit"]["valid"][:16])

    train_size, train_sha = gate.RAW_DATA["train"]
    valid_size, valid_sha = gate.RAW_DATA["valid"]
    parent = {
        "run": "c20261003_combo_s3_f0",
        "epoch": 3,
        "weight_source": "evaluation_ema",
        "npz_path": "parent/model_ep3.jax.npz",
        "npz_sha256": gate.PARENT_NPZ_SHA256,
        "sidecar_path": "parent/model_ep3.jax.npz.json",
        "sidecar_sha256": gate.PARENT_SIDECAR_SHA256,
        "source_checkpoint_path": "parent/model_ep3.pt",
        "source_checkpoint_sha256": gate.PARENT_CHECKPOINT_SHA256,
        "source_provenance_path": "parent/provenance.json",
        "source_provenance_sha256": gate.PARENT_PROVENANCE_SHA256,
        "dimensions": {"n_in": 211, "hidden": 256, "layers": 2, "n_out": 23},
    }
    assets = {}
    asset_files = []
    for index, relative in enumerate(gate.REQUIRED_V6_ASSETS):
        if relative == PORTABLE_PATH:
            digest = gate.V6_PORTABLE_INPUTS_SHA256
        elif relative == FOLD_PATH:
            digest = gate.V6_FOLD_SHA256
        else:
            digest = hashlib.sha256(f"asset-{index}".encode()).hexdigest()
        assets[relative] = {"bytes": 5, "sha256": digest}
        asset_files.append({"path": relative, "sha256": digest})
    manifest_files = [
        *asset_files,
        {"path": "datasets/train.parquet", "sha256": train_sha},
        {"path": "datasets/valid.parquet", "sha256": valid_sha},
        {"path": parent["npz_path"], "sha256": parent["npz_sha256"]},
        {"path": parent["sidecar_path"], "sha256": parent["sidecar_sha256"]},
        {"path": parent["source_checkpoint_path"], "sha256": parent["source_checkpoint_sha256"]},
        {"path": parent["source_provenance_path"], "sha256": parent["source_provenance_sha256"]},
    ]
    input_manifest = {
        "format": "wnn.e465-tpu-input-manifest",
        "version": 1,
        "fold_manifest": FOLD_PATH,
        "fold_manifest_sha256": gate.V6_FOLD_SHA256,
        "teacher_runs": ["c20260929_since_s3_f0", "c20260929_since_s4_f0"],
        "propensity_run": "c20260929_prop_f0",
        "parent_export": parent,
        "files": manifest_files,
    }
    input_manifest_path = project / "input-manifest.json"
    _write_json(input_manifest_path, input_manifest)

    update = {
        "status": "passed",
        "classification": "one real-data B16/C2000 distributed E463 checkpoint/reload compatibility update",
        "parent_checkpoint_sha256": gate.PARENT_CHECKPOINT_SHA256,
        "parent_npz_sha256": gate.PARENT_NPZ_SHA256,
        "parent_provenance_sha256": gate.PARENT_PROVENANCE_SHA256,
        "fold_manifest_sha256": gate.V6_FOLD_SHA256,
        "source_qualified_membership": {
            "status": "passed",
            "split": "valid",
            "batch_size": 16,
            "row_groups": selected,
            "qualified_membership": [["valid", group] for group in selected],
            "evaluation_overlap_count": 0,
            "outside_parent_fit_count": 0,
        },
        "global_batch": 16,
        "local_batch_per_tpu": 2,
        "chunk_rows": 2000,
        "input_width": 211,
        "hidden_width": 256,
        "layers": 2,
        "output_width": 23,
        "completed_update_count_per_replica": [1] * 8,
        "checkpoint": {
            "sha256": hashlib.sha256(b"synthetic-checkpoint-bytes").hexdigest(),
            "bytes": len(b"synthetic-checkpoint-bytes"),
            "ema_sha256": hashlib.sha256(b"synthetic-checkpoint-bytes").hexdigest(),
            "ema_bytes": len(b"synthetic-checkpoint-bytes"),
            "optimizer_state_sha256": hashlib.sha256(b"synthetic-checkpoint-bytes").hexdigest(),
            "optimizer_state_bytes": len(b"synthetic-checkpoint-bytes"),
            "save_reload_parameter_identity": "bitwise_equal_all_tensors",
            "save_reload_ema_identity": "bitwise_equal_all_tensors",
            "save_reload_optimizer_identity": "bitwise_equal_all_moments_and_count",
        },
        "inference_causality": {
            "status": "passed",
            "checkpoint_reload_prediction_identity": "bitwise_equal",
            "future_perturbation_changed_prefix": False,
            "omitted_state_matches_explicit_zero_state": True,
            "prediction_exported": False,
        },
        "training_data_used": True,
        "fit_valid_is_already_exposed": True,
        "heldout_claim": None,
        "metric_or_score": None,
        "public_forecast": None,
    }
    source_materialization = {
        "status": "verified_direct_read_only_tree",
        "source_manifest_sha256": gate.V6_SOURCE_MANIFEST_SHA256,
        "declared_source_archive_sha256": gate.V6_SOURCE_ARCHIVE_SHA256,
        "verified_file_count": 17,
        "files": {f"source_{i}.py": {"sha256": "a" * 64} for i in range(17)},
    }
    checkpoint_identity = {
        "checkpoint_sha256": gate.PARENT_CHECKPOINT_SHA256,
        "npz_sha256": gate.PARENT_NPZ_SHA256,
        "sidecar_sha256": gate.PARENT_SIDECAR_SHA256,
        "model_source_sha256": gate.MODEL_SOURCE_SHA256,
        "dimensions": {"n_in": 211, "hidden": 256, "layers": 2, "n_out": 23},
    }
    realdata_input = {
        "status": "verified_and_staged_read_only_inputs",
        "source_manifest_sha256": gate.V6_SOURCE_MANIFEST_SHA256,
        "source_archive_sha256": gate.V6_SOURCE_ARCHIVE_SHA256,
        "source_file_count": 17,
        "fold_assets_manifest_sha256": gate.V6_ASSET_MANIFEST_SHA256,
        "verified_asset_files": assets,
        "raw_data": {
            "train": {"dataset_ref": "vladkochetov07/wunder-funder", "bytes": train_size, "sha256": train_sha},
            "valid": {"dataset_ref": "vladkochetov07/wunder-funder", "bytes": valid_size, "sha256": valid_sha},
        },
    }
    devices = [
        {"platform": "tpu", "device_kind": "TPU v5 lite", "id": index}
        for index in range(8)
    ]
    result = {
        "status": "passed",
        "backend": "tpu",
        "device_count": 8,
        "scientific_fit_or_score": None,
        "realdata_update": update,
    }
    values: dict[str, object] = {
        "terminal.json": {"status": "passed", "phase": "complete", "worker_returncode": 0},
        "complete.json": {"status": "passed", "phase": "complete"},
        "startup.json": {"packet": gate.V6_PACKET_ID},
        "device_worker_start.json": {"packet": gate.V6_PACKET_ID},
        "device_worker_terminal.json": {"status": "passed", "phase": "complete", "result": result},
        "device_worker_result.json": result,
        "result.json": result,
        "tpu_diagnostic_complete.json": result,
        "source_materialization.json": source_materialization,
        "e463_checkpoint_identity.json": checkpoint_identity,
        "realdata_input_identity.json": realdata_input,
        "realdata_update_receipt.json": update,
        "tpu_inventory.json": {"jax_backend": "tpu", "local_device_count": 8, "global_device_count": 8, "devices": devices},
        "tpu_hbm_before.json": [{"hbm_current_bytes": 100, "hbm_capacity_bytes": 1000} for _ in range(8)],
        "tpu_hbm_after.json": [{"hbm_peak_bytes": 700, "hbm_capacity_bytes": 1000} for _ in range(8)],
        "tpu_objective_reduction_parity.json": {"status": "passed"},
        "tpu_fullshape_profile.json": {"status": "passed", "synthetic_inputs_and_targets": True, "scientific_score_claim": None},
    }
    for name, value in values.items():
        _write_json(outputs / name, value)
    for name in (
        "model_after_one_chunk_fp32.npz",
        "ema_after_one_chunk_fp32.npz",
        "adamw_after_one_chunk.npz",
    ):
        (outputs / name).write_bytes(b"synthetic-checkpoint-bytes")

    inventory = []
    for path in sorted(outputs.iterdir()):
        raw = path.read_bytes()
        inventory.append(
            {
                "path": f"research/cloud_preflight_tpu_v6/{path.name}",
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    terminal_receipt = {
        "record_id": "E463_TPU_V6_TERMINAL_OPERATOR_RECEIPT_20261008_R1",
        "canonical_kernel": canonical_kernel,
        "provider_status": provider_status,
        "download_exit_code": 0,
        "result_status": "passed",
        "actual_devices": 8,
        "device_kind": "TPU v5 lite",
        "inventory": inventory,
    }
    shutil.copyfile(ACTUAL_V6_DELIVERY, bundle / "delivery_receipt.json")
    _write_json(bundle / "terminal_receipt.json", terminal_receipt)
    return bundle, input_manifest_path, project


def _precision_report() -> dict[str, object]:
    hashes = ["a" * 64, "b" * 64]
    arms = {
        precision: {
            "precision": precision,
            "actual_data": True,
            "batch_count": 2,
            "attempted_updates": 40,
            "completed_updates": 40,
            "finite_loss_gradient_checks": 40,
            "final_params_optimizer_ema_and_recurrent_states_finite": True,
            "batch_data_sha256": hashes,
        }
        for precision in ("fp32", "bf16")
    }
    diagnostics = [
        {
            "update_index": index,
            "fp32_loss": 1.0,
            "bf16_loss": 1.01,
            "fp32_gradient_norm": 2.0,
            "bf16_gradient_norm": 2.01,
            "loss_abs_delta": 0.01,
            "loss_relative_delta": 0.01,
            "gradient_norm_abs_delta": 0.01,
            "gradient_norm_relative_delta": 0.005,
        }
        for index in range(40)
    ]
    return {
        "schema": "wnn.e465-actual-data-precision-gate.v1",
        "status": "passed",
        "profile_batches_per_precision": 2,
        "global_batch_size": 32,
        "sequence_rows": 20_000,
        "chunk_rows": 1_000,
        "chunks_per_batch": 20,
        "arms": arms,
        "paired_step_diagnostics": diagnostics,
        "numerical_review_required_before_fit": True,
    }


def test_delivery_gate_accepts_actual_main_v6_operator_receipt() -> None:
    receipt = gate._validate_delivery(ACTUAL_V6_DELIVERY)

    assert receipt["record_id"] == "E463_TPU_V6_OPERATOR_DELIVERY_20261008_R1"
    assert receipt["provider_target"]["canonical_kernel_id"] == gate.V6_KERNEL
    assert receipt["push"]["submitted_kernel_id"] == gate.V6_SUBMITTED_KERNEL
    assert receipt["packet"]["source_commit"] == "22aff7b3e733a20ec42637f76110bbd248d84452"


def test_v6_consumer_constants_match_actual_frozen_producer() -> None:
    manifest_path = ACTUAL_V6_PACKET / "packet_manifest.json"
    probe_path = ACTUAL_V6_PACKET / "packet/probe.py"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert gate.sha256_file(manifest_path) == gate.V6_MANIFEST_SHA256
    assert manifest["code_sha256"] == gate.V6_CODE_SHA256
    assert gate.sha256_file(probe_path) == gate.V6_CODE_SHA256

    module = ast.parse(probe_path.read_text(encoding="utf-8"), filename=str(probe_path))
    literals = {}
    for statement in module.body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            if isinstance(target, ast.Name):
                try:
                    literals[target.id] = ast.literal_eval(statement.value)
                except (ValueError, TypeError):
                    continue
    assert literals["PACKET_ID"] == gate.V6_PACKET_ID
    assert set(literals["REQUIRED_ASSET_PATHS"]) == set(gate.REQUIRED_V6_ASSETS)
    assert literals["ASSET_MANIFEST_SHA256"] == gate.V6_ASSET_MANIFEST_SHA256
    assert literals["PORTABLE_INPUTS_SHA256"] == gate.V6_PORTABLE_INPUTS_SHA256
    assert literals["SOURCE_MANIFEST_SHA256"] == gate.V6_SOURCE_MANIFEST_SHA256
    assert literals["SOURCE_ARCHIVE_SHA256"] == gate.V6_SOURCE_ARCHIVE_SHA256


def test_v6_bundle_consumes_complete_terminal_and_source_qualified_inputs(tmp_path: Path) -> None:
    bundle, input_manifest, project = _write_bundle(tmp_path)

    result = gate.validate_e463_v6_bundle(bundle, input_manifest, project)

    assert result["status"] == "passed"
    assert result["provider_kernel"] == gate.V6_KERNEL
    assert result["precision_scope"].startswith("V6 admits only the FP32")
    assert result["v6_input_identity"]["evaluation_group_count"] == 468


@pytest.mark.parametrize(
    ("provider_status", "canonical_kernel"),
    [("QUEUED", gate.V6_KERNEL), ("ERROR", gate.V6_KERNEL), ("COMPLETE", "vladkochetov07/other")],
)
def test_v6_bundle_rejects_nonterminal_or_wrong_kernel(
    tmp_path: Path, provider_status: str, canonical_kernel: str
) -> None:
    bundle, input_manifest, project = _write_bundle(
        tmp_path, provider_status=provider_status, canonical_kernel=canonical_kernel
    )

    with pytest.raises(ValueError, match="terminal receipt"):
        gate.validate_e463_v6_bundle(bundle, input_manifest, project)


def test_v6_bundle_rejects_mutated_collected_update_receipt(tmp_path: Path) -> None:
    bundle, input_manifest, project = _write_bundle(tmp_path)
    (bundle / "outputs/realdata_update_receipt.json").write_text("{}\n")

    with pytest.raises(ValueError, match="identity differs"):
        gate.validate_e463_v6_bundle(bundle, input_manifest, project)


def test_v6_bundle_rejects_partial_terminal_output_collection(tmp_path: Path) -> None:
    bundle, input_manifest, project = _write_bundle(tmp_path)
    (bundle / "outputs/realdata_update_receipt.json").unlink()

    with pytest.raises(ValueError, match="incomplete"):
        gate.validate_e463_v6_bundle(bundle, input_manifest, project)


def test_v6_update_checkpoint_hashes_bind_collected_roundtrip_files(tmp_path: Path) -> None:
    bundle, _, _ = _write_bundle(tmp_path)
    outputs = {path.name: path for path in (bundle / "outputs").iterdir()}
    update = json.loads((bundle / "outputs/realdata_update_receipt.json").read_text())
    gate._validate_update_checkpoint_artifacts(update, outputs)

    update["checkpoint"]["ema_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="round-trip receipt differs"):
        gate._validate_update_checkpoint_artifacts(update, outputs)


def test_v6_bundle_rejects_data_identity_drift(tmp_path: Path) -> None:
    bundle, input_manifest, project = _write_bundle(tmp_path)
    manifest = json.loads(input_manifest.read_text())
    for row in manifest["files"]:
        if row["path"] == "datasets/valid.parquet":
            row["sha256"] = "0" * 64
    _write_json(input_manifest, manifest)

    with pytest.raises(ValueError, match="raw valid parquet identities differ"):
        gate.validate_e463_v6_bundle(bundle, input_manifest, project)


def test_v6_bundle_rejects_unqualified_membership_alias(tmp_path: Path) -> None:
    bundle, input_manifest, project = _write_bundle(tmp_path)
    update = json.loads((bundle / "outputs/realdata_update_receipt.json").read_text())
    realdata = json.loads((bundle / "outputs/realdata_input_identity.json").read_text())
    checkpoint = json.loads((bundle / "outputs/e463_checkpoint_identity.json").read_text())
    update["source_qualified_membership"]["qualified_membership"][0] = ["train", 0]

    with pytest.raises(ValueError, match="source-qualified"):
        gate._validate_manifest_identity(
            input_manifest,
            project,
            realdata,
            checkpoint,
            update,
        )


def test_actual_precision_gate_requires_real_finite_both_arm_evidence() -> None:
    report = {"actual_precision_gate": _precision_report()}

    accepted = gate.validate_actual_precision_gate(report)

    assert accepted["arms"]["bf16"]["completed_updates"] == 40
    report["actual_precision_gate"]["arms"]["bf16"]["actual_data"] = False
    with pytest.raises(ValueError, match="bf16"):
        gate.validate_actual_precision_gate(report)


def test_actual_precision_gate_requires_shared_batches_and_all_diagnostics() -> None:
    report = {"actual_precision_gate": _precision_report()}
    report["actual_precision_gate"]["arms"]["bf16"]["batch_data_sha256"] = ["c" * 64, "d" * 64]
    with pytest.raises(ValueError, match="different actual host batches"):
        gate.validate_actual_precision_gate(report)

    report = {"actual_precision_gate": _precision_report()}
    report["actual_precision_gate"]["paired_step_diagnostics"][0]["loss_abs_delta"] = 0.5
    with pytest.raises(ValueError, match="inconsistent delta"):
        gate.validate_actual_precision_gate(report)


def test_fit_authorization_binds_profile_source_and_v6_terminal(tmp_path: Path) -> None:
    profile = {"actual_precision_gate": _precision_report()}
    profile_path = tmp_path / "profile.json"
    _write_json(profile_path, profile)
    preflight = {"v6_terminal_receipt_sha256": "e" * 64}
    source = {"worker.py": "f" * 64}
    authorization = {
        "schema": "wnn.e465-fit-authorization.v1",
        "status": "approved",
        "approved_by": "root",
        "fit_authorized": True,
        "approved_precisions": ["fp32", "bf16"],
        "profile_report_sha256": gate.sha256_file(profile_path),
        "precision_gate_sha256": hashlib.sha256(gate._json_bytes(profile["actual_precision_gate"])).hexdigest(),
        "input_manifest_sha256": "d" * 64,
        "source_hashes": source,
        "v6_terminal_receipt_sha256": preflight["v6_terminal_receipt_sha256"],
        "whole_job_limit_seconds": gate.E465_MAX_WORKER_SECONDS,
        "public_forecast": None,
    }
    authorization_path = tmp_path / "fit-auth.json"
    _write_json(authorization_path, authorization)

    assert gate.validate_fit_authorization(
        authorization_path,
        profile_path=profile_path,
        profile_report=profile,
        input_manifest_sha256="d" * 64,
        source_identity=source,
        preflight_identity=preflight,
        max_worker_seconds=gate.E465_MAX_WORKER_SECONDS,
    )["status"] == "approved"
    profile_path.write_text("{}\n")
    with pytest.raises(ValueError, match="root fit authorization"):
        gate.validate_fit_authorization(
            authorization_path,
            profile_path=profile_path,
            profile_report=profile,
            input_manifest_sha256="d" * 64,
            source_identity=source,
            preflight_identity=preflight,
            max_worker_seconds=gate.E465_MAX_WORKER_SECONDS,
        )


def test_worker_profile_gate_rejects_nonfinite_resource_metrics(tmp_path: Path) -> None:
    source = {"src/wnn_jax/e465_tpu_worker.py": "f" * 64}
    preflight = {
        "v6_delivery_receipt_sha256": "a" * 64,
        "v6_terminal_receipt_sha256": "b" * 64,
        "v6_realdata_update_sha256": "c" * 64,
    }
    arms = {
        precision: {
            "batch_count": 2,
            "attempt_count": 40,
            "completed_update_count": 40,
            "final_state_finite": True,
            "mean_fullshape_batch_wall_seconds": 1.0,
            "peak_host_rss_bytes": 1024,
            "peak_device_fraction": 0.5,
        }
        for precision in ("fp32", "bf16")
    }
    report = {
        "format": "wnn.e465-tpu-fullshape-profile",
        "status": "profile_complete",
        "input_manifest_sha256": "d" * 64,
        "source_hashes": source,
        "preflight_delivery_receipt_sha256": preflight["v6_delivery_receipt_sha256"],
        "preflight_terminal_receipt_sha256": preflight["v6_terminal_receipt_sha256"],
        "preflight_realdata_update_sha256": preflight["v6_realdata_update_sha256"],
        "profile_batches": 2,
        "global_batch_size": 32,
        "sequence_rows": 20_000,
        "chunk_rows": 1_000,
        "device_inventory": [{"kind": "TPU v5 lite"}] * 8,
        "arms": arms,
        "actual_precision_gate": _precision_report(),
        "projected_training_seconds": 100.0,
    }
    assert gate.validate_profile_report(
        report,
        input_manifest_sha256="d" * 64,
        source_identity=source,
        preflight_identity=preflight,
    )["status"] == "profile_complete"

    report["arms"]["bf16"]["peak_device_fraction"] = float("nan")
    with pytest.raises(ValueError, match="resource/update gate"):
        gate.validate_profile_report(
            report,
            input_manifest_sha256="d" * 64,
            source_identity=source,
            preflight_identity=preflight,
        )
