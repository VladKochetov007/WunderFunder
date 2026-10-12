#!/usr/bin/env python3
"""Bind the native single-member controls to completed R2 producers."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PACKET = Path(__file__).resolve().parent
R2_PACKET = PACKET.parent / "native_shared_cleanfold_q8_r2"
CONTRACT_PATH = PACKET / "CONTRACT_SINGLE_R2.json"
INCIDENT_PATH = PACKET / "PREPARATION_INCIDENT_UV_R1.json"
MAIN_PYTHON = "/home/vlad/development/wunder/wnn_connectome_starterpack/.venv/bin/python"
SERVING_PYTHON = "/usr/bin/python3.11"
SUPPLEMENTARY_FIXTURE_PATH = PACKET / "CALLBACK_FIXTURE_SINGLE_R2_SUPPLEMENTARY_PY313.json"
PY311_DEPENDENCY_PROBE_PATH = PACKET / "PY311_RUNTIME_DEPENDENCY_PROBE_R1.json"


def sha256_file(path: Path) -> str:
    """Hash a file with bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def bind(path: Path) -> dict[str, str]:
    """Return the resolved path and content digest for an existing input."""
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from a frozen campaign receipt."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def build_contract() -> dict[str, Any]:
    """Build a source-bound plan for student-only and HFT-only controls."""
    r2_path = R2_PACKET / "CONTRACT_R2.json"
    terminal_path = R2_PACKET / "TERMINAL_HANDOFF_R2.json"
    fixture_path = R2_PACKET / "CALLBACK_FIXTURE_R2.json"
    r2_contract = read_json(r2_path)
    terminal = read_json(terminal_path)
    quality_path = Path(terminal["bindings"]["quality_result_path"]).resolve(strict=True)
    quality = read_json(quality_path)
    fixture = read_json(fixture_path)
    supplementary_fixture = read_json(SUPPLEMENTARY_FIXTURE_PATH)
    dependency_probe = read_json(PY311_DEPENDENCY_PROBE_PATH)
    incident = read_json(INCIDENT_PATH)

    r1_path = Path(r2_contract["producer_binding"]["r1_contract"]["path"])
    r1_contract = read_json(r1_path)
    if quality.get("status") != "PASS_FULL_EXPOSED_CROSSFOLD_Q8_OUTPUT_BLEND":
        raise ValueError("R2 fixed-half output control is not a completed grade")
    if quality.get("endpoint", {}).get("coefficients") != [0.5, 0.5]:
        raise ValueError("R2 reference is not the fixed half-output endpoint")
    if terminal.get("bindings", {}).get("quality_result_sha256") != sha256_file(quality_path):
        raise ValueError("R2 terminal receipt does not bind its quality result")
    if terminal.get("bindings", {}).get("contract_sha256") != sha256_file(r2_path):
        raise ValueError("R2 terminal receipt is bound to another contract")
    if terminal.get("bindings", {}).get("quality_result_sha256") != sha256_file(quality_path):
        raise ValueError("R2 terminal receipt is bound to another quality result")
    if fixture.get("status") != "PASS_ACTUAL_COMPLETE_SEQUENCE_CALLBACK_AND_RESET_REPLAY":
        raise ValueError("R2 actual callback/reset fixture is not a pass")
    if supplementary_fixture.get("runtime", {}).get("python_version_prefix") != [3, 13]:
        raise ValueError("preserved supplementary callback fixture is not CPython 3.13")
    if supplementary_fixture.get("quality_evaluation_performed") is not False:
        raise ValueError("supplementary callback fixture unexpectedly evaluated quality")
    if dependency_probe.get("python_executable") != SERVING_PYTHON:
        raise ValueError("Py3.11 dependency probe targets a different interpreter")
    if dependency_probe.get("status") != "blocked_missing_dependencies":
        raise ValueError("Py3.11 dependency status is not the observed blocked state")
    if incident.get("schema") != "wnn.preparation.uv_environment_incident.r1":
        raise ValueError("preparation incident receipt schema differs")
    if incident.get("process", {}).get("status") != "exited":
        raise ValueError("the bounded uv process is not recorded as exited")

    folds: dict[str, Any] = {}
    for fold in ("0", "1"):
        r1_fold = r1_contract["folds"][fold]
        terminal_fold = terminal["evaluation"]["folds"][fold]
        quality_fold = quality["folds"][fold]
        if terminal_fold["fit_evaluation_component_overlap_count"] != 0:
            raise ValueError(f"fold {fold} has fit/evaluation component overlap")
        if terminal_fold["evaluation_groups"] != quality_fold["complete_sequences_scored"]:
            raise ValueError(f"fold {fold} R2 group count differs between receipts")
        if (
            terminal_fold["evaluation_group_list_sha256"]
            != quality_fold["evaluation_groups_canonical_sha256"]
        ):
            raise ValueError(f"fold {fold} R2 group ordering differs between receipts")
        models: dict[str, Any] = {}
        for role in ("student", "hft"):
            source_model = r1_fold["models"][role]
            terminal_model = terminal_fold["source_qualified_models"][role]
            checkpoint = Path(source_model["checkpoint_path"]).resolve(strict=True)
            provenance = Path(source_model["provenance_path"]).resolve(strict=True)
            checkpoint_sha = sha256_file(checkpoint)
            provenance_sha = sha256_file(provenance)
            if checkpoint_sha != source_model["checkpoint_sha256"]:
                raise ValueError(f"fold {fold}/{role} checkpoint hash changed")
            if provenance_sha != source_model["provenance_sha256"]:
                raise ValueError(f"fold {fold}/{role} provenance hash changed")
            for key in (
                "checkpoint_sha256",
                "provenance_sha256",
                "run",
                "epoch",
                "fit_source",
                "fit_membership_identity",
                "initial_checkpoint_sha256",
            ):
                if terminal_model.get(key) != source_model.get(key):
                    raise ValueError(f"fold {fold}/{role} R2 model identity differs: {key}")
            models[role] = {
                "run": source_model["run"],
                "epoch": source_model["epoch"],
                "fit_source": source_model["fit_source"],
                "checkpoint": {
                    "path": str(checkpoint),
                    "sha256": checkpoint_sha,
                    "bytes": checkpoint.stat().st_size,
                },
                "provenance": {"path": str(provenance), "sha256": provenance_sha},
                "initial_checkpoint_sha256": source_model["initial_checkpoint_sha256"],
                "fit_membership_identity": source_model["fit_membership_identity"],
            }
        if (
            models["student"]["fit_source"] != "train+valid"
            or models["hft"]["fit_source"] != "valid"
        ):
            raise ValueError(f"fold {fold} producer fit-source semantics changed")
        folds[fold] = {
            "evaluation_groups": terminal_fold["evaluation_groups"],
            "evaluation_group_list_sha256": terminal_fold["evaluation_group_list_sha256"],
            "evaluation_component_count": terminal_fold["evaluation_component_count"],
            "excluded_train_groups": terminal_fold["excluded_train_groups"],
            "fit_evaluation_component_overlap_count": 0,
            "split_manifest_sha256": r1_fold["split_manifest"]["sha256"],
            "models": models,
            "fixed_half_output_reference": quality_fold["metric"],
            "fixed_half_prediction_sha256": quality_fold["prediction_sha256"],
            "fixed_half_prediction_bytes": quality_fold["prediction_bytes"],
        }

    source_bindings = {
        "contract_builder": bind(PACKET / "build_contract_single_r2.py"),
        "single_control_consumer": bind(PACKET / "run_single_controls_r2.py"),
    }
    source_bindings.update(r2_contract["source_bindings"])
    valid = r2_contract["valid_dataset"]
    per_fold_bytes = {
        fold: {
            "shape": [folds[fold]["evaluation_groups"], 20_000, 2],
            "dtype": "float32",
            "bytes": folds[fold]["evaluation_groups"] * 20_000 * 2 * 4 + 128,
        }
        for fold in ("0", "1")
    }
    return {
        "schema": "wnn.native.single-member-cleanfold-q8-contract.r2",
        "packet_id": "native-q8-single-model-controls-packet-r2",
        "candidate_id": "oldz4_f0_f1_student_only_hft_only_native_q8_r2",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "prepared_pending_py311_runtime_validation_and_root_execution_admission",
        "scope": {
            "claim": "native Q8 standalone student and HFT scores on the same exposed F0/F1 partitions used by R2",
            "folds_are_exposed": True,
            "fresh_holdout": False,
            "training": False,
            "public_forecast": False,
            "blend_search": False,
            "parameter_averaging": False,
        },
        "endpoint": {
            "models": ["student_only", "hft_only"],
            "one_member_per_predictor": True,
            "independent_head_weights": True,
            "shared_backbone": False,
            "coefficients": [1.0],
            "fast_activations": True,
            "precision_request": "q8vnni",
            "required_actual_precision": "q8vnni",
            "reference": "completed R2 student/HFT fixed 0.5/0.5 output blend, read only",
        },
        "producer_binding": {
            "r2_contract": bind(r2_path),
            "r2_terminal_handoff": bind(terminal_path),
            "r2_quality_result": bind(quality_path),
            "r2_callback_fixture": bind(fixture_path),
            "preparation_incident_receipt": bind(INCIDENT_PATH),
            "supplementary_callback_fixture": bind(SUPPLEMENTARY_FIXTURE_PATH),
            "supplementary_callback_fixture_contract_sha256": supplementary_fixture[
                "contract_sha256"
            ],
            "python311_dependency_probe": bind(PY311_DEPENDENCY_PROBE_PATH),
            "r1_contract": bind(r1_path),
            "weights_archive": r2_contract["producer_binding"]["r1_weights_archive"],
            "weights_receipt": r2_contract["producer_binding"]["r1_weights_receipt"],
            "member_order": ["student", "hft"],
            "member_array_map": read_json(
                Path(r2_contract["producer_binding"]["r1_weights_receipt"]["path"])
            )["array_map"],
        },
        "valid_dataset": valid,
        "metric": r2_contract["metric"],
        "runtime": {
            "python_executable": SERVING_PYTHON,
            "python_requirement": "CPython 3.11 is required for serving validation; the existing system interpreter currently lacks the recorded dependencies, so full scoring remains blocked until those dependencies are available in an approved existing runtime",
            "python_version_prefix": [3, 11],
            "preparation_runtime": {
                "python_executable": MAIN_PYTHON,
                "python_version": sys.version,
                "python_version_prefix": list(sys.version_info[:2]),
                "qualification": "supplementary only; CPython 3.13 does not satisfy the Py3.11 serving-validation gate",
            },
            "validation_dependency_probe": dependency_probe,
            "native_package_dir": r2_contract["runtime"]["native_package_dir"],
            "loaded_elf_path": r2_contract["runtime"]["loaded_elf_path"],
            "loaded_elf_sha256": r2_contract["runtime"]["loaded_elf_sha256"],
            "cpu_quota_percent": 100,
            "thread_count": 1,
            "memory_max_bytes": 1_500_000_000,
            "wall_timeout_seconds": 1_800,
            "measured_r2_reference_wall_seconds": read_json(terminal_path)["evaluation"][
                "elapsed_quality_wall_seconds"
            ],
            "prospective_wall_note": "1800s is a root-admission ceiling; two standalone passes are not measured by the shared-backbone R2 timing",
        },
        "expected_output": {
            "output_root": str((PACKET / "quality_output_single_r2").resolve()),
            "callback_fixture": str((PACKET / "CALLBACK_FIXTURE_SINGLE_R2_PY311.json").resolve()),
            "prediction_files": {
                f"fold{fold}_{role}": per_fold_bytes[fold]
                for fold in ("0", "1")
                for role in ("student", "hft")
            },
            "total_prediction_bytes": sum(per_fold_bytes[fold]["bytes"] * 2 for fold in ("0", "1")),
            "result_file": "SINGLE_MEMBER_RESULT_R2.json",
            "overwrite_policy": "create-only; preexisting output root or prediction is a hard refusal",
        },
        "folds": folds,
        "execution_gate": {
            "required_root_receipt_schema": "wnn.root.native-single-member-cleanfold-q8-admission.r2",
            "required_decision": "PASS_FOR_ONE_FROZEN_CROSSFOLD_NATIVE_SINGLE_MEMBER_EVALUATION",
            "required_bindings": [
                "contract_sha256",
                "r2_contract_sha256",
                "r2_terminal_handoff_sha256",
                "r2_quality_result_sha256",
                "r2_callback_fixture_sha256",
                "preparation_incident_receipt_sha256",
                "supplementary_callback_fixture_sha256",
                "python311_dependency_probe_sha256",
                "standalone_callback_fixture_sha256",
                "weights_archive_sha256",
                "weights_receipt_sha256",
                "native_elf_sha256",
                "valid_dataset_footer_identity",
                "fold_membership_sha256s",
                "checkpoint_provenance_sha256s",
                "output_root_absent",
            ],
        },
        "source_bindings": source_bindings,
    }


def main() -> None:
    """Write one create-only contract after verifying every producer checkpoint."""
    if CONTRACT_PATH.exists():
        raise FileExistsError(f"refusing to replace frozen contract: {CONTRACT_PATH}")
    contract = build_contract()
    CONTRACT_PATH.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"contract": str(CONTRACT_PATH), "status": contract["status"]}))


if __name__ == "__main__":
    main()
