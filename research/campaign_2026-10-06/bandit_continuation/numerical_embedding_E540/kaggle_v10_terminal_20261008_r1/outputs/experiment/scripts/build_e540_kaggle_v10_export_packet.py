#!/usr/bin/env python3
"""Build a private export-only packet for the completed E540 V8/V9 checkpoints."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from e540_export_validation import (  # noqa: E402
    sha256_file,
    validate_fit_artifact_metadata,
    validate_pair_metadata,
)
from kaggle_assets import prepare as prepare_private_assets  # noqa: E402
from kaggle_bundle import build_bundle  # noqa: E402
from kaggle_common import canonical_sha256  # noqa: E402

LINEAGE = ROOT / "research/campaign_2026-10-06/bandit_continuation/numerical_embedding_E540"
PACKET_DIR = LINEAGE / "v10_export_only"
PACKET_ID = "E540_FP32_V10_FIXED_CHECKPOINT_EXPORT_ONLY"
KERNEL_DATASET = "vladkochetov07/e540-v10-export-only-20261008"
ASSET_DATASET = "vladkochetov07/e540-v10-artifact-assets-20261008"
SPEC_NAME = "e540-v10-fixed-checkpoint-export-20261008"
RECIPE_PATH = PACKET_DIR / "RECIPE_V10_EXPORT_ONLY.json"
BINDING_PATH = PACKET_DIR / "E540_V10_ARTIFACT_BINDING.json"
AUTH_PATH = PACKET_DIR / "E540_V10_ROOT_EXPORT_AUTHORIZATION.json"
CARD_PATH = PACKET_DIR / "EXPERIMENT_CARD_FP32_V10_EXPORT_ONLY.json"
SUBSET_PATH = LINEAGE / "E540_FIT_SUBSET_V5.json"
SPLIT_PATH = ROOT / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
PORTABLE_PATH = ROOT / "research/campaign_2026-10-04/portable_inputs.json"
EVIDENCE_DIR = PACKET_DIR / "source_evidence"
SOURCE_EVIDENCE = [
    EVIDENCE_DIR / "E540_V9_TERMINAL_OPERATOR_RECEIPT_20261008_R1.json",
    EVIDENCE_DIR / "E540_V9_ROOT_FINAL_PACKET_REVIEW_20261008_R1.json",
    EVIDENCE_DIR / "E540_V9_ROOT_INDEPENDENT_SOURCE_REVIEW_20261008_R1.json",
    EVIDENCE_DIR / "E540_V9_PLE_FIXED_EP0_ROOT_TENSOR_CHECK_20261008_R1.json",
    EVIDENCE_DIR / "E540_V8_RESULT.json",
    EVIDENCE_DIR / "E540_V9_RESULT.json",
    EVIDENCE_DIR / "E540_V9_SOURCE_PACKET.json",
    EVIDENCE_DIR / "E540_V8_CONTROL_MANIFEST.json",
]
V8_TERMINAL_PATH = LINEAGE / "E540_V8_TERMINAL_OPERATOR_RECEIPT_20261008_R1.json"
RUN_NAMES = {
    "linear": "e540v8-linear-s540-fp32",
    "ple": "e540v9-ple-s540-fp32",
}
ASSET_FILENAMES = {
    "linear_model": ("linear/model.pt", "model.pt"),
    "linear_epoch0": ("linear/model_ep0.pt", "model_ep0.pt"),
    "linear_provenance": ("linear/provenance.json", "provenance.json"),
    "linear_history": ("linear/history.json", "history.json"),
    "linear_control_manifest": ("linear/control-manifest.json", "control-manifest.json"),
    "ple_model": ("ple/model.pt", "model.pt"),
    "ple_epoch0": ("ple/model_ep0.pt", "model_ep0.pt"),
    "ple_provenance": ("ple/provenance.json", "provenance.json"),
    "ple_history": ("ple/history.json", "history.json"),
}
ARTIFACTS = [
    "e540_v10_export",
    "artifacts/runs/e540v8-linear-s540-fp32",
    "artifacts/runs/e540v9-ple-s540-fp32",
    "cache/pred_e540v8-linear-s540-fp32-eval_valid.npy",
    "cache/pred_e540v8-linear-s540-fp32-eval_valid.json",
    "cache/pred_e540v9-ple-s540-fp32-eval_valid.npy",
    "cache/pred_e540v9-ple-s540-fp32-eval_valid.json",
]


def write_json(path: Path, value: object) -> None:
    """Write deterministic finite JSON metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def descriptor(path: Path, logical_name: str | None = None) -> dict[str, object]:
    return {
        "path": logical_name or path.relative_to(ROOT).as_posix(),
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _git_head() -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _source_paths() -> list[Path]:
    return [
        Path("scripts/build_e540_kaggle_v10_export_packet.py"),
        Path("scripts/e540_cli_preflight.py"),
        Path("scripts/e540_export_validation.py"),
        Path("scripts/e540_kaggle_v10_export_worker.py"),
        Path("scripts/kaggle_assets.py"),
        Path("scripts/kaggle_bundle.py"),
        Path("scripts/kaggle_common.py"),
        Path("scripts/kaggle_gpu_runner.py"),
        Path("scripts/kaggle_kernel.py"),
        Path("scripts/predict_run.py"),
        Path("scripts/prepare_e540_fit_subset.py"),
        Path("scripts/train_rnn.py"),
        Path("src/wnn/__init__.py"),
        *sorted(
            path.relative_to(ROOT)
            for path in (ROOT / "src/wnn").rglob("*.py")
            if path.name != "__init__.py"
        ),
    ]


def _verify_result_artifacts(
    result_path: Path,
    run_dir: Path,
    run_name: str,
    *,
    result_name: str,
    result_expected_status: str,
) -> dict[str, dict[str, object]]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("name") != result_name or result.get("status") != result_expected_status:
        raise ValueError(f"{run_name} terminal result identity/status differs")
    entries = {
        item["path"]: item
        for item in result.get("artifacts", {}).get("files", [])
        if isinstance(item, dict) and isinstance(item.get("path"), str)
    }
    verified = {}
    for filename in ("model.pt", "model_ep0.pt", "provenance.json", "history.json"):
        path = run_dir / filename
        relative = f"artifacts/runs/{run_name}/{filename}"
        entry = entries.get(relative)
        if (
            not isinstance(entry, dict)
            or not path.is_file()
            or path.stat().st_size != entry.get("size")
            or sha256_file(path) != entry.get("sha256")
        ):
            raise ValueError(f"{run_name} artifact does not match its terminal result: {filename}")
        verified[filename] = {"size": path.stat().st_size, "sha256": sha256_file(path)}
    return verified


def _artifact_binding(
    linear_run_dir: Path,
    ple_run_dir: Path,
    v8_result_path: Path,
    v9_result_path: Path,
    output_dir: Path,
) -> tuple[dict[str, Any], Path]:
    subset = json.loads(SUBSET_PATH.read_text(encoding="utf-8"))
    subset_sha = sha256_file(SUBSET_PATH)
    split_sha = sha256_file(SPLIT_PATH)
    if (
        subset_sha != "ce84816336bc457d487cbddf2f7214ef4d8b0b02e83f9c8e60921b9ce2374d4d"
        or split_sha != "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d"
        or subset.get("selected_component_count") != 512
        or subset.get("evaluation_group_count") != 468
    ):
        raise ValueError("E540 V5 subset or fold0 split identity changed")
    linear_run_dir = linear_run_dir.resolve(strict=True)
    ple_run_dir = ple_run_dir.resolve(strict=True)
    linear_files = _verify_result_artifacts(
        v8_result_path,
        linear_run_dir,
        RUN_NAMES["linear"],
        result_name="e540-v8-fp32-20261008",
        result_expected_status="failed",
    )
    ple_files = _verify_result_artifacts(
        v9_result_path,
        ple_run_dir,
        RUN_NAMES["ple"],
        result_name="e540-v9-ple-reuse-control-20261008",
        result_expected_status="failed",
    )

    linear_provenance = json.loads((linear_run_dir / "provenance.json").read_text())
    ple_provenance = json.loads((ple_run_dir / "provenance.json").read_text())
    linear_history = json.loads((linear_run_dir / "history.json").read_text())
    ple_history = json.loads((ple_run_dir / "history.json").read_text())
    subset_with_sha = {**subset, "_sha256": subset_sha}
    linear_summary = validate_fit_artifact_metadata(
        "linear",
        linear_history,
        linear_provenance,
        subset_with_sha,
        model_sha256=linear_files["model.pt"]["sha256"],
        epoch0_sha256=linear_files["model_ep0.pt"]["sha256"],
        expected_run_name=RUN_NAMES["linear"],
    )
    ple_summary = validate_fit_artifact_metadata(
        "ple",
        ple_history,
        ple_provenance,
        subset_with_sha,
        model_sha256=ple_files["model.pt"]["sha256"],
        epoch0_sha256=ple_files["model_ep0.pt"]["sha256"],
        expected_run_name=RUN_NAMES["ple"],
    )
    pair_summary = validate_pair_metadata(linear_provenance, ple_provenance)

    v8_terminal = json.loads(V8_TERMINAL_PATH.read_text(encoding="utf-8"))
    v9_terminal = json.loads(
        (EVIDENCE_DIR / "E540_V9_TERMINAL_OPERATOR_RECEIPT_20261008_R1.json").read_text()
    )
    v9_source_review = json.loads(
        (EVIDENCE_DIR / "E540_V9_ROOT_INDEPENDENT_SOURCE_REVIEW_20261008_R1.json").read_text()
    )
    v9_final_review = json.loads(
        (EVIDENCE_DIR / "E540_V9_ROOT_FINAL_PACKET_REVIEW_20261008_R1.json").read_text()
    )
    ple_tensor_check = json.loads(
        (EVIDENCE_DIR / "E540_V9_PLE_FIXED_EP0_ROOT_TENSOR_CHECK_20261008_R1.json").read_text()
    )
    v8_result = json.loads(v8_result_path.read_text())
    v9_result = json.loads(v9_result_path.read_text())
    if (
        v8_terminal.get("arm_outcomes", {}).get("linear", {}).get("checkpoint_produced") is not True
        or v9_source_review.get("status")
        != "PASS_source_conditional_profile_fit_final_packet_bytes_pending"
        or v9_final_review.get("status") != "PASS_final_bytes_one_quota_admitted_push_authorized"
        or v9_terminal.get("provider_status") != "KernelWorkerStatus.ERROR"
        or v9_terminal.get("ple_fit", {}).get("optimizer_completed_updates") != 72_996
        or v9_terminal.get("ple_fit", {}).get("amp_skipped_updates") != 0
        or v9_terminal.get("ple_fit", {}).get("history_train_seconds") != 559.7
        or v9_terminal.get("ple_fit", {}).get("provenance_fitting_seconds") != 559.8
        or v9_terminal.get("ple_fit", {}).get("predictions") != "not produced"
        or v9_terminal.get("terminal_error", {}).get("validation_function") != "validate_history"
        or v8_terminal.get("artifact_collection", {}).get("provider_result_json_sha256")
        != sha256_file(v8_result_path)
        or v9_terminal.get("runner_result", {}).get("result_json_sha256")
        != sha256_file(v9_result_path)
        or v9_tensor_checkpoint_sha(ple_tensor_check) != ple_files["model_ep0.pt"]["sha256"]
        or ple_tensor_check.get("all_model_tensors_finite") is not True
        or ple_tensor_check.get("checkpoint_epoch") != 0
        or v8_result.get("status") != "failed"
        or v9_result.get("status") != "failed"
    ):
        raise ValueError("Upstream V8/V9 fit, terminal, review or tensor evidence is inconsistent")

    control_manifest = json.loads((EVIDENCE_DIR / "E540_V8_CONTROL_MANIFEST.json").read_text())
    if (
        control_manifest.get("schema") != "wnn.e540-v8-linear-control.v1"
        or control_manifest.get("files", {}).get("model_ep0.pt", {}).get("sha256")
        != linear_files["model_ep0.pt"]["sha256"]
        or control_manifest.get("files", {}).get("model.pt", {}).get("sha256")
        != linear_files["model.pt"]["sha256"]
        or control_manifest.get("files", {}).get("provenance.json", {}).get("sha256")
        != linear_files["provenance.json"]["sha256"]
        or control_manifest.get("files", {}).get("history.json", {}).get("sha256")
        != linear_files["history.json"]["sha256"]
    ):
        raise ValueError("V8 independent source manifest does not bind the reused control")
    portable = json.loads(PORTABLE_PATH.read_text(encoding="utf-8"))
    valid = portable["sources"]["datasets/valid.parquet"]["original_identity"]
    if (
        portable["sources"]["datasets/valid.parquet"]["sha256"]
        != "2861ee30914065702626a3ac033d80d0563cdacaea50c667fc828c2eeb418782"
        or int(valid["bytes"]) != 5_615_509_128
    ):
        raise ValueError("valid.parquet identity differs from the reviewed E540 input")

    stage = output_dir / "artifact_source_stage"
    stage.mkdir(parents=True, exist_ok=False)
    source_by_key = {
        "linear_model": linear_run_dir / "model.pt",
        "linear_epoch0": linear_run_dir / "model_ep0.pt",
        "linear_provenance": linear_run_dir / "provenance.json",
        "linear_history": linear_run_dir / "history.json",
        "linear_control_manifest": EVIDENCE_DIR / "E540_V8_CONTROL_MANIFEST.json",
        "ple_model": ple_run_dir / "model.pt",
        "ple_epoch0": ple_run_dir / "model_ep0.pt",
        "ple_provenance": ple_run_dir / "provenance.json",
        "ple_history": ple_run_dir / "history.json",
    }
    staged_assets: dict[str, dict[str, object]] = {}
    for key, (asset_relative, flat_name) in ASSET_FILENAMES.items():
        target = stage / asset_relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_by_key[key], target)
        staged_assets[key] = {
            "source_path": asset_relative,
            "source_filename": flat_name,
            "size": target.stat().st_size,
            "sha256": sha256_file(target),
            "logical_path": f"e540_assets/{key}{target.suffix or '.json'}",
        }
    private_asset_dir = output_dir / "private_artifact_assets"
    prepare_private_assets(
        stage,
        [entry["source_path"] for entry in staged_assets.values()],
        private_asset_dir,
        ASSET_DATASET,
    )
    asset_manifest = json.loads((private_asset_dir / "assets-manifest.json").read_text())
    by_source = {entry["path"]: entry for entry in asset_manifest["files"]}
    for entry in staged_assets.values():
        uploaded = by_source.get(entry["source_path"])
        if not isinstance(uploaded, dict) or uploaded.get("sha256") != entry["sha256"]:
            raise ValueError("prepared private upload asset differs from its source file")
        entry["upload_filename"] = uploaded["filename"]

    evidence = {
        path.name: {"size": path.stat().st_size, "sha256": sha256_file(path)}
        for path in [*SOURCE_EVIDENCE, V8_TERMINAL_PATH]
    }
    training_source_hashes = {
        "scripts/train_rnn.py": linear_provenance["trainer_source_sha256"],
        "src/wnn/nn.py": linear_provenance["model_source_sha256"],
        "src/wnn/run_identity.py": linear_provenance["run_identity_source_sha256"],
        "src/wnn/numeric_ple.py": linear_provenance["numeric_preprocessing"][
            "module_source_sha256"
        ],
    }
    if any(
        getattr(ple_provenance, "get", lambda *_: None)(field)
        != getattr(linear_provenance, "get", lambda *_: None)(field)
        for field in ("trainer_source_sha256", "model_source_sha256", "run_identity_source_sha256")
    ) or (
        ple_provenance["numeric_preprocessing"]["module_source_sha256"]
        != training_source_hashes["src/wnn/numeric_ple.py"]
    ):
        raise ValueError("paired fits do not share the exact reviewed training sources")
    binding = {
        "schema": "wnn.e540-v10-artifact-binding.v1",
        "packet_id": PACKET_ID,
        "source_fits": {
            "linear": {
                "source_packet": "E540_FP32_V8",
                "run_name": RUN_NAMES["linear"],
                "terminal_result_sha256": sha256_file(v8_result_path),
                "terminal_fit_reuse_validation": "V9 verified and reused the V8 control; no refit in V9 or V10",
                "artifacts": linear_files,
                "fit_metadata_sha256": {
                    "history.json": sha256_file(linear_run_dir / "history.json"),
                    "provenance.json": sha256_file(linear_run_dir / "provenance.json"),
                },
                "duration_validation": linear_summary["duration"],
            },
            "ple": {
                "source_packet": "E540_FP32_V9_REUSE_CONTROL",
                "run_name": RUN_NAMES["ple"],
                "terminal_result_sha256": sha256_file(v9_result_path),
                "terminal_receipt_sha256": evidence[
                    "E540_V9_TERMINAL_OPERATOR_RECEIPT_20261008_R1.json"
                ]["sha256"],
                "root_tensor_review_sha256": evidence[
                    "E540_V9_PLE_FIXED_EP0_ROOT_TENSOR_CHECK_20261008_R1.json"
                ]["sha256"],
                "fit_status": "completed_72996_of_72996_zero_skips",
                "consumer_status": "failed_after_fit_before_prediction_due_to_strict_duration_equality",
                "worker_error": "ValueError: history training duration differs from sibling provenance",
                "artifacts": ple_files,
                "fit_metadata_sha256": {
                    "history.json": sha256_file(ple_run_dir / "history.json"),
                    "provenance.json": sha256_file(ple_run_dir / "provenance.json"),
                },
                "duration_validation": ple_summary["duration"],
            },
        },
        "source_evidence": evidence,
        "v9_source_origin": {
            "source_commit": "703ead63d268b6d8cc1df69ed51e9cae0656b717",
            "source_review_sha256": evidence[
                "E540_V9_ROOT_INDEPENDENT_SOURCE_REVIEW_20261008_R1.json"
            ]["sha256"],
            "final_packet_review_sha256": evidence[
                "E540_V9_ROOT_FINAL_PACKET_REVIEW_20261008_R1.json"
            ]["sha256"],
            "code_manifest_sha256": v9_source_review["code_manifest_sha256"],
            "recipe_sha256": v9_source_review["recipe_sha256"],
            "control_manifest_sha256": sha256_file(EVIDENCE_DIR / "E540_V8_CONTROL_MANIFEST.json"),
        },
        "subset_manifest_sha256": subset_sha,
        "split_manifest_sha256": split_sha,
        "selected_component_count": subset["selected_component_count"],
        "evaluation_group_count": subset["evaluation_group_count"],
        "evaluation_membership_sha256": subset["evaluation_membership_sha256"],
        "paired_fit_identity": pair_summary,
        "training_source_hashes": training_source_hashes,
        "valid_source": {
            "dataset": "vladkochetov07/wunder-funder",
            "path": "datasets/valid.parquet",
            "size": int(valid["bytes"]),
            "sha256": portable["sources"]["datasets/valid.parquet"]["sha256"],
        },
        "assets": staged_assets,
    }
    write_json(BINDING_PATH, binding)
    shutil.rmtree(stage)
    return binding, private_asset_dir


def v9_tensor_checkpoint_sha(receipt: dict[str, Any]) -> object:
    return receipt.get("checkpoint_sha256")


def _ensure_pending_authorization(
    *,
    code_sha: str,
    recipe_sha: str,
    binding_sha: str,
    subset_sha: str,
    binding: dict[str, Any],
) -> dict[str, object]:
    expected = {
        "schema": "wnn.e540-export-only-authorization.v1",
        "owner": "root",
        "status": "pending_root_review",
        "authorized": False,
        "export_only": True,
        "fit_allowed": False,
        "provider_launch_authorized": False,
        "packet_id": PACKET_ID,
        "code_manifest_sha256": code_sha,
        "recipe_sha256": recipe_sha,
        "artifact_binding_sha256": binding_sha,
        "subset_manifest_sha256": subset_sha,
        "kernel_process_wall_seconds_max": 6_000,
        "worker_wall_seconds_max": 5_200,
        "linear_epoch0_sha256": binding["source_fits"]["linear"]["artifacts"]["model_ep0.pt"][
            "sha256"
        ],
        "ple_epoch0_sha256": binding["source_fits"]["ple"]["artifacts"]["model_ep0.pt"]["sha256"],
        "linear_provenance_sha256": binding["source_fits"]["linear"]["fit_metadata_sha256"][
            "provenance.json"
        ],
        "ple_provenance_sha256": binding["source_fits"]["ple"]["fit_metadata_sha256"][
            "provenance.json"
        ],
        "linear_history_sha256": binding["source_fits"]["linear"]["fit_metadata_sha256"][
            "history.json"
        ],
        "ple_history_sha256": binding["source_fits"]["ple"]["fit_metadata_sha256"]["history.json"],
        "v9_terminal_receipt_sha256": binding["source_fits"]["ple"]["terminal_receipt_sha256"],
        "authorization_id": None,
        "prediction_membership_count": 468,
        "prediction_arms": [RUN_NAMES["linear"], RUN_NAMES["ple"]],
        "fit_arms": [],
        "profile_arms": [],
    }
    if AUTH_PATH.exists():
        existing = json.loads(AUTH_PATH.read_text(encoding="utf-8"))
        if existing.get("status") == "approved" and existing.get("authorized") is True:
            for key, value in expected.items():
                if key in {
                    "status",
                    "authorized",
                    "provider_launch_authorized",
                    "authorization_id",
                }:
                    continue
                if existing.get(key) != value:
                    raise ValueError(f"existing root export authorization differs in {key}")
            return existing
        write_json(AUTH_PATH, expected)
        return expected
    write_json(AUTH_PATH, expected)
    return expected


def build_packet(
    output_dir: Path,
    linear_run_dir: Path,
    ple_run_dir: Path,
    v8_result_path: Path,
    v9_result_path: Path,
) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite V10 packet directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(exist_ok=False)
    binding, asset_dir = _artifact_binding(
        linear_run_dir, ple_run_dir, v8_result_path, v9_result_path, output_dir
    )
    subset_sha = sha256_file(SUBSET_PATH)
    recipe_sha = sha256_file(RECIPE_PATH)
    source_paths = _source_paths()
    source_hashes = {path.as_posix(): sha256_file(ROOT / path) for path in source_paths}
    code_descriptors = [descriptor(ROOT / path) for path in source_paths]
    code_sha = canonical_sha256(sorted(code_descriptors, key=lambda item: item["path"]))
    binding_sha = sha256_file(BINDING_PATH)
    authorization = _ensure_pending_authorization(
        code_sha=code_sha,
        recipe_sha=recipe_sha,
        binding_sha=binding_sha,
        subset_sha=subset_sha,
        binding=binding,
    )

    asset_manifest = json.loads((asset_dir / "assets-manifest.json").read_text())
    asset_entries = {entry["path"]: entry for entry in asset_manifest["files"]}
    external = ["datasets/valid.parquet=valid.parquet"]
    external_descriptors = [
        {
            "path": "datasets/valid.parquet",
            "size": binding["valid_source"]["size"],
            "sha256": binding["valid_source"]["sha256"],
        }
    ]
    for _key, asset in binding["assets"].items():
        upload = asset_entries[asset["source_path"]]
        external.append(f"{asset['logical_path']}={ASSET_DATASET}:{upload['filename']}")
        external_descriptors.append(
            {"path": asset["logical_path"], "size": asset["size"], "sha256": asset["sha256"]}
        )

    local_inputs = [
        BINDING_PATH,
        RECIPE_PATH,
        CARD_PATH,
        AUTH_PATH,
        SUBSET_PATH,
        SPLIT_PATH,
        PORTABLE_PATH,
        V8_TERMINAL_PATH,
        *SOURCE_EVIDENCE,
    ]
    local_descriptors = [descriptor(path) for path in local_inputs]
    external_paths = {item["path"] for item in external_descriptors}
    if {item["path"] for item in local_descriptors} & external_paths:
        raise ValueError("V10 local and mounted input paths collide")
    code_manifest = canonical_sha256(sorted(code_descriptors, key=lambda item: item["path"]))
    if code_manifest != code_sha:
        raise ValueError("V10 source code manifest changed during packet construction")
    spec = {
        "version": 1,
        "name": SPEC_NAME,
        "kind": "fixed_checkpoint_pair_prediction_export",
        "argv": ["{python}", "scripts/e540_kaggle_v10_export_worker.py"],
        "code": code_descriptors,
        "code_sha256": code_sha,
        "inputs": [*local_descriptors, *external_descriptors],
        "artifacts": ARTIFACTS,
        "limits": {
            "cpus": 2,
            "ram_bytes": 8 * 1024**3,
            "timeout_seconds": 5_200,
            "gpu_memory_bytes": 4_300_000_000,
            "gpu_stop_bytes": 5_000_000_000,
        },
        "environment": {
            "PYTHONHASHSEED": "0",
            "WNN_THREADS": "2",
            "WNN_LOADER_RAM_GB": "1",
            "WNN_LOADER_WORKERS": "0",
            "WNN_DEFERRED_IDENTITY_PATHS": "datasets/valid.parquet",
            "WNN_E540_PACKET_ID": PACKET_ID,
            "WNN_E540_AUTHORIZATION_FILENAME": AUTH_PATH.name,
            "WNN_E540_AUTHORIZATION_PATH": AUTH_PATH.relative_to(ROOT).as_posix(),
            "WNN_E540_CODE_MANIFEST_SHA256": code_sha,
            "WNN_E540_RECIPE_SHA256": recipe_sha,
            "WNN_E540_ARTIFACT_BINDING_SHA256": binding_sha,
            "WNN_E540_SUBSET_SHA256": subset_sha,
            "WNN_E540_KERNEL_WALL_SECONDS": "6000",
            "WNN_E540_WORKER_WALL_SECONDS": "5200",
            "WNN_E540_LINEAR_EPOCH0_SHA256": binding["source_fits"]["linear"]["artifacts"][
                "model_ep0.pt"
            ]["sha256"],
            "WNN_E540_PLE_EPOCH0_SHA256": binding["source_fits"]["ple"]["artifacts"][
                "model_ep0.pt"
            ]["sha256"],
            "WNN_E540_LINEAR_PROVENANCE_SHA256": binding["source_fits"]["linear"][
                "fit_metadata_sha256"
            ]["provenance.json"],
            "WNN_E540_PLE_PROVENANCE_SHA256": binding["source_fits"]["ple"]["fit_metadata_sha256"][
                "provenance.json"
            ],
            "WNN_E540_LINEAR_HISTORY_SHA256": binding["source_fits"]["linear"][
                "fit_metadata_sha256"
            ]["history.json"],
            "WNN_E540_PLE_HISTORY_SHA256": binding["source_fits"]["ple"]["fit_metadata_sha256"][
                "history.json"
            ],
            "WNN_E540_V9_TERMINAL_RECEIPT_SHA256": binding["source_fits"]["ple"][
                "terminal_receipt_sha256"
            ],
        },
    }
    spec_path = output_dir / f"{SPEC_NAME}.spec.json"
    write_json(spec_path, spec)
    build_bundle(
        ROOT,
        [spec_path],
        output_dir / "bundle",
        KERNEL_DATASET,
        external,
        [ASSET_DATASET],
    )
    packet = {
        "schema": "wnn.e540-kaggle-export-cohort.v1",
        "packet_id": PACKET_ID,
        "status": "prepared_pending_root_source_and_final_byte_review",
        "fit_admitted": False,
        "provider_launch_authorized": False,
        "grade_owner": "root independent grader",
        "generated_from": {
            "source_checkout": str(ROOT),
            "git_head_before_packet_commit": _git_head(),
            "source_files_hash_bound": source_hashes,
            "code_manifest_sha256": code_sha,
        },
        "recipe": {"path": str(RECIPE_PATH.relative_to(ROOT)), "sha256": recipe_sha},
        "experiment_card": {
            "path": str(CARD_PATH.relative_to(ROOT)),
            "sha256": sha256_file(CARD_PATH),
        },
        "artifact_binding": {
            "path": str(BINDING_PATH.relative_to(ROOT)),
            "sha256": binding_sha,
            "linear_epoch0_sha256": binding["source_fits"]["linear"]["artifacts"]["model_ep0.pt"][
                "sha256"
            ],
            "ple_epoch0_sha256": binding["source_fits"]["ple"]["artifacts"]["model_ep0.pt"][
                "sha256"
            ],
            "asset_manifest_sha256": sha256_file(asset_dir / "assets-manifest.json"),
            "private_asset_upload_directory": str(asset_dir),
            "fit_completed_before_packet": True,
        },
        "data": {
            "valid_source": binding["valid_source"],
            "subset_manifest_sha256": subset_sha,
            "split_manifest_sha256": sha256_file(SPLIT_PATH),
            "evaluation_group_count": 468,
            "only_valid_parquet_mounted": True,
            "no_target_npz_mount": True,
            "no_local_full_parquet_footer_or_target_npz_read": True,
        },
        "authorization": {
            "path": str(AUTH_PATH.relative_to(ROOT)),
            "sha256": sha256_file(AUTH_PATH),
            "status": authorization["status"],
            "owner": "root",
            "provider_launch_authorized": authorization.get("provider_launch_authorized") is True,
        },
        "execution": {
            "kernel_process_wall_seconds_max": 6_000,
            "worker_wall_seconds_max": 5_200,
            "runner_finalization_reserve_seconds": 300,
            "profile_arms": [],
            "fit_arms": [],
            "prediction_arms": ["linear", "ple"],
            "grade_performed": False,
            "provider_upload_performed": False,
            "provider_push_performed": False,
        },
        "bundle": {
            "archive_sha256": sha256_file(output_dir / "bundle/bundle.wnn"),
            "manifest_sha256": sha256_file(output_dir / "bundle/bundle-manifest.json"),
            "bootstrap_sha256": sha256_file(output_dir / "bundle/kernel/kernel.py"),
            "kernel_metadata_sha256": sha256_file(
                output_dir / "bundle/kernel/kernel-metadata.json"
            ),
        },
    }
    write_json(output_dir / "E540_KAGGLE_PACKET_V10.json", packet)
    return packet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--linear-run-dir", type=Path, required=True)
    parser.add_argument("--ple-run-dir", type=Path, required=True)
    parser.add_argument("--v8-result", type=Path, required=True)
    parser.add_argument("--v9-result", type=Path, required=True)
    args = parser.parse_args()
    packet = build_packet(
        args.output_dir,
        args.linear_run_dir,
        args.ple_run_dir,
        args.v8_result,
        args.v9_result,
    )
    print(json.dumps(packet, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
