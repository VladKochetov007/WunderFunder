#!/usr/bin/env python3
"""Freeze an R3-aware single-member Q8 consumer without running inference."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

MAIN_ROOT = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
R3_REL = PurePosixPath("research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/native_single_member_cleanfold_q8_r3")
R4_REL = PurePosixPath("research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/native_single_member_cleanfold_q8_r4")
EXPECTED_R3_SHA = "312b57d7cff1386be2cb727a30ce66133588db1804fb50180ef96768374eea12"
PREFLIGHT_NAME = "INPUT_PREFLIGHT_MAIN_R1.json"
EXPECTED_PREFLIGHT_SHA = "e2b7bab342ee35dc1122f88909f06d9f980188d80a6471341a9ec9104352264c"


def sha256_file(path: Path) -> str:
    """Hash a file incrementally."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def repo_path(root: Path, relative: str) -> Path:
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or "\\" in relative:
        raise ValueError(f"unsafe repository path: {relative}")
    result = (root / Path(*path.parts)).resolve(strict=True)
    result.relative_to(root.resolve(strict=True))
    return result


def binding(root: Path, relative: str) -> dict[str, str]:
    return {"repo_path": relative, "sha256": sha256_file(repo_path(root, relative))}


def build(root: Path, main_root: Path = MAIN_ROOT) -> tuple[dict[str, Any], dict[str, Any]]:
    root = root.resolve(strict=True)
    main_root = main_root.resolve(strict=True)
    if not (root / ".git").exists():
        raise ValueError("project root is not a Git worktree")
    if main_root != MAIN_ROOT:
        raise ValueError("R4 source authority is tied to the exact MAIN repository root")
    r3_path = main_root / Path(*R3_REL.parts) / "CONTRACT_SINGLE_R3.json"
    preflight_path = main_root / Path(*R3_REL.parts) / PREFLIGHT_NAME
    r3_sha = sha256_file(r3_path)
    preflight_sha = sha256_file(preflight_path)
    if r3_sha != EXPECTED_R3_SHA or preflight_sha != EXPECTED_PREFLIGHT_SHA:
        raise ValueError("committed R3 contract/preflight differs from the reviewed source snapshot")
    r3 = json_object(r3_path)
    preflight = json_object(preflight_path)
    if r3.get("schema") != "wnn.native.single-member-cleanfold-q8-contract.r3":
        raise ValueError("unexpected R3 contract schema")
    if preflight.get("schema") != "wnn.native.single-member-input-preflight.r3" or preflight.get("status") != "PASS_FROZEN_FOOTER_AND_METADATA_ONLY":
        raise ValueError("R3 metadata-only input receipt is not a pass")
    if preflight.get("contract_sha256") != r3_sha:
        raise ValueError("R3 input preflight is bound to different contract bytes")
    if preflight.get("repository_root") != str(main_root):
        raise ValueError("R3 input preflight is bound to a different repository root")
    if preflight.get("valid_dataset_repo_path") != r3["valid_dataset"]["repo_path"]:
        raise ValueError("R3 input preflight names a different dataset path")
    identity_keys = ("bytes", "footer_bytes", "footer_sha256_including_8_byte_trailer")
    expected_identity = {key: r3["valid_dataset"][key] for key in identity_keys}
    observed_identity = {key: preflight.get("valid_dataset_identity", {}).get(key) for key in identity_keys}
    if observed_identity != expected_identity:
        raise ValueError("R3 MAIN input preflight footer identity differs from the frozen contract")
    if preflight.get("prediction_data_loaded") is not False or preflight.get("full_fold_score_performed") is not False:
        raise ValueError("R3 MAIN input preflight must remain metadata-only")
    data_path = repo_path(main_root, r3["valid_dataset"]["repo_path"])
    data_stat = data_path.stat()
    stat_identity = preflight["valid_dataset_identity"]
    if (data_stat.st_size, data_stat.st_ino, data_stat.st_mtime_ns) != (
        stat_identity.get("bytes"), stat_identity.get("inode"), stat_identity.get("mtime_ns")
    ):
        raise ValueError("MAIN valid.parquet stat identity differs from the pinned input preflight")

    r3_rel = R3_REL.as_posix()
    r4_rel = R4_REL.as_posix()
    source_bindings: dict[str, dict[str, str]] = {
        "r3_contract": binding(main_root, f"{r3_rel}/CONTRACT_SINGLE_R3.json"),
        "r3_input_preflight_main": binding(main_root, f"{r3_rel}/{PREFLIGHT_NAME}"),
        "r3_input_validator": binding(main_root, f"{r3_rel}/validate_input_identity_r3.py"),
        "r4_consumer": binding(root, f"{r4_rel}/run_single_controls_r4.py"),
        "r4_builder": binding(root, f"{r4_rel}/build_contract_r4.py"),
    }
    for name, item in r3["source_bindings"].items():
        if sha256_file(repo_path(main_root, item["repo_path"])) != item["sha256"]:
            raise ValueError(f"R3 source binding changed in MAIN: {name}")
        source_bindings[f"r3_{name}"] = {"repo_path": item["repo_path"], "sha256": item["sha256"]}
    for r3_name, r4_name in (
        ("callback_runner", "callback_runner"),
        ("project_utils", "project_utils"),
        ("official_metric", "official_metric"),
        ("wrapped_predictor", "wrapped_predictor"),
        ("r1_quality_consumer", "r1_weight_consumer"),
    ):
        item = r3["source_bindings"][r3_name]
        source_bindings[r4_name] = {"repo_path": item["repo_path"], "sha256": item["sha256"]}

    r1_binding = r3["producer_receipts"]["shared_contract_r1"]
    r1_path = repo_path(main_root, r1_binding["repo_path"])
    if sha256_file(r1_path) != r1_binding["sha256"]:
        raise ValueError("R3 source fold contract changed")
    r1_contract = json_object(r1_path)
    folds: dict[str, Any] = {}
    for fold in ("0", "1"):
        source_fold = r3["folds"][fold]
        folds[fold] = {
            "evaluation_groups": source_fold["evaluation_groups"],
            "evaluation_group_list_sha256": source_fold["evaluation_group_list_sha256"],
            "evaluation_component_count": source_fold["evaluation_component_count"],
            "fit_evaluation_component_overlap_count": source_fold["fit_evaluation_component_overlap_count"],
            "exposure": r1_contract["folds"][fold]["exposure"],
            "fixed_half_output_reference": source_fold["fixed_half_output_reference"],
            "models": {},
        }
        if source_fold["fit_evaluation_component_overlap_count"] != 0:
            raise ValueError(f"R3 fold {fold} has source-qualified overlap")
        for role in ("student", "hft"):
            item = source_fold["models"][role]
            folds[fold]["models"][role] = {
                "run": item["run"],
                "epoch": item["epoch"],
                "fit_source": item["fit_source"],
                "checkpoint_repo_path": item["checkpoint"]["repo_path"],
                "checkpoint_sha256": item["checkpoint"]["sha256"],
                "checkpoint_bytes": item["checkpoint"]["bytes"],
                "provenance_repo_path": item["provenance"]["repo_path"],
                "provenance_sha256": item["provenance"]["sha256"],
                "initial_checkpoint_sha256": item["initial_checkpoint_sha256"],
                "fit_membership_identity": item["fit_membership_identity"],
            }

    external = r3["external_runtime_assets"]
    runtime_assets = {
        name: {"absolute_path": item["path"], "sha256": item["sha256"]}
        for name, item in external.items()
    }
    native_package = Path(external["native_package_init"]["path"]).parent
    runtime_assets["native_package"] = {"absolute_path": str(native_package)}
    receipts = {name: dict(item) for name, item in r3["producer_receipts"].items()}
    for name, item in receipts.items():
        if sha256_file(repo_path(main_root, item["repo_path"])) != item["sha256"]:
            raise ValueError(f"R3 producer receipt changed in MAIN: {name}")
    output = {
        "root_repo_path": r3["expected_output"]["output_root_repo_path"],
        "callback_fixture_repo_path": r3["expected_output"]["callback_fixture_repo_path"],
        "result_filename": r3["expected_output"]["result_file"],
        "prediction_files": r3["expected_output"]["prediction_files"],
        "overwrite_policy": "create-only; root and prediction files must be absent",
    }
    contract = {
        "schema": "wnn.native.single-member-cleanfold-q8-contract.r4",
        "status": "prepared_pending_r3_root_admission_and_py311_runtime",
        "created_utc": datetime.now(UTC).isoformat(),
        "repository": {"absolute_root": str(MAIN_ROOT), "actual_root": str(MAIN_ROOT), "r3_source_tree_base_commit": "bf3ea0740f3e1a4fd4675a87ea5ba1295144bc55"},
        "r3_binding": {
            "contract_repo_path": f"{r3_rel}/CONTRACT_SINGLE_R3.json",
            "contract_sha256": r3_sha,
            "input_preflight_repo_path": f"{r3_rel}/{PREFLIGHT_NAME}",
            "input_preflight_sha256": preflight_sha,
            "valid_dataset_repo_path": r3["valid_dataset"]["repo_path"],
            "valid_dataset_identity": {key: r3["valid_dataset"][key] for key in ("bytes", "footer_bytes", "footer_sha256_including_8_byte_trailer")},
            "preflight_stat_identity": {key: preflight["valid_dataset_identity"][key] for key in ("inode", "mtime_ns")},
        },
        "r3_producer_receipts": receipts,
        "r3_source_bindings": {name: dict(item) for name, item in r3["source_bindings"].items()},
        "source_bindings": source_bindings,
        "models": folds,
        "weights": {
            "archive_absolute_path": external["weights_archive"]["path"],
            "archive_sha256": external["weights_archive"]["sha256"],
            "receipt_absolute_path": external["weights_receipt"]["path"],
            "receipt_sha256": external["weights_receipt"]["sha256"],
            "member_order": ["student", "hft"],
        },
        "runtime_assets": runtime_assets,
        "metric": r3["metric"],
        "valid_dataset": {key: r3["valid_dataset"][key] for key in ("repo_path", "bytes", "footer_bytes", "footer_sha256_including_8_byte_trailer", "rows", "row_groups")},
        "feature_column_count": r3["feature_column_count"],
        "fixed_execution": {"folds": ["0", "1"], "roles": ["student", "hft"], "member_count_per_prediction": 1, "precision": "q8vnni", "shared_backbone": False, "blend_search": False, "parameter_averaging": False, "public_forecast": False},
        "callback_fixture": {"repo_path": output["callback_fixture_repo_path"], "schema": "wnn.native.single-member-cleanfold-q8-callback-fixture.r3", "status": "REQUIRED_BEFORE_FULL_SCORE", "sha256": None, "required_before_full_score": True},
        "output": output,
        "root_admission": {"repo_path": f"{r3_rel}/ROOT_ADMISSION_R3.json", "schema": r3["execution_gate"]["root_admission_schema"], "decision": r3["execution_gate"]["decision"]},
        "resource_budget": {"cpu_cores": 1, "ram_max_bytes": 1_500_000_000, "wall_seconds": 1800, "gpu": False, "new_spend_usd": 0, "thread_count": 1, "provider_actions": False},
        "runtime": {"python_executable": "/usr/bin/python3.11", "python_version": [3, 11], "required_dependencies": ["numpy", "pyarrow", "torch"], "environment_creation_or_install": False},
        "limitations": ["F0/F1 are exposed source-qualified folds, not fresh holdouts", "single seed lineage per fold/role", "standalone single-member controls only; no blend search or parameter averaging", "score claims remain blocked until an actual Py3.11 callback fixture and root admission pass"],
    }
    freeze = {
        "schema": "wnn.native.single-member-cleanfold-q8-preparation-freeze.r4",
        "status": "FROZEN_CODE_ONLY_PENDING_RUNTIME_GATES",
        "contract_repo_path": f"{r4_rel}/CONTRACT_SINGLE_R4.json",
        "contract_sha256": hashlib.sha256((json.dumps(contract, indent=2, sort_keys=True) + "\n").encode()).hexdigest(),
        "r3_contract_sha256": r3_sha,
        "r3_input_preflight_sha256": preflight_sha,
        "consumer_sha256": source_bindings["r4_consumer"]["sha256"],
        "builder_sha256": source_bindings["r4_builder"]["sha256"],
        "root_admission_created": False,
        "runtime_executed": False,
        "score_claim": False,
    }
    return contract, freeze


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[6], help="isolated worktree where R4 contract/freeze will be written")
    parser.add_argument("--main-root", type=Path, default=MAIN_ROOT, help="fixed repository containing R3 receipts, source pins, and data")
    args = parser.parse_args()
    contract, freeze = build(args.project_root, args.main_root)
    packet = args.project_root.resolve(strict=True) / Path(*R4_REL.parts)
    contract_path = packet / "CONTRACT_SINGLE_R4.json"
    freeze_path = packet / "PREPARATION_FREEZE_R4.json"
    for path, value in ((contract_path, contract), (freeze_path, freeze)):
        with path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"contract": str(contract_path), "freeze": str(freeze_path), "contract_sha256": freeze["contract_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
