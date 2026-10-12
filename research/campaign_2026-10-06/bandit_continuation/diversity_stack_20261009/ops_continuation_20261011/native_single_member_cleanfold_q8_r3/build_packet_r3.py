#!/usr/bin/env python3
"""Build an R3 root-side input-admission contract from frozen R2 evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

MAIN_ROOT = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
OPS_REL = Path(
    "research/campaign_2026-10-06/bandit_continuation/"
    "diversity_stack_20261009/ops_continuation_20261011"
)
PACKET_REL = OPS_REL / "native_single_member_cleanfold_q8_r3"
R2_SINGLE_REL = OPS_REL / "native_single_member_cleanfold_q8_r2"
R2_SHARED_REL = OPS_REL / "native_shared_cleanfold_q8_r2"
R1_SHARED_REL = OPS_REL / "native_shared_cleanfold_q8_r1"
SHARED_HELPERS_REL = OPS_REL / "cpu_shared_backbone_r1"

R3_CODE = {
    "packet_builder": PACKET_REL / "build_packet_r3.py",
    "input_validator": PACKET_REL / "validate_input_identity_r3.py",
}
CURRENT_MAIN_SOURCES = {
    "official_metric": Path("src/wnn/gmetric.py"),
    "project_utils": Path("utils.py"),
    "callback_runner": Path("scripts/benchmark_submission_callbacks.py"),
    "wrapped_predictor": SHARED_HELPERS_REL / "benchmark_shared_f0.py",
    "r1_quality_consumer": R1_SHARED_REL / "run_quality.py",
    "r2_quality_consumer": R2_SHARED_REL / "run_quality_r2.py",
    "r2_shared_contract_builder": R2_SHARED_REL / "build_contract_r2.py",
    "r2_single_control_consumer": R2_SINGLE_REL / "run_single_controls_r2.py",
    "r2_single_contract_builder": R2_SINGLE_REL / "build_contract_single_r2.py",
}
R2_RECEIPTS = {
    "single_control_contract_r2": R2_SINGLE_REL / "CONTRACT_SINGLE_R2.json",
    "single_control_uv_incident_r1": R2_SINGLE_REL / "PREPARATION_INCIDENT_UV_R1.json",
    "python311_dependency_probe_r1": R2_SINGLE_REL / "PY311_RUNTIME_DEPENDENCY_PROBE_R1.json",
    "supplementary_callback_fixture_r2": R2_SINGLE_REL
    / "CALLBACK_FIXTURE_SINGLE_R2_SUPPLEMENTARY_PY313.json",
    "shared_contract_r2": R2_SHARED_REL / "CONTRACT_R2.json",
    "shared_terminal_handoff_r2": R2_SHARED_REL / "TERMINAL_HANDOFF_R2.json",
    "shared_root_admission_r2": R2_SHARED_REL / "ROOT_ADMISSION_R2.json",
    "shared_callback_fixture_r2": R2_SHARED_REL / "CALLBACK_FIXTURE_R2.json",
    "shared_quality_result_r2": R2_SHARED_REL / "quality_output_r2/QUALITY_RESULT_R2.json",
    "shared_contract_r1": R1_SHARED_REL / "CONTRACT_R1.json",
}
IDENTITY_KEYS = (
    "bytes",
    "footer_bytes",
    "footer_sha256_including_8_byte_trailer",
)


def sha256_file(path: Path) -> str:
    """Hash a file with bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from an existing frozen receipt."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def bind_repo_path(relative: Path, *, tree_root: Path) -> dict[str, str]:
    """Bind a repository-relative file and reject paths outside its tree."""
    path = (tree_root / relative).resolve(strict=True)
    try:
        path.relative_to(tree_root.resolve(strict=True))
    except ValueError as exc:
        raise ValueError(f"binding escapes repository: {relative}") from exc
    return {"repo_path": relative.as_posix(), "sha256": sha256_file(path)}


def footer_identity(receipt: dict[str, Any], label: str) -> dict[str, Any]:
    """Return a complete footer identity from a known receipt shape."""
    if label == "shared_contract":
        source = receipt["valid_dataset"]
    elif label in {"root_admission", "terminal_handoff"}:
        source = receipt["bindings"]["valid_dataset_footer_identity"]
    elif label == "single_contract":
        source = receipt["valid_dataset"]
    else:
        raise ValueError(f"unknown identity source: {label}")
    identity = {key: source[key] for key in IDENTITY_KEYS}
    if not isinstance(identity["bytes"], int) or not isinstance(identity["footer_bytes"], int):
        raise ValueError(f"non-integer Parquet identity in {label}")
    if len(identity["footer_sha256_including_8_byte_trailer"]) != 64:
        raise ValueError(f"malformed footer SHA256 in {label}")
    return identity


def build_contract(tree_root: Path) -> dict[str, Any]:
    """Build a path-portable contract without rewriting any R2 packet file."""
    if MAIN_ROOT.resolve(strict=True) != MAIN_ROOT:
        raise ValueError("the declared MAIN root is not the actual canonical path")
    if tree_root.resolve(strict=True) == MAIN_ROOT.resolve(strict=True):
        raise ValueError("build the R3 packet in its isolated worktree")

    packet_dir = tree_root / PACKET_REL
    if packet_dir.resolve(strict=True) != (tree_root / PACKET_REL).resolve(strict=True):
        raise ValueError("R3 packet path resolution changed")
    shared = read_json(tree_root / R2_SHARED_REL / "CONTRACT_R2.json")
    admission = read_json(tree_root / R2_SHARED_REL / "ROOT_ADMISSION_R2.json")
    terminal = read_json(tree_root / R2_SHARED_REL / "TERMINAL_HANDOFF_R2.json")
    single = read_json(tree_root / R2_SINGLE_REL / "CONTRACT_SINGLE_R2.json")

    identities = {
        "shared_contract": footer_identity(shared, "shared_contract"),
        "root_admission": footer_identity(admission, "root_admission"),
        "terminal_handoff": footer_identity(terminal, "terminal_handoff"),
        "single_contract": footer_identity(single, "single_contract"),
    }
    if len({json.dumps(value, sort_keys=True) for value in identities.values()}) != 1:
        raise ValueError("R2 frozen input receipts disagree on valid.parquet identity")
    if single["valid_dataset"]["path"] != str(MAIN_ROOT / "datasets/valid.parquet"):
        raise ValueError("R2 single-control packet names a different valid.parquet path")
    if shared["valid_dataset"]["path"] != str(MAIN_ROOT / "datasets/valid.parquet"):
        raise ValueError("R2 shared-control packet names a different valid.parquet path")

    source_paths = {**CURRENT_MAIN_SOURCES, **R3_CODE}
    current_sources = {
        name: bind_repo_path(relative, tree_root=tree_root)
        for name, relative in source_paths.items()
    }
    expected_source_hashes = {
        "official_metric": shared["source_bindings"]["official_metric"]["sha256"],
        "project_utils": shared["source_bindings"]["project_utils"]["sha256"],
        "callback_runner": shared["source_bindings"]["callback_runner"]["sha256"],
        "wrapped_predictor": shared["source_bindings"]["wrapped_predictor_source"]["sha256"],
        "r1_quality_consumer": shared["source_bindings"]["r1_quality_consumer"]["sha256"],
        "r2_quality_consumer": shared["source_bindings"]["r2_consumer"]["sha256"],
        "r2_shared_contract_builder": shared["source_bindings"]["r2_contract_builder"]["sha256"],
        "r2_single_control_consumer": single["source_bindings"]["single_control_consumer"][
            "sha256"
        ],
        "r2_single_contract_builder": single["source_bindings"]["contract_builder"]["sha256"],
    }
    for name, expected_sha in expected_source_hashes.items():
        if current_sources[name]["sha256"] != expected_sha:
            raise ValueError(f"current-main source differs from frozen R2 source: {name}")
    dependency_probe = read_json(
        tree_root / R2_SINGLE_REL / "PY311_RUNTIME_DEPENDENCY_PROBE_R1.json"
    )
    if dependency_probe.get("status") != "blocked_missing_dependencies":
        raise ValueError("the frozen Python 3.11 dependency blocker status changed")
    incident = read_json(tree_root / R2_SINGLE_REL / "PREPARATION_INCIDENT_UV_R1.json")
    if incident.get("process", {}).get("status") != "exited":
        raise ValueError("the preserved environment-setup incident is not terminal")
    r2_receipts = {
        name: bind_repo_path(relative, tree_root=tree_root)
        for name, relative in R2_RECEIPTS.items()
    }
    external_runtime_assets = {
        "weights_archive": single["producer_binding"]["weights_archive"],
        "weights_receipt": single["producer_binding"]["weights_receipt"],
        "native_elf": single["runtime"]["loaded_elf_path"],
        "native_package_init": single["runtime"]["native_package_dir"] + "/wnn_native/__init__.py",
        "native_wheel": single["source_bindings"]["native_wheel"]["path"],
    }
    for name, value in list(external_runtime_assets.items()):
        if isinstance(value, dict):
            path = value["path"]
            digest = value["sha256"]
        else:
            path = value
            key = {
                "native_elf": "native_elf",
                "native_package_init": "native_package_init",
                "native_wheel": "native_wheel",
            }[name]
            digest = single["source_bindings"][key]["sha256"]
        external_runtime_assets[name] = {"path": path, "sha256": digest}

    expected_valid = dict(single["valid_dataset"])
    expected_valid.update(
        {
            "repo_path": "datasets/valid.parquet",
            "identity_receipt_sources": {
                name: r2_receipts[name]
                for name in (
                    "shared_contract_r2",
                    "shared_terminal_handoff_r2",
                    "shared_root_admission_r2",
                    "single_control_contract_r2",
                )
            },
        }
    )
    expected_valid.pop("path", None)
    folds = json.loads(json.dumps(single["folds"]))
    for fold in folds.values():
        for model in fold["models"].values():
            for artifact in ("checkpoint", "provenance"):
                path = Path(model[artifact]["path"]).resolve(strict=True)
                try:
                    model[artifact]["repo_path"] = path.relative_to(MAIN_ROOT).as_posix()
                except ValueError as exc:
                    raise ValueError(f"model artifact is outside MAIN: {path}") from exc
                model[artifact].pop("path", None)

    contract = {
        "schema": "wnn.native.single-member-cleanfold-q8-contract.r3",
        "packet_id": "native-q8-single-model-controls-packet-r3",
        "status": "prepared_pending_py311_dependencies_root_admission_and_execution",
        "repository": {
            "absolute_root": str(MAIN_ROOT),
            "base_main_commit": "bf3ea0740f3e1a4fd4675a87ea5ba1295144bc55",
            "packet_paths_are_repository_relative": True,
            "validator_requires_this_exact_root": True,
        },
        "r2_preservation": {
            "r2_packet_directory": R2_SINGLE_REL.as_posix(),
            "historical_r2_files_modified": False,
            "r2_contract": r2_receipts["single_control_contract_r2"],
            "historical_errata_and_uv_incident_retained": True,
        },
        "research_scope": {
            "claim": "standalone native Q8 student and HFT controls on exposed F0/F1 partitions from the already completed fixed-output R2 comparison",
            "training": False,
            "inference_performed": False,
            "full_fold_score_performed": False,
            "fresh_holdout": False,
            "public_forecast": False,
            "blend_search": False,
            "source_membership_or_recipe_changed": False,
        },
        "inherited_endpoint": single["endpoint"],
        "folds": folds,
        "metric": {
            "id": single["metric"]["id"],
            "repo_path": "src/wnn/gmetric.py",
            "sha256": current_sources["official_metric"]["sha256"],
            "mask": single["metric"]["mask"],
            "clip": single["metric"]["clip"],
            "weight": single["metric"]["weight"],
            "aggregation": single["metric"]["aggregation"],
        },
        "valid_dataset": expected_valid,
        "feature_column_count": shared["callback_contract"]["feature_columns"],
        "expected_output": {
            "callback_fixture_repo_path": (
                PACKET_REL / "CALLBACK_FIXTURE_SINGLE_R3_PY311.json"
            ).as_posix(),
            "output_root_repo_path": (PACKET_REL / "quality_output_single_r3").as_posix(),
            "overwrite_policy": single["expected_output"]["overwrite_policy"],
            "result_file": "SINGLE_MEMBER_RESULT_R3.json",
            "prediction_files": single["expected_output"]["prediction_files"],
            "total_prediction_bytes": single["expected_output"]["total_prediction_bytes"],
        },
        "frozen_input_receipts": identities,
        "source_bindings": current_sources,
        "producer_receipts": r2_receipts,
        "external_runtime_assets": external_runtime_assets,
        "runtime": {
            "serving_python": "/usr/bin/python3.11",
            "required_python_version_prefix": [3, 11],
            "full_scoring_dependencies": ["numpy", "pyarrow", "torch"],
            "dependency_status": "blocked_missing_dependencies_per_frozen_R2_probe",
            "new_environment_or_package_install_authorized": False,
        },
        "input_gate": {
            "required_before_model_import_construction_or_prediction_data_load": True,
            "command": "scripts/limited .venv/bin/python <R3 packet>/validate_input_identity_r3.py --output <R3 packet>/INPUT_PREFLIGHT_R3.json",
            "preflight_interpreter_scope": "metadata-only gate; this does not replace or weaken the CPython 3.11 serving-runtime requirement",
            "must_pass_receipt_schema": "wnn.native.single-member-input-preflight.r3",
            "root_admission_must_bind_preflight_sha256": True,
            "reads_full_valid_parquet_body": False,
            "reads_footer_and_metadata_only": True,
        },
        "execution_gate": {
            "root_admission_required": True,
            "root_admission_schema": "wnn.root.native-single-member-cleanfold-quality-admission.r3",
            "decision": "PASS_FOR_ONE_FROZEN_FULL_CLEANFOLD_SINGLE_MODEL_Q8_EVALUATION",
            "py311_callback_fixture_required_before_full_score": True,
            "output_directory_create_only": True,
        },
        "resource_proposal": {
            "cpu_cores": 1,
            "memory_max_bytes": 1500000000,
            "thread_count": 1,
            "gpu": False,
            "provider_actions": False,
            "new_spend_usd": 0,
            "full_score_wall_seconds": 1800,
            "timing_status": "pending_root_admission; R2 shared endpoint runtime is not transferred as a standalone guarantee",
        },
    }
    return contract


def main() -> None:
    """Write the deterministic contract create-only."""
    tree_root = Path(__file__).resolve().parents[6]
    packet = tree_root / PACKET_REL
    output = packet / "CONTRACT_SINGLE_R3.json"
    if output.exists():
        raise FileExistsError(f"refusing to replace frozen R3 contract: {output}")
    contract = build_contract(tree_root)
    output.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"contract_path": str(output), "status": contract["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
