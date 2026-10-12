#!/usr/bin/env python3
"""Validate R4 packet/source bindings without importing data or model libraries."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_runner(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("r4_static_validation_subject", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load R4 runner source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-root", type=Path, required=True)
    parser.add_argument("--worktree-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main_root = args.main_root.resolve(strict=True)
    worktree_root = args.worktree_root.resolve(strict=True)
    packet_rel = Path("research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/native_single_member_cleanfold_q8_r4")
    r4_packet = worktree_root / packet_rel
    contract_path = r4_packet / "CONTRACT_SINGLE_R4.json"
    freeze_path = r4_packet / "PREPARATION_FREEZE_R4.json"
    runner_path = r4_packet / "run_single_controls_r4.py"
    contract = json_object(contract_path)
    freeze = json_object(freeze_path)
    contract_sha = sha256_file(contract_path)
    if freeze.get("contract_sha256") != contract_sha:
        raise ValueError("R4 preparation freeze does not match contract bytes")
    if contract.get("repository", {}).get("absolute_root") != str(main_root):
        raise ValueError("R4 contract does not bind the exact MAIN root")

    runner = load_runner(runner_path)
    r3_rel = packet_rel.parent / "native_single_member_cleanfold_q8_r3"
    r3_path = main_root / r3_rel / "CONTRACT_SINGLE_R3.json"
    preflight_path = main_root / r3_rel / "INPUT_PREFLIGHT_MAIN_R1.json"
    for name, source_binding in contract["source_bindings"].items():
        binding_root = worktree_root if name in {"r4_consumer", "r4_builder"} else main_root
        bound_path = runner.resolve_repo_path(binding_root, source_binding["repo_path"])
        if sha256_file(bound_path) != source_binding["sha256"]:
            raise ValueError(f"R4 source binding changed in its owning tree: {name}")
    r3 = json_object(r3_path)
    preflight = json_object(preflight_path)
    if sha256_file(r3_path) != contract["r3_binding"]["contract_sha256"]:
        raise ValueError("MAIN R3 contract differs from the R4 pin")
    if contract.get("r3_producer_receipts") != r3.get("producer_receipts"):
        raise ValueError("R4 producer receipt map differs from the frozen MAIN R3 contract")
    if contract.get("r3_source_bindings") != r3.get("source_bindings"):
        raise ValueError("R4 source map differs from the frozen MAIN R3 contract")
    if sha256_file(preflight_path) != contract["r3_binding"]["input_preflight_sha256"]:
        raise ValueError("MAIN R3 input-preflight differs from the R4 pin")
    if preflight.get("status") != "PASS_FROZEN_FOOTER_AND_METADATA_ONLY" or preflight.get("prediction_data_loaded") is not False or preflight.get("full_fold_score_performed") is not False:
        raise ValueError("MAIN R3 preflight is not the expected metadata-only pass")
    identity_keys = ("bytes", "footer_bytes", "footer_sha256_including_8_byte_trailer")
    if {key: preflight.get("valid_dataset_identity", {}).get(key) for key in identity_keys} != {key: r3["valid_dataset"][key] for key in identity_keys}:
        raise ValueError("MAIN R3 preflight footer identity differs from the frozen contract")
    verified_main_receipts = 0
    for name, receipt_binding in contract["r3_producer_receipts"].items():
        main_copy = runner.resolve_repo_path(main_root, receipt_binding["repo_path"])
        if sha256_file(main_copy) != receipt_binding["sha256"]:
            raise ValueError(f"MAIN R3 producer receipt changed: {name}")
        verified_main_receipts += 1
    data_path = runner.resolve_repo_path(main_root, r3["valid_dataset"]["repo_path"])
    plan = runner.prediction_plan(r3)
    data_stat = data_path.stat()
    if data_stat.st_size != r3["valid_dataset"]["bytes"]:
        raise ValueError("MAIN valid.parquet byte size differs from the frozen R3 identity")
    preflight_stat = preflight["valid_dataset_identity"]
    if (data_stat.st_ino, data_stat.st_mtime_ns) != (preflight_stat.get("inode"), preflight_stat.get("mtime_ns")):
        raise ValueError("MAIN valid.parquet stat identity differs from the exact input preflight")

    admission_path = main_root / packet_rel.parent / "native_single_member_cleanfold_q8_r3" / "ROOT_ADMISSION_R3.json"
    callback_path = main_root / contract["callback_fixture"]["repo_path"]
    output_path = main_root / contract["output"]["root_repo_path"]
    dependency_probe = subprocess.run(
        [
            "/usr/bin/python3.11",
            "-c",
            "import importlib.util,json; print(json.dumps({n: importlib.util.find_spec(n) is not None for n in ('numpy','pyarrow','torch','ruff')}, sort_keys=True))",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    dependency_result: dict[str, Any] = {
        "executable": "/usr/bin/python3.11",
        "returncode": dependency_probe.returncode,
        "stdout": dependency_probe.stdout.strip(),
        "stderr": dependency_probe.stderr.strip()[:500],
    }
    parsed_dependencies = json.loads(dependency_probe.stdout) if dependency_probe.returncode == 0 else {}
    missing = sorted(name for name in ("numpy", "pyarrow", "torch") if parsed_dependencies.get(name) is not True)
    output = {
        "schema": "wnn.native.single-member-cleanfold-q8-preparation-validation.r4",
        "status": "PASS_STATIC_PACKET_BINDINGS_BLOCKED_RUNTIME_GATES",
        "created_utc": datetime.now(UTC).isoformat(),
        "worktree_root": str(worktree_root),
        "main_root": str(main_root),
        "r4_contract_sha256": contract_sha,
        "r4_runner_sha256": sha256_file(runner_path),
        "r4_builder_sha256": sha256_file(r4_packet / "build_contract_r4.py"),
        "r3_contract_sha256": sha256_file(main_root / contract["r3_binding"]["contract_repo_path"]),
        "r3_preflight_sha256": sha256_file(main_root / contract["r3_binding"]["input_preflight_repo_path"]),
        "r3_preflight_repo_path": contract["r3_binding"]["input_preflight_repo_path"],
        "r3_producer_receipt_count_rehashed_in_main": len(contract["r3_producer_receipts"]),
        "r3_producer_receipts_verified_in_main": verified_main_receipts,
        "r3_source_binding_count_rehashed": len(contract["source_bindings"]),
        "fold_plan": {
            fold: {
                "groups": r3["folds"][fold]["evaluation_groups"],
                "evaluation_group_list_sha256": r3["folds"][fold]["evaluation_group_list_sha256"],
                "component_overlap_count": r3["folds"][fold]["fit_evaluation_component_overlap_count"],
            }
            for fold in runner.FOLDS
        },
        "prediction_plan": {
            name: {"shape": item["shape"], "dtype": item["dtype"], "bytes": item["bytes"], "payload_bytes": item["payload_bytes"], "npy_header_bytes": item["npy_header_bytes"]}
            for name, item in plan.items()
        },
        "valid_parquet_stat": {"bytes": data_stat.st_size, "inode": data_stat.st_ino, "mtime_ns": data_stat.st_mtime_ns},
        "valid_parquet_footer_rechecked": False,
        "root_admission_present": admission_path.is_file(),
        "callback_fixture_present": callback_path.is_file(),
        "output_root_already_exists": output_path.exists(),
        "python311_dependency_probe": dependency_result,
        "python311_missing_runtime_dependencies": missing,
        "runtime_status": "BLOCKED_PENDING_R3_ROOT_ADMISSION_CALLBACK_FIXTURE_AND_PY311_DEPENDENCIES",
        "inference_performed": False,
        "score_performed": False,
        "provider_action_performed": False,
        "notes": [
            "This preparation validator checks metadata/path/source bindings only; it does not hash the full Parquet body or read row groups.",
            "The R4 runner rechecks the live Parquet footer on its open descriptor immediately before row-group access.",
            "The stdlib tests and any Python 3.13 syntax checks are not serving or inference validation.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"validation": str(args.output), "status": output["status"], "contract_sha256": contract_sha}, sort_keys=True))


if __name__ == "__main__":
    main()
