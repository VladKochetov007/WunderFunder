from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

PACKET_ROOT = Path(__file__).resolve().parent
R5_PACKET_ROOT = PACKET_ROOT.parent / "oldz4_student_core_hft_f0_r5"
R6_CROP_PACKET_ROOT = PACKET_ROOT.parent / "oldz4_student_core_hft_f0_r6"
CONTRACT_PATH = PACKET_ROOT / "HFT_PROFILE_CONTRACT_R6.json"
LAUNCH_RECEIPT_PATH = PACKET_ROOT / "ROOT_HFT_PROFILE_LAUNCH_R6.json"
PROJECT_ROOT = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
R5_RUNNER_PATH = R5_PACKET_ROOT / "run_hft_stage_r5.py"


class _StopBeforeDevice(RuntimeError):
    """Signal the exact trainer entrypoint reached the device boundary."""


def sha256_file(path: Path) -> str:
    """Hash a file incrementally so source checks stay bounded in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    """Read and validate a JSON object from a frozen packet path."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _load_r5_runner() -> ModuleType:
    """Import the unchanged R5 consumer that verifies frozen source and inputs."""
    spec = importlib.util.spec_from_file_location("oldz4_hft_stage_r5", R5_RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import the frozen R5 stage runner: {R5_RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _remove_profile_options(
    argv: list[str], value_options: set[str], flag_options: set[str]
) -> list[str]:
    """Remove only named options for recipe-equivalence checks."""
    normalized: list[str] = []
    skip_value = False
    for token in argv:
        if skip_value:
            skip_value = False
            continue
        if token in flag_options:
            continue
        if token in value_options:
            skip_value = True
            continue
        normalized.append(token)
    if skip_value:
        raise ValueError("argv ends with a switch that requires a value")
    return normalized


def validate_contract(r5_runner: ModuleType) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind the profile-only correction to unchanged R5 source and fit recipe."""
    contract = read_object(CONTRACT_PATH)
    fit = r5_runner.read_object(r5_runner.FIT_CONTRACT_PATH)
    r5_profile = r5_runner.read_object(r5_runner.PROFILE_CONTRACT_PATH)
    bindings_sha = sha256_file(r5_runner.SOURCE_BINDINGS_PATH)
    if contract.get("schema") != "wnn.oldz4-hft-profile-contract.r6":
        raise ValueError("unexpected R6 profile contract schema")
    if contract.get("source_snapshot_bindings_sha256") != bindings_sha:
        raise ValueError("R6 profile contract is not bound to the R5 source snapshot")
    if contract.get("fit_contract_sha256") != sha256_file(r5_runner.FIT_CONTRACT_PATH):
        raise ValueError("R6 profile contract does not bind the unchanged fit contract")
    if contract.get("r5_profile_contract_sha256") != sha256_file(r5_runner.PROFILE_CONTRACT_PATH):
        raise ValueError("R6 profile contract does not identify the failed R5 profile")
    if contract.get("r5_runner_sha256") != sha256_file(R5_RUNNER_PATH):
        raise ValueError("R5 stage consumer changed after the R6 profile was frozen")
    failed_attempt = contract["profile_correction"]
    failed_receipt_path = PROJECT_ROOT / failed_attempt["failed_r5_attempt_receipt"]
    failed_log_path = PROJECT_ROOT / failed_attempt["failed_r5_log"]
    if sha256_file(failed_receipt_path) != failed_attempt["failed_r5_attempt_receipt_sha256"]:
        raise ValueError("the preserved failed R5 attempt receipt changed")
    if sha256_file(failed_log_path) != failed_attempt["failed_r5_log_sha256"]:
        raise ValueError("the preserved failed R5 profile log changed")
    profile_argv = contract.get("trainer_argv")
    fit_argv = fit.get("trainer_argv")
    if not isinstance(profile_argv, list) or not isinstance(fit_argv, list):
        raise ValueError("R6 profile or unchanged fit argv is missing")
    if "--save-training-state" in profile_argv:
        raise ValueError("profile argv must not request epoch continuation state")
    if "--save-training-state" not in fit_argv:
        raise ValueError("unchanged fit argv lost its required continuation-state save")
    fit_base = _remove_profile_options(fit_argv, {"--name"}, {"--save-training-state"})
    profile_base = _remove_profile_options(
        profile_argv,
        {"--name", "--profile-warmup", "--profile-steps"},
        {"--grad-stats"},
    )
    if fit_base != profile_base:
        raise ValueError("R6 profile changes an HFT fit argument beyond profile-only flags")
    if contract.get("fit_run") != fit.get("run"):
        raise ValueError("R6 profile points at a different fit run")
    if contract.get("run") == r5_profile.get("run"):
        raise ValueError("R6 profile must use a new run identity")
    return contract, fit


def _validate_inputs(r5_runner: ModuleType, project_root: Path) -> dict[str, Any]:
    """Run existing exact source, parent, crop and zero-worker checks."""
    project_root = project_root.resolve()
    if project_root != PROJECT_ROOT:
        raise ValueError(f"project root must be {PROJECT_ROOT}")
    inputs = r5_runner.verify_project_inputs(project_root)
    loader = r5_runner.verify_resource_environment()
    contract = read_object(CONTRACT_PATH)
    crop_contract = r5_runner.read_object(r5_runner.CORE_CONTRACT_PATH)
    crop_path = Path(contract["core_crop_receipt_path"])
    expected_crop_path = project_root / (
        "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/"
        "ops_continuation_20261011/oldz4_student_core_hft_f0_r5/"
        "CORE_CROP_PRODUCER_RECEIPT_R5.json"
    )
    if crop_path != expected_crop_path:
        raise ValueError("crop receipt path differs from the frozen MAIN receipt location")
    crop = read_object(crop_path)
    crop_receipt_sha = sha256_file(crop_path)
    retry_path = R6_CROP_PACKET_ROOT / "CORE_CROP_RETRY_CONTRACT_R6.json"
    consumer_path = R6_CROP_PACKET_ROOT / "prepare_core_crop_r6.py"
    if sha256_file(retry_path) != contract["core_crop_retry_contract_sha256"]:
        raise ValueError("R6 crop retry contract changed after the profile was frozen")
    if sha256_file(consumer_path) != contract["core_crop_consumer_sha256"]:
        raise ValueError("R6 crop consumer changed after the profile was frozen")
    if (
        crop.get("schema") != "wnn.oldz4-core-crop-producer-receipt.v1"
        or crop.get("status") != "PASS_ACTUAL_CORE_CROP"
        or crop.get("consumer_revision") != "R6"
        or crop.get("retry_contract_sha256") != contract["core_crop_retry_contract_sha256"]
        or crop.get("consumer_source_sha256") != contract["core_crop_consumer_sha256"]
        or crop.get("contract_sha256") != sha256_file(r5_runner.CORE_CONTRACT_PATH)
        or crop.get("destination_run") != crop_contract["destination_run"]
    ):
        raise ValueError("actual crop receipt does not bind the frozen R6 producer and R5 contract")
    checkpoint = project_root / crop["output_checkpoint_relative_path"]
    provenance = project_root / crop["output_provenance_relative_path"]
    if (
        sha256_file(checkpoint) != crop["verification"]["output_checkpoint_sha256"]
        or sha256_file(provenance) != crop["verification"]["output_provenance_sha256"]
    ):
        raise ValueError("cropped core checkpoint or provenance changed after its receipt")
    return {
        "project_inputs": inputs,
        "loader": loader,
        "crop_receipt_sha256": crop_receipt_sha,
        "crop_output_checkpoint_sha256": crop["verification"]["output_checkpoint_sha256"],
    }


def _import_trainer(r5_runner: ModuleType, project_root: Path) -> tuple[ModuleType, Path]:
    """Use the same exact c337 import and actual module-global ROOT binding as R5."""
    return r5_runner.import_trainer(project_root)


def validate_actual_cli(project_root: Path) -> dict[str, Any]:
    """Exercise actual c337 main with the exact R6 argv, stopping before device/data."""
    r5_runner = _load_r5_runner()
    contract, _ = validate_contract(r5_runner)
    inputs = _validate_inputs(r5_runner, project_root)
    trainer, trainer_path = _import_trainer(r5_runner, project_root.resolve())
    run_dir = project_root / "artifacts/runs" / contract["run"]
    if run_dir.exists():
        raise FileExistsError(f"R6 profile output already exists: {run_dir}")
    reached_device_boundary = False

    def stop_before_device() -> None:
        nonlocal reached_device_boundary
        reached_device_boundary = True
        raise _StopBeforeDevice("actual c337 configure_torch boundary reached")

    original_configure_torch = trainer.configure_torch
    trainer.configure_torch = stop_before_device
    original_argv = sys.argv
    try:
        sys.argv = [str(trainer_path), *contract["trainer_argv"]]
        with contextlib.suppress(_StopBeforeDevice):
            trainer.main()
    finally:
        trainer.configure_torch = original_configure_torch
        sys.argv = original_argv
    if not reached_device_boundary:
        raise RuntimeError("actual c337 main did not reach the expected device boundary")
    if run_dir.exists():
        raise RuntimeError("CLI preflight unexpectedly created an output directory")
    return {
        "schema": "wnn.oldz4-hft-profile-cli-preflight.r6",
        "status": "PASS_ACTUAL_C337_ARGV_BEFORE_DEVICE_AND_DATA",
        "trainer_path": str(trainer_path.resolve()),
        "trainer_source_sha256": contract["trainer_source_sha256"],
        "profile_contract_sha256": sha256_file(CONTRACT_PATH),
        "profile_runner_sha256": sha256_file(Path(__file__).resolve()),
        "fit_contract_sha256": contract["fit_contract_sha256"],
        "crop_receipt_sha256": inputs["crop_receipt_sha256"],
        "argv": contract["trainer_argv"],
        "next_phase": "configure_torch/device initialization; no output directory or data loader entered",
        "device_or_data_started": False,
    }


def _require_launch_receipt(contract: dict[str, Any], crop_receipt_sha: str) -> str:
    """Require root's release for this unique profile contract and crop receipt."""
    receipt = read_object(LAUNCH_RECEIPT_PATH)
    expected = {
        "schema": "wnn.oldz4-hft-profile-launch.r6",
        "decision": "PASS",
        "profile_run": contract["run"],
        "profile_contract_sha256": sha256_file(CONTRACT_PATH),
        "fit_contract_sha256": contract["fit_contract_sha256"],
        "core_crop_receipt_sha256": crop_receipt_sha,
        "trainer_source_sha256": contract["trainer_source_sha256"],
    }
    mismatches = {
        key: {"expected": value, "actual": receipt.get(key)}
        for key, value in expected.items()
        if receipt.get(key) != value
    }
    if mismatches:
        raise ValueError(f"root R6 profile launch receipt does not match: {mismatches}")
    return sha256_file(LAUNCH_RECEIPT_PATH)


def run_profile(project_root: Path) -> None:
    """Run only the root-released short profile; the exact fit contract is untouched."""
    r5_runner = _load_r5_runner()
    contract, _ = validate_contract(r5_runner)
    inputs = _validate_inputs(r5_runner, project_root)
    _require_launch_receipt(contract, inputs["crop_receipt_sha256"])
    run_dir = project_root / "artifacts/runs" / contract["run"]
    if run_dir.exists():
        raise FileExistsError(f"R6 profile output already exists: {run_dir}")
    trainer, trainer_path = _import_trainer(r5_runner, project_root.resolve())
    sys.argv = [str(trainer_path), *contract["trainer_argv"]]
    trainer.main()
    print(
        json.dumps(
            {
                "schema": "wnn.oldz4-hft-profile-execution.r6",
                "profile_run": contract["run"],
                "profile_contract_sha256": sha256_file(CONTRACT_PATH),
                "core_crop_receipt_sha256": inputs["crop_receipt_sha256"],
            },
            sort_keys=True,
        )
    )


def main() -> None:
    """Run the safe exact-CLI check or the root-released short profile."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--validate-cli", action="store_true")
    args = parser.parse_args()
    if args.validate_cli:
        print(json.dumps(validate_actual_cli(args.project_root), sort_keys=True))
    else:
        run_profile(args.project_root)


if __name__ == "__main__":
    main()
