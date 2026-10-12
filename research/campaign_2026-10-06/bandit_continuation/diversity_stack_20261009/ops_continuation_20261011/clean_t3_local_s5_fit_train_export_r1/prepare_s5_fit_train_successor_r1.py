"""Prepare the serial s5 FIT_TRAIN export while admitted s4 is healthy."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

WORKTREE_ROOT = Path(__file__).resolve().parents[6]
OUTPUT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
OPS = WORKTREE_ROOT / (
    "research/campaign_2026-10-06/bandit_continuation/"
    "diversity_stack_20261009/ops_continuation_20261011"
)
MAIN_OPS = PROJECT_ROOT / (
    "research/campaign_2026-10-06/bandit_continuation/"
    "diversity_stack_20261009/ops_continuation_20261011"
)
BASE_PROFILE = OPS / "clean_t3_local_s3_fit_train_profile_r1"
BASE_CONTRACT_PATH = BASE_PROFILE / "R1_FULL_EXPORT_CONTRACT_PENDING_ROOT_ADMISSION.json"
BASE_SUCCESSOR_PATH = BASE_PROFILE / "FULL_EXPORT_SUCCESSOR_R1.json"
S4_STARTUP_PATH = OPS / "clean_t3_local_s4_fit_train_export_r1/S4_STARTUP_OBSERVATION_R1.json"
S4_ATTEMPT_PATH = PROJECT_ROOT / (
    "research/campaign_2026-10-06/bandit_continuation/runtime/attempts/"
    "clean-t3-f0-s4-fit-train-export-r1-20261012.json"
)
SELECTOR_PREFLIGHT_PATH = (
    MAIN_OPS / "clean_t3_local_s3_fit_train_preflight_r1/LOCAL_S3_FIT_TRAIN_PREFLIGHT_R1.json"
)
SEED_READINESS_PATH = MAIN_OPS / "clean_t3_local_s3_terminal_r1/S4_S5_EXPORT_PREFLIGHT_R1.json"


def sha256_file(path: Path) -> str:
    """Hash a bound file without loading its payload into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON evidence record with a checked object shape."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Write one new JSON receipt atomically and refuse replacement."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as destination:
        json.dump(value, destination, indent=2, sort_keys=True, allow_nan=False)
        destination.write("\n")
        destination.flush()
        os.fsync(destination.fileno())
    os.replace(temporary, path)


def footer_identity(path: Path, footer_bytes: int) -> str:
    """Hash the bound Parquet footer, footer length, and trailing magic."""
    digest = hashlib.sha256()
    remaining = footer_bytes + 8
    with path.open("rb") as source:
        source.seek(-remaining, os.SEEK_END)
        while remaining:
            chunk = source.read(min(1024 * 1024, remaining))
            if not chunk:
                raise EOFError(f"truncated Parquet footer: {path}")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def main() -> None:
    """Bind the s5 endpoint to the frozen exporter and actual source receipts."""
    outputs = (
        OUTPUT_DIR / "S5_FIT_TRAIN_EXPORT_CONTRACT_R1.json",
        OUTPUT_DIR / "S5_FIT_TRAIN_EXPORT_SUCCESSOR_R1.json",
    )
    if any(path.exists() for path in outputs):
        raise FileExistsError(f"refusing to overwrite successor evidence in {OUTPUT_DIR}")

    s4_startup = read_json(S4_STARTUP_PATH)
    s4_attempt = read_json(S4_ATTEMPT_PATH)
    if s4_startup.get("status") != "PASS_ADMITTED_WORKER_RUNNING":
        raise ValueError("s4 does not have a committed healthy startup observation")
    if s4_attempt.get("status") != "running" or s4_attempt.get("failure") is not None:
        raise ValueError("s4 controller attempt is no longer healthy and running")
    service = subprocess.run(
        [
            "systemctl",
            "--user",
            "show",
            "wnn-bandit-clean-t3-f0-s4-fit-train-export-r1-20261012.service",
            "--property=ActiveState,SubState,ExecMainStatus,MemoryCurrent",
            "--no-pager",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    service_properties = dict(
        line.split("=", 1) for line in service.stdout.splitlines() if "=" in line
    )
    if service_properties.get("ActiveState") != "active":
        raise ValueError("s4 service is not active")
    if service_properties.get("SubState") != "running":
        raise ValueError("s4 service is not running")

    base_contract = read_json(BASE_CONTRACT_PATH)
    base_successor = read_json(BASE_SUCCESSOR_PATH)
    selector = read_json(SELECTOR_PREFLIGHT_PATH)
    readiness = read_json(SEED_READINESS_PATH)
    if readiness.get("status") != "PASS_PROVENANCE_AND_OUTPUT_COLLISION_PREFLIGHT_ONLY":
        raise ValueError("seed-specific s4/s5 readiness receipt is not passing")
    if selector.get("status") != (
        "PASS_ACTUAL_FIT_TRAIN_SELECTOR_BEFORE_CUDA; PREPARED_NOT_LAUNCHED"
    ):
        raise ValueError("frozen source-qualified FIT_TRAIN selector preflight is not passing")

    endpoint = next(row for row in readiness["endpoints"] if row["seed"] == 5)
    if endpoint["run"] != "clean_t3_f0_muonall_h384_e3_s5_r5":
        raise ValueError("seed-5 readiness row identifies an unexpected run")
    if endpoint["selected_epoch"] != 2 or not endpoint["membership"]["train_set_equals_split"]:
        raise ValueError("seed-5 selected CP2 is not bound to complete FIT_TRAIN")
    if not endpoint["membership"]["valid_fit_eval_disjoint"]:
        raise ValueError("seed-5 readiness does not prove FIT_VALID/evaluation disjointness")
    if (
        endpoint["membership"]["train_sorted_int64_le_sha256"]
        != base_contract["selector"]["full_membership_sha256"]
    ):
        raise ValueError("seed-5 FIT_TRAIN membership differs from the frozen selector")

    checkpoint_path = PROJECT_ROOT / endpoint["checkpoint_path"]
    provenance_path = PROJECT_ROOT / endpoint["provenance_path"]
    if checkpoint_path.stat().st_size != endpoint["checkpoint_bytes"]:
        raise ValueError("seed-5 CP2 byte count differs from the readiness receipt")
    if sha256_file(checkpoint_path) != endpoint["checkpoint_sha256"]:
        raise ValueError("seed-5 CP2 hash differs from the readiness receipt")
    if provenance_path.stat().st_size != endpoint["provenance_bytes"]:
        raise ValueError("seed-5 provenance byte count differs from the readiness receipt")
    if sha256_file(provenance_path) != endpoint["provenance_sha256"]:
        raise ValueError("seed-5 provenance hash differs from the readiness receipt")

    boundary = selector["actual_consumer_boundary"]
    raw_source = selector["raw_train_source_identity"]
    train_path = PROJECT_ROOT / "datasets/train.parquet"
    train_stat = train_path.stat()
    if train_stat.st_size != raw_source["verified_size_bytes"]:
        raise ValueError("train.parquet size differs from the immutable source binding")
    if train_stat.st_mtime_ns != raw_source["verified_mtime_ns"]:
        raise ValueError("train.parquet mtime differs from the immutable source binding")
    if (
        footer_identity(train_path, raw_source["verified_split_footer_bytes"])
        != raw_source["verified_footer_plus_length_plus_PAR1_sha256"]
    ):
        raise ValueError("train.parquet footer identity differs from the frozen source")
    if (
        boundary["actual_campaign_split_sha256"]
        != base_contract["bindings"]["campaign_split_sha256"]
    ):
        raise ValueError("FIT_TRAIN selector preflight split differs from the frozen contract")
    if boundary["actual_fit_train_count"] != 10_535:
        raise ValueError("FIT_TRAIN selector count differs from the complete F0 membership")
    if (
        boundary["selected_fit_train_groups_ordered_int64_le_sha256"]
        != endpoint["membership"]["train_sorted_int64_le_sha256"]
    ):
        raise ValueError("seed-5 membership differs from the actual frozen selector order")

    runner_path = BASE_PROFILE / "run_clean_t3_fit_train_export_r1.py"
    if sha256_file(runner_path) != base_successor["successor_runner_sha256"]:
        raise ValueError("frozen s3 export wrapper hash changed")
    contract = copy.deepcopy(base_contract)
    run_name = endpoint["run"]
    output_name = f"{run_name}_local_fit_train_r1"
    output_path = PROJECT_ROOT / "cache" / f"pred_{output_name}_train.npy"
    sidecar_path = output_path.with_suffix(".json")
    partial_path = output_path.with_name(output_path.name + ".partial")
    for path in (output_path, sidecar_path, partial_path):
        if path.exists():
            raise FileExistsError(f"seed-5 FIT_TRAIN output already exists: {path}")

    disk = shutil.disk_usage(output_path.parent)
    required_free = int(contract["output"]["minimum_free_bytes"])
    if disk.free < required_free:
        raise OSError(f"insufficient cache disk for s5 export: {disk.free} free bytes")

    contract["scope"] = (
        "Single frozen s5 CP2 FIT_TRAIN feature-prediction cache; reuses the completed "
        "s3 16-sequence resource geometry/source (B16, chunk 4000, FP32); no refit."
    )
    contract["status"] = "PENDING_ROOT_SERIAL_RESOURCE_RELEASE"
    contract["run"] = {"epoch": 2, "name": run_name, "seed": 5}
    contract["bindings"]["checkpoint_sha256"] = endpoint["checkpoint_sha256"]
    contract["bindings"]["provenance_sha256"] = endpoint["provenance_sha256"]
    contract["bindings"]["run_identity_source_sha256"] = endpoint["run_identity_source_sha256"]
    contract["paths"]["checkpoint"] = str(checkpoint_path)
    contract["output"]["output_name"] = output_name
    contract["output"]["path"] = str(output_path)
    contract["output"]["sidecar_path"] = str(sidecar_path)
    contract["output"]["partial_path"] = str(partial_path)
    contract["endpoint_binding"] = {
        "seed": 5,
        "run": run_name,
        "selected_epoch": 2,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": endpoint["checkpoint_sha256"],
        "provenance_path": str(provenance_path),
        "provenance_sha256": endpoint["provenance_sha256"],
        "run_identity_source_sha256": endpoint["run_identity_source_sha256"],
        "fit_train_count": endpoint["membership"]["train_count"],
        "fit_train_sorted_int64_le_sha256": endpoint["membership"]["train_sorted_int64_le_sha256"],
        "selector_ordered_int64_le_sha256": boundary[
            "selected_fit_train_groups_ordered_int64_le_sha256"
        ],
        "fit_valid_count": endpoint["membership"]["valid_count"],
        "fit_valid_sorted_int64_le_sha256": endpoint["membership"]["valid_sorted_int64_le_sha256"],
        "evaluation_count": endpoint["membership"]["evaluation_count"],
        "fit_valid_evaluation_disjoint": endpoint["membership"]["valid_fit_eval_disjoint"],
        "seed_specific_readiness_path": str(SEED_READINESS_PATH),
        "seed_specific_readiness_sha256": sha256_file(SEED_READINESS_PATH),
        "selector_preflight_path": str(SELECTOR_PREFLIGHT_PATH),
        "selector_preflight_sha256": sha256_file(SELECTOR_PREFLIGHT_PATH),
        "selector_reuse_scope": (
            "Actual frozen predictor boundary proved the split-derived FIT_TRAIN "
            "selector for s3. Membership is seed-independent and is cross-bound to "
            "the source-qualified s5 provenance readiness row; runtime repeats full "
            "membership validation before allocating CUDA."
        ),
        "source_manifest_sha256": contract["bindings"]["source_manifest_sha256"],
        "campaign_split_sha256": contract["bindings"]["campaign_split_sha256"],
        "train_content_sha256": contract["bindings"]["train_content_sha256"],
        "train_footer_sha256": raw_source["verified_footer_plus_length_plus_PAR1_sha256"],
    }
    contract_path = OUTPUT_DIR / "S5_FIT_TRAIN_EXPORT_CONTRACT_R1.json"
    write_json(contract_path, contract)
    contract_sha = sha256_file(contract_path)

    runtime = PROJECT_ROOT / "research/campaign_2026-10-06/bandit_continuation/runtime"
    python = Path("/home/vlad/development/wunder/wnn_connectome_starterpack/.venv/bin/python")
    observation_path = OUTPUT_DIR / "S5_FULL_EXPORT_OBSERVATION_R1.json"
    command = [
        "/usr/bin/timeout",
        "--signal=TERM",
        "--kill-after=2s",
        "3600s",
        "/usr/bin/env",
        f"PYTHONPATH={contract['paths']['source_manifest'].rsplit('/', 1)[0]}/src",
        f"WNN_PROJECT_ROOT={PROJECT_ROOT}",
        "WNN_LOADER_WORKERS=0",
        str(python),
        str(runner_path),
        "--contract",
        str(contract_path),
        "--preflight",
        str(SELECTOR_PREFLIGHT_PATH),
        "--predictor",
        contract["paths"]["predictor"],
        "--trainer",
        contract["paths"]["trainer"],
        "--source-manifest",
        contract["paths"]["source_manifest"],
        "--project-root",
        str(PROJECT_ROOT),
        "--receipt",
        str(observation_path),
    ]
    attempt = "clean-t3-f0-s5-fit-train-export-r1-20261012"
    profile = "clean_t3_f0_s5_fit_train_export_r1_20261012"
    controller_path = WORKTREE_ROOT / "scripts/bandit_resources.py"
    controller_argv = [
        str(python),
        str(controller_path),
        "--root",
        str(WORKTREE_ROOT),
        "--state",
        str(runtime),
        "launch",
        "--attempt",
        attempt,
        "--kind",
        "gpu",
        "--cpus",
        "1",
        "--reserve-bytes",
        "5500000000",
        "--profile",
        profile,
        "--contract",
        str(contract_path),
        "--python",
        str(python),
        "--admission-timeout",
        "20",
        "--",
        *command,
    ]
    manifest = {
        "schema": "wnn.clean-t3-local-fit-train-export-successor.r1",
        "status": "PENDING_ROOT_SERIAL_RESOURCE_RELEASE",
        "purpose": "Complete already-fitted own s5 CP2 FIT_TRAIN feature-prediction export after s4 terminal review; no refit or scoring.",
        "seed": 5,
        "run": run_name,
        "selected_epoch": 2,
        "attempt": attempt,
        "profile": profile,
        "contract_path": str(contract_path),
        "contract_sha256": contract_sha,
        "runner_path": str(runner_path),
        "runner_sha256": sha256_file(runner_path),
        "controller_path": str(controller_path),
        "controller_sha256": sha256_file(controller_path),
        "controller_argv": controller_argv,
        "source_and_membership": contract["endpoint_binding"],
        "output": {
            "prediction_path": str(output_path),
            "sidecar_path": str(sidecar_path),
            "partial_path": str(partial_path),
            "shape": contract["output"]["shape"],
            "dtype": contract["output"]["dtype"],
            "expected_bytes": contract["output"]["expected_size_bytes"],
            "minimum_free_bytes_at_prepare": required_free,
            "free_bytes_at_prepare": disk.free,
            "all_output_paths_absent_at_prepare": True,
        },
        "resources": {
            "cpus": 1,
            "process_ram_reserve_bytes": 5_500_000_000,
            "allocator_cap_bytes": 4_300_000_000,
            "observed_vram_cap_bytes": 5_000_000_000,
            "swap_bytes": 0,
            "loader_workers": 0,
            "timeout_seconds": 3600,
            "chunk_rows": 4000,
            "batch_sequences": 16,
            "precision": "fp32",
            "device": "NVIDIA GeForce RTX 3060 Laptop GPU",
            "gpu_lock": "shared bandit_resources state lock; controller-owned",
        },
        "actual_s3_geometry_source": {
            "source_successor_path": str(BASE_SUCCESSOR_PATH),
            "source_successor_sha256": sha256_file(BASE_SUCCESSOR_PATH),
            "profile_observation_path": base_successor["profile_receipt_path"],
            "profile_observation_sha256": base_successor["profile_receipt_sha256"],
            "s3_full_export_attempt": base_successor["successful_attempt"],
            "geometry_reused_without_profile_repeat": True,
        },
        "s4_serial_parent": {
            "startup_receipt_path": str(S4_STARTUP_PATH),
            "startup_receipt_sha256": sha256_file(S4_STARTUP_PATH),
            "attempt_receipt_path": str(S4_ATTEMPT_PATH),
            "status_when_prepared": s4_attempt["status"],
            "phase_when_prepared": s4_attempt["phase"],
            "systemd_service_properties": service_properties,
            "must_wait_for_s4_terminal_and_root_release": True,
        },
        "fresh_root_admission_required_after_s4_terminal": True,
    }
    manifest_path = OUTPUT_DIR / "S5_FIT_TRAIN_EXPORT_SUCCESSOR_R1.json"
    write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "contract_path": str(contract_path),
                "contract_sha256": contract_sha,
                "manifest_path": str(manifest_path),
                "manifest_sha256": sha256_file(manifest_path),
                "preparer_sha256": sha256_file(Path(__file__).resolve()),
                "attempt": attempt,
                "output_path": str(output_path),
                "checkpoint_sha256": endpoint["checkpoint_sha256"],
                "provenance_sha256": endpoint["provenance_sha256"],
                "fit_train_membership_sha256": endpoint["membership"][
                    "train_sorted_int64_le_sha256"
                ],
                "output_free_bytes": disk.free,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
