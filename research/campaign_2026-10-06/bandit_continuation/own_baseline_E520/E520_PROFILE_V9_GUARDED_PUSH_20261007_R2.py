"""Validate and make one guarded E520 V9 Kaggle push."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]
PACKET = BASE / "packet_v9"
SLUG = "vladkochetov07/e520-profile-v9-memory-domains"
PACKET_ID = "E520_PROFILE_GPU_MEMORY_DOMAIN_V9"
PREPUSH = BASE / "E520_PROFILE_V9_PREPUSH_ADMISSION_20261007_R2.json"
CONTROLLER = BASE / "E520_PROFILE_V9_CONTROLLER_BUILDER_ADMISSION_20261007_R1.json"
ASSEMBLY = BASE / "E520_PROFILE_V9_PACKET_ASSEMBLY_20261007_R1.json"
MANIFEST = BASE / "E520_PROFILE_V9_PACKET_SHA256SUMS_20261007_R1.txt"
SOURCE_REVIEW = BASE / "E520_PROFILE_V9_INDEPENDENT_SOURCE_REVIEW_20261007_R1.json"
TRANSPORT_REVIEW = BASE / "E520_PROFILE_V9_PACKET_TRANSPORT_INDEPENDENT_REVIEW_20261007_R1.json"
REGISTRATION = BASE / "E520_PROFILE_V9_PROVIDER_TARGET_REGISTRATION_20261007_R1.json"
ALLOCATION = BASE / "E520_V9_ROOT_CONDITIONAL_REPAIR_ALLOCATION_20261007_V1.json"
TARGETS = ROOT / "research/campaign_2026-10-06/cloud_preflight/provider_targets.json"
ATTEMPT_MARKER = BASE / "E520_PROFILE_V9_PUSH_ATTEMPT_STARTED_20261007_R2.json"
PUSH_STDOUT = BASE / "E520_PROFILE_V9_PUSH_STDOUT_20261007_R2.txt"
PUSH_STDERR = BASE / "E520_PROFILE_V9_PUSH_STDERR_20261007_R2.txt"
PUSH_RECEIPT = BASE / "E520_PROFILE_V9_PUSH_RECEIPT_20261007_R2.json"
KERNEL_LIST = BASE / "E520_V9_KERNEL_LIST_IMMEDIATE_PREPUSH_20261007_R2.txt"
KERNEL_LIST_STDERR = BASE / "E520_V9_KERNEL_LIST_IMMEDIATE_PREPUSH_20261007_R2.stderr.txt"

EXPECTED = {
    "controller": "14f1c9d3500f5ebaeb4281f34451dc5f4b313e02125dd24b78a7893f479de436",
    "assembly": "0817674b77d35b2c31dbd708ad86ea8afa047880a90dddd386da2e2035f2de07",
    "manifest": "a019fb208c95b9a986c00077c7201d2b70c126e7630b5a02b395a762897359ad",
    "source_review": "b8c2d3f9dbb481a6009c4f8bd6e4ae2d5b7e40ea2327d1e0f44baa22e5d2f1f7",
    "transport_review": "212606d0ccc20a002d291dd9e9c7baa0026e91721b3222c31066b1655102ed40",
    "registration": "e9ab52462f8e1520592454679e890940d2ec841debf61f0c9da3bae703a6112d",
    "allocation": "67302c079b6d04e343eb0676717bbd11d23ccd191abe87c12f21a93c21a83784",
    "provider_targets": "30503ecd19f1ce095210237f6755f6b7ffcba5b6131d662e343aee9d51eecdac",
}


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {path}")
    return value


def require_sha(path: Path, expected: str, label: str) -> None:
    actual = sha256_path(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: {actual} != {expected}")


def require_fresh(path: Path, observed_at: str, label: str) -> float:
    observed = datetime.fromisoformat(observed_at.replace("Z", "+00:00"))
    age = (datetime.now(UTC) - observed).total_seconds()
    if not 0 <= age <= 300:
        raise ValueError(f"{label} is stale or future dated: age={age:.3f}s")
    return age


def validate_preconditions(admission_sha256: str) -> dict[str, Any]:
    """Pure local validation; this function makes no provider calls."""
    require_sha(PREPUSH, admission_sha256, "pre-push admission")
    admission = read_json(PREPUSH)
    if admission.get("status") != "approved":
        raise ValueError("pre-push admission is not approved")
    if admission.get("provider_push_now_authorized") is not True:
        raise ValueError("provider push is not authorized by the receipt")
    if (
        admission.get("scope") != "profile-only"
        or admission.get("fit_or_score_admitted") is not False
    ):
        raise ValueError("pre-push admission scope is not profile-only")
    if admission.get("kernel_id") != SLUG or admission.get("experiment_id") != PACKET_ID:
        raise ValueError("pre-push admission identifies a different attempt")
    if admission.get("guarded_push_script_sha256") != sha256_path(Path(__file__)):
        raise ValueError("guarded push script hash mismatch")

    for key, path in (
        ("controller_admission_sha256", CONTROLLER),
        ("packet_assembly_sha256", ASSEMBLY),
        ("packet_manifest_sha256", MANIFEST),
        ("source_review_sha256", SOURCE_REVIEW),
        ("transport_review_sha256", TRANSPORT_REVIEW),
        ("provider_target_registration_sha256", REGISTRATION),
        ("root_allocation_sha256", ALLOCATION),
    ):
        if admission.get(key) != sha256_path(path):
            raise ValueError(f"pre-push admission binding mismatch for {key}")
    for key, path in (
        ("controller", CONTROLLER),
        ("assembly", ASSEMBLY),
        ("manifest", MANIFEST),
        ("source_review", SOURCE_REVIEW),
        ("transport_review", TRANSPORT_REVIEW),
        ("registration", REGISTRATION),
        ("allocation", ALLOCATION),
        ("provider_targets", TARGETS),
    ):
        require_sha(path, EXPECTED[key], key)

    source = read_json(SOURCE_REVIEW)
    if source.get("result") != "PASS_SOURCE_ONLY":
        raise ValueError("independent source review is not PASS_SOURCE_ONLY")
    source_map_hash = source.get("target", {}).get("review_target_sha256")
    if source_map_hash != "60bac7cba692a3462cc0ffaee03bac7ca107762050fbfce396d9a6c29b4a88af":
        raise ValueError("source review is bound to a different map")

    transport = read_json(TRANSPORT_REVIEW)
    assembly_hash = sha256_path(ASSEMBLY)
    if transport.get("decision") != "PASS":
        raise ValueError("independent packet transport review is not PASS")
    if transport.get("freeze", {}).get("assembly_receipt_sha256") != assembly_hash:
        raise ValueError("transport review is bound to a different packet assembly")

    controller = read_json(CONTROLLER)
    if controller.get("experiment_id") != PACKET_ID or controller.get("scope") != "profile-only":
        raise ValueError("builder admission packet/scope mismatch")
    if (
        controller.get("status") != "approved"
        or controller.get("reviewer", {}).get("role") != "independent-controller"
    ):
        raise ValueError("builder admission schema rejected")

    allocation = read_json(ALLOCATION)
    if (
        allocation.get("new_recovery_budget_quota_seconds") != 9000
        or allocation.get("new_profile_max_worker_wall_seconds") != 1700
        or allocation.get("new_profile_max_provisional_quota_seconds") != 3400
        or allocation.get("paid_compute_authorized") is not False
    ):
        raise ValueError("root allocation limits do not match this attempt")

    packet_meta = read_json(PACKET / "packet_meta.json")
    kernel_meta = read_json(PACKET / "kernel-metadata.json")
    if (
        packet_meta.get("packet_id") != PACKET_ID
        or packet_meta.get("profile_only") is not True
        or packet_meta.get("fit_enabled") is not False
        or packet_meta.get("prediction_export_enabled") is not False
        or packet_meta.get("profile_worker_wall_cap_seconds") != 1700
        or packet_meta.get("external_review_sha256") != sha256_path(CONTROLLER)
    ):
        raise ValueError("packet metadata scope or review binding mismatch")
    if (
        kernel_meta.get("id") != SLUG
        or kernel_meta.get("is_private") is not True
        or kernel_meta.get("kernel_type") != "script"
        or kernel_meta.get("enable_gpu") is not True
        or kernel_meta.get("code_file") != "profile_kernel_bundle.py"
    ):
        raise ValueError("private kernel metadata mismatch")

    check = subprocess.run(
        ["sha256sum", "--check", "--status", str(MANIFEST)],
        cwd=PACKET,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    if check.returncode != 0:
        raise ValueError("packet member SHA-256 manifest failed")

    quota_admission = admission.get("quota_admission", {})
    quota_path = ROOT / quota_admission["path"]
    require_sha(quota_path, quota_admission["sha256"], "quota poll")
    quota_poll = read_json(quota_path)
    quota_age = require_fresh(quota_path, quota_poll["observed_at_utc"], "quota poll")
    gpu = quota_poll["quota"]["gpu"]
    if float(gpu["used_seconds"]) != float(quota_admission["gpu_account_used_seconds"]):
        raise ValueError("admission GPU usage differs from latest quota poll")
    if (
        float(quota_admission.get("e520_remaining_after_attempt_reservation_seconds", -1)) < 0
        or float(
            quota_admission.get("campaign_headroom_after_attempt_and_conditional_fit_seconds", -1)
        )
        <= 0
        or float(
            quota_admission.get(
                "account_headroom_after_reserve_attempt_and_conditional_fit_seconds", -1
            )
        )
        <= 0
        or quota_admission.get("paid_compute") is not False
    ):
        raise ValueError("quota reserve, campaign headroom, or paid-compute guard failed")

    local = admission.get("local_resource", {})
    local_path = ROOT / local["path"]
    require_sha(local_path, local["sha256"], "local resource snapshot")
    local_snapshot = read_json(local_path)
    local_age = require_fresh(
        local_path, local_snapshot["observed_at_utc"], "local resource snapshot"
    )
    parent = local_snapshot["parent_slice"]
    if (
        parent.get("cpu_quota_per_sec_usec") != "6s"
        or parent.get("memory_high_bytes") != 18_000_000_000
        or parent.get("memory_max_bytes") != 20_000_000_000
        or parent.get("memory_swap_max_bytes") != 0
        or parent.get("memory_current_bytes", 20_000_000_000) >= 18_000_000_000
        or local_snapshot.get("gpu_fit_started") is not False
    ):
        raise ValueError("local resource boundary or no-local-fit guard failed")

    preflight = admission.get("provider_preflight", {})
    list_path = ROOT / preflight["kernel_list_stdout_path"]
    list_err_path = ROOT / preflight["kernel_list_stderr_path"]
    require_sha(list_path, preflight["kernel_list_stdout_sha256"], "pre-push slug query stdout")
    require_sha(list_err_path, preflight["kernel_list_stderr_sha256"], "pre-push slug query stderr")
    if list_path.read_text().strip() != "Not found" or list_err_path.read_text() != "":
        raise ValueError("pre-push exact slug query did not prove absence")
    targets = read_json(TARGETS).get("targets", [])
    expected_target = {
        "artifact_dir": "downloads/e520_profile_v9",
        "kernel": SLUG,
        "resource": "gpu",
        "terminal_receipt": "e520_profile_terminal_v9_poll_receipt.json",
    }
    if [row for row in targets if row.get("kernel") == SLUG] != [expected_target]:
        raise ValueError("durable polling target registration is missing or ambiguous")
    registration = read_json(REGISTRATION)
    if registration.get("provider_targets_sha256_after") != sha256_path(TARGETS):
        raise ValueError("provider target registry changed after registration")

    if ATTEMPT_MARKER.exists() or PUSH_RECEIPT.exists():
        raise ValueError(
            "a V9 provider push attempt marker already exists; inspect before any retry"
        )

    return {
        "experiment_id": PACKET_ID,
        "kernel_id": SLUG,
        "packet_path": str(PACKET),
        "prepush_admission_sha256": admission_sha256,
        "guarded_push_script_sha256": sha256_path(Path(__file__)),
        "packet_manifest_sha256": sha256_path(MANIFEST),
        "packet_assembly_sha256": assembly_hash,
        "quota_age_seconds": round(quota_age, 3),
        "local_resource_age_seconds": round(local_age, 3),
        "profile_wall_cap_seconds": 1700,
        "profile_quota_reservation_seconds": 3400,
        "provider_push_authorized": True,
        "fit_or_score_admitted": False,
    }


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run_one_push(validation: dict[str, Any]) -> int:
    list_command = [
        "kaggle",
        "kernels",
        "list",
        "--mine",
        "--search",
        "e520-profile-v9-memory-domains",
        "--format",
        "json",
    ]
    listing = subprocess.run(
        list_command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    KERNEL_LIST.write_text(listing.stdout, encoding="utf-8")
    KERNEL_LIST_STDERR.write_text(listing.stderr, encoding="utf-8")
    if listing.returncode != 0 or listing.stdout.strip() != "Not found" or listing.stderr:
        raise RuntimeError("immediate exact-slug query failed or slug is present; refusing push")

    admission = read_json(PREPUSH)
    target_rows = read_json(TARGETS).get("targets", [])
    if sha256_path(TARGETS) != admission.get("provider_targets_sha256"):
        raise RuntimeError("provider target registry changed after admission; refusing push")
    if len([row for row in target_rows if row.get("kernel") == SLUG]) != 1:
        raise RuntimeError("durable provider target changed; refusing push")
    if ATTEMPT_MARKER.exists():
        raise RuntimeError("push attempt marker already exists; refusing any repeated push")

    started = datetime.now(UTC).isoformat()
    marker = {
        "attempt_id": PACKET_ID,
        "status": "provider_push_invoked_next",
        "invoked_at_utc": started,
        "kernel_id": SLUG,
        "prepush_admission_sha256": validation["prepush_admission_sha256"],
        "guarded_push_script_sha256": validation["guarded_push_script_sha256"],
        "immediate_kernel_list_stdout_sha256": sha256_path(KERNEL_LIST),
        "automatic_retry": False,
    }
    with ATTEMPT_MARKER.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(marker, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())

    command = ["kaggle", "kernels", "push", "-p", str(PACKET), "-t", "1700"]
    timed_out = False
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=100,
        )
        exit_code = result.returncode
        stdout = result.stdout
        stderr = result.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        exit_code = None
        stdout = (
            exc.stdout.decode(errors="replace")
            if isinstance(exc.stdout, bytes)
            else (exc.stdout or "")
        )
        stderr = (
            exc.stderr.decode(errors="replace")
            if isinstance(exc.stderr, bytes)
            else (exc.stderr or "")
        )

    PUSH_STDOUT.write_text(stdout, encoding="utf-8")
    PUSH_STDERR.write_text(stderr, encoding="utf-8")
    finished = datetime.now(UTC).isoformat()
    status = (
        "ambiguous_timeout_inspect_provider_before_retry"
        if timed_out
        else (
            "cli_exit_zero_provider_status_verification_required"
            if exit_code == 0
            else "cli_error_inspect_provider_before_retry"
        )
    )
    receipt = {
        "receipt_id": "E520_PROFILE_V9_PUSH_RECEIPT_20261007_R2",
        "schema_version": 1,
        "attempt_id": PACKET_ID,
        "kernel_id": SLUG,
        "push_command": command,
        "push_started_at_utc": started,
        "push_finished_at_utc": finished,
        "status": status,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "prepush_admission_path": str(PREPUSH),
        "prepush_admission_sha256": validation["prepush_admission_sha256"],
        "guarded_push_script_path": str(Path(__file__)),
        "guarded_push_script_sha256": validation["guarded_push_script_sha256"],
        "immediate_kernel_list_command": list_command,
        "immediate_kernel_list_stdout_path": str(KERNEL_LIST),
        "immediate_kernel_list_stdout_sha256": sha256_path(KERNEL_LIST),
        "immediate_kernel_list_stderr_path": str(KERNEL_LIST_STDERR),
        "immediate_kernel_list_stderr_sha256": sha256_path(KERNEL_LIST_STDERR),
        "stdout_path": str(PUSH_STDOUT),
        "stdout_sha256": sha256_path(PUSH_STDOUT),
        "stderr_path": str(PUSH_STDERR),
        "stderr_sha256": sha256_path(PUSH_STDERR),
        "attempt_marker_path": str(ATTEMPT_MARKER),
        "automatic_retry": False,
        "fit_or_score_run": False,
    }
    atomic_write_json(PUSH_RECEIPT, receipt)
    atomic_write_json(
        ATTEMPT_MARKER,
        {
            **marker,
            "status": status,
            "finished_at_utc": finished,
            "push_receipt_sha256": sha256_path(PUSH_RECEIPT),
        },
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if exit_code == 0 and not timed_out else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check-only", action="store_true")
    action.add_argument("--push", action="store_true")
    parser.add_argument("--admission-sha256", required=True)
    args = parser.parse_args()
    validation = validate_preconditions(args.admission_sha256)
    if args.check_only:
        print(
            json.dumps(
                {"status": "preconditions_pass_no_provider_call", **validation},
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    return run_one_push(validation)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(
            f"E520 V9 guarded push refused before or during provider call: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
