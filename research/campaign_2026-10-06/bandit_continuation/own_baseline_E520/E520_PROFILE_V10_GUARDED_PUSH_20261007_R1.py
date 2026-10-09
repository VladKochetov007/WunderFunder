"""Admit and perform one fresh-gated E520 V10 profile push."""

from __future__ import annotations

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
CANDIDATE = BASE / "live_cuda_nvml_binding_v10_candidate_v1"
PACKET = BASE / "packet_v10"
SLUG = "vladkochetov07/e520-profile-v10-live-uuid"
PACKET_ID = "E520_PROFILE_GPU_IDENTITY_LIVE_BINDING_V10"
ALLOCATION = BASE / "E520_PROFILE_V10_ROOT_SUCCESSOR_ALLOCATION_20261007_R1.json"
SOURCE_REVIEW = BASE / "E520_V10_INDEPENDENT_SOURCE_REVIEW_20261007_R1.json"
TARGET_MAP = CANDIDATE / "E520_V10_REVIEW_TARGETS.json"
CODE_RECEIPT = CANDIDATE / "E520_V10_CODE_RECEIPT.json"
CONTROLLER = BASE / "E520_PROFILE_V10_CONTROLLER_BUILDER_ADMISSION_20261007_R1.json"
TRANSPORT_REVIEW = BASE / "E520_V10_INDEPENDENT_TRANSPORT_REVIEW_20261007_R1.json"
ASSEMBLY = BASE / "E520_PROFILE_V10_PACKET_ASSEMBLY_20261007_R1.json"
MANIFEST = BASE / "E520_PROFILE_V10_PACKET_SHA256SUMS_20261007_R1.txt"
REGISTRATION = BASE / "E520_PROFILE_V10_PROVIDER_TARGET_REGISTRATION_20261007_R1.json"
TARGETS = ROOT / "research/campaign_2026-10-06/cloud_preflight/provider_targets.json"
QUOTA_TARGETS = BASE / "E520_V10_QUOTA_ONLY_TARGETS_20261007_R1.json"
POLL_SCRIPT = ROOT / "research/campaign_2026-10-06/cloud_preflight/poll_provider.py"
QUOTA_LOG = BASE / "E520_V10_PREPUSH_QUOTA_FINAL_20261007_R1.jsonl"
QUOTA_RECEIPT = BASE / "E520_V10_PREPUSH_QUOTA_FINAL_20261007_R1.json"
LOCAL_RECEIPT = BASE / "E520_V10_LOCAL_RESOURCE_PREPUSH_20261007_R1.json"
KERNEL_LIST = BASE / "E520_V10_KERNEL_LIST_IMMEDIATE_PREPUSH_20261007_R1.txt"
KERNEL_LIST_STDERR = BASE / "E520_V10_KERNEL_LIST_IMMEDIATE_PREPUSH_20261007_R1.stderr.txt"
PREPUSH = BASE / "E520_V10_PREPUSH_ADMISSION_20261007_R1.json"
ATTEMPT_MARKER = BASE / "E520_V10_PUSH_ATTEMPT_STARTED_20261007_R1.json"
PUSH_STDOUT = BASE / "E520_V10_PUSH_STDOUT_20261007_R1.txt"
PUSH_STDERR = BASE / "E520_V10_PUSH_STDERR_20261007_R1.txt"
PUSH_RECEIPT = BASE / "E520_V10_PUSH_RECEIPT_20261007_R1.json"
PROVIDER_TARGET = {
    "artifact_dir": "downloads/e520_profile_v10",
    "kernel": SLUG,
    "resource": "gpu",
    "terminal_receipt": "e520_profile_terminal_v10_poll_receipt.json",
}
EXPECTED = {
    "allocation": "0d4f48f970195e9579c98d83352def407bd0e22ab4ca2c5b6f8e4022df434d6a",
    "source_review": "84d03aa52afc86837210710841f2ee4592364003aa4695ab00565ae5891f8a62",
    "target_map": "924bcf726cd282d80a156d6994f7fbca03078090d186c127478b1fd74abf0581",
    "code_receipt": "6f4613b2f5c54ed36f0fd981d9e2a97cc2807fbee718ec9d396420fe1cf3d1a2",
    "controller": "d1ad048372a0655a507c0b4bfdcfb79bcbfe2ffc2c7c2ae06340a34fd34ca802",
    "transport": "4bec4c5e25e9a5822024d45adc0771840ba49fa40a5329cf32656e78f5ab07d6",
    "assembly": "628d53d14066d01d48d1702c282e3a28cdbce56e12d2581bde6a1b7c5e13d50e",
    "manifest": "328b42d72896925962dacb4cec0903f779322a2191f43148e40fcffa6e2628fa",
    "registration": "a625302315de8544c4c83e9cf9c627101fb5ee62046a69becd66f998fd0e0800",
    "provider_targets": "59134fd18b22e4fdcc2aa9956a2afc81b04ea364a8c670e4f4aca97b246696f4",
}
PROFILE_WALL_CAP_S = 1450
QUOTA_RESERVATION_S = 2900.0
E520_RECOVERY_CAP_S = 9000.0
ACCOUNT_RESERVE_FRACTION = 0.20
ROOT_PRIOR_RECOVERY_SPEND_S = 5858.447


def sha256_path(path: Path) -> str:
    """Hash a file using bounded reads."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {path}")
    return value


def require_sha(path: Path, expected: str, label: str) -> None:
    """Fail closed on an exact byte mismatch."""
    actual = sha256_path(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: {actual} != {expected}")


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Atomically persist a receipt."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def verify_frozen_evidence() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Verify the reviewed source, packet, transport and allocation bytes."""
    for key, label, path in (
        ("allocation", "allocation", ALLOCATION),
        ("source_review", "source review", SOURCE_REVIEW),
        ("target_map", "target map", TARGET_MAP),
        ("code_receipt", "code receipt", CODE_RECEIPT),
        ("controller", "controller admission", CONTROLLER),
        ("transport", "transport review", TRANSPORT_REVIEW),
        ("assembly", "packet assembly", ASSEMBLY),
        ("manifest", "packet manifest", MANIFEST),
        ("registration", "provider target registration", REGISTRATION),
        ("provider_targets", "provider targets", TARGETS),
    ):
        require_sha(path, EXPECTED[key], label)

    allocation = read_json(ALLOCATION)
    source = read_json(SOURCE_REVIEW)
    target_map = read_json(TARGET_MAP)
    controller = read_json(CONTROLLER)
    transport = read_json(TRANSPORT_REVIEW)
    assembly = read_json(ASSEMBLY)
    registration = read_json(REGISTRATION)
    if allocation.get("successor_scope", {}).get("candidate_id") != PACKET_ID:
        raise ValueError("root allocation candidate mismatch")
    scope = allocation["successor_scope"]
    if (
        scope.get("exact_profile_attempts_authorized") != 1
        or scope.get("max_worker_wall_seconds") != PROFILE_WALL_CAP_S
        or scope.get("max_provisional_quota_seconds") != QUOTA_RESERVATION_S
        or scope.get("cumulative_recovery_quota_cap_seconds") != E520_RECOVERY_CAP_S
        or scope.get("fit_export_grade_score") is not False
        or scope.get("paid_compute") is not False
    ):
        raise ValueError("root V10 allocation scope does not match guarded attempt")
    if (
        source.get("status") != "approved"
        or source.get("verdict") != "PASS_SOURCE_ONLY_PENDING_STATIC_TRANSPORT_REVIEW"
        or source.get("review_target_map", {}).get("sha256") != EXPECTED["target_map"]
        or source.get("code_freeze_receipt", {}).get("sha256") != EXPECTED["code_receipt"]
        or source.get("root_allocation", {}).get("sha256") != EXPECTED["allocation"]
        or source.get("reviewed_hashes") != target_map.get("reviewed_hashes")
    ):
        raise ValueError("source review does not approve the exact frozen V10 map")
    if target_map.get("packet_id") != PACKET_ID:
        raise ValueError("V10 target-map packet ID mismatch")
    if (
        controller.get("status") != "approved"
        or controller.get("experiment_id") != PACKET_ID
        or controller.get("scope") != "profile-only"
        or controller.get("reviewed_hashes") != target_map.get("reviewed_hashes")
        or controller.get("builder_admission", {}).get("provider_push_now_authorized") is not False
    ):
        raise ValueError("builder-only controller admission mismatch")
    if (
        transport.get("status") != "approved"
        or transport.get("verdict") != "PASS_STATIC_TRANSPORT_ONLY"
        or transport.get("experiment_id") != PACKET_ID
        or transport.get("assembly", {}).get("sha256") != EXPECTED["assembly"]
        or transport.get("source_review", {}).get("sha256") != EXPECTED["source_review"]
        or transport.get("builder_admission", {}).get("sha256") != EXPECTED["controller"]
    ):
        raise ValueError("independent static transport review binding mismatch")
    if (
        assembly.get("packet_path") != str(PACKET.relative_to(ROOT))
        or assembly.get("provider_called") is not False
        or assembly.get("fit_or_score_run") is not False
        or assembly.get("packet_manifest_sha256") != EXPECTED["manifest"]
    ):
        raise ValueError("packet assembly scope or path mismatch")
    if (
        registration.get("target") != PROVIDER_TARGET
        or registration.get("provider_targets_sha256_after") != EXPECTED["provider_targets"]
        or registration.get("provider_mutation_performed") is not False
    ):
        raise ValueError("durable target registration receipt mismatch")

    # Recheck each source member, packet member, and manifest immediately before admission.
    for name, expected in target_map["reviewed_hashes"].items():
        require_sha(CANDIDATE / name, expected, f"source member {name}")
    manifest_check = subprocess.run(
        ["sha256sum", "--check", "--status", str(MANIFEST)],
        cwd=PACKET,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    if manifest_check.returncode != 0:
        raise ValueError("packet SHA-256 manifest check failed")
    packet_members = assembly.get("packet_members", {})
    if len(packet_members) != 19:
        raise ValueError("unexpected V10 packet member count")
    for name, facts in packet_members.items():
        path = PACKET / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"packet member missing or not a regular file: {name}")
        if path.stat().st_size != facts["bytes"] or sha256_path(path) != facts["sha256"]:
            raise ValueError(f"packet assembly hash/size mismatch: {name}")
    packet_meta = read_json(PACKET / "packet_meta.json")
    kernel_meta = read_json(PACKET / "kernel-metadata.json")
    if (
        packet_meta.get("packet_id") != PACKET_ID
        or packet_meta.get("profile_only") is not True
        or packet_meta.get("fit_enabled") is not False
        or packet_meta.get("prediction_export_enabled") is not False
        or packet_meta.get("profile_worker_wall_cap_seconds") != PROFILE_WALL_CAP_S
        or packet_meta.get("external_review_sha256") != EXPECTED["controller"]
        or kernel_meta.get("id") != SLUG
        or kernel_meta.get("is_private") is not True
        or kernel_meta.get("kernel_type") != "script"
        or kernel_meta.get("enable_gpu") is not True
        or kernel_meta.get("code_file") != "profile_kernel_bundle.py"
    ):
        raise ValueError("V10 packet metadata/scope mismatch")
    return allocation, source, transport


def capture_local_resource() -> dict[str, Any]:
    """Read the live campaign parent boundary and active model services."""
    env = dict(os.environ)
    env["XDG_RUNTIME_DIR"] = "/run/user/1000"
    env["DBUS_SESSION_BUS_ADDRESS"] = "unix:path=/run/user/1000/bus"
    command = [
        "systemctl",
        "--user",
        "show",
        "wnn-campaign.slice",
        "-p",
        "ControlGroup",
        "-p",
        "CPUQuotaPerSecUSec",
        "-p",
        "MemoryHigh",
        "-p",
        "MemoryMax",
        "-p",
        "MemorySwapMax",
    ]
    shown = subprocess.run(
        command, capture_output=True, text=True, env=env, timeout=10, check=False
    )
    if shown.returncode != 0:
        raise RuntimeError(f"systemd resource query failed: {shown.stderr}")
    properties = dict(line.split("=", 1) for line in shown.stdout.splitlines() if "=" in line)
    control_group = properties.get("ControlGroup")
    if not control_group:
        raise ValueError("campaign cgroup path missing")
    cgroup = Path("/sys/fs/cgroup") / control_group.lstrip("/")
    raw = {
        key: (cgroup / filename).read_text().strip()
        for key, filename in {
            "cpu_max": "cpu.max",
            "memory_current_bytes": "memory.current",
            "memory_high": "memory.high",
            "memory_max": "memory.max",
            "memory_swap_max": "memory.swap.max",
        }.items()
    }
    active_cmd = [
        "systemctl",
        "--user",
        "list-units",
        "--type=service",
        "--state=running",
        "--no-legend",
        "--plain",
    ]
    active = subprocess.run(
        active_cmd, capture_output=True, text=True, env=env, timeout=10, check=False
    )
    if active.returncode != 0:
        raise RuntimeError(f"active-service query failed: {active.stderr}")
    model_units = [
        line.split()[0]
        for line in active.stdout.splitlines()
        if line.split() and line.split()[0].startswith("wnn-bandit-")
    ]
    fit_units = [
        unit
        for unit in model_units
        if any(part in unit.lower() for part in ("-fit-", "-train-", "-predict-"))
    ]
    current = int(raw["memory_current_bytes"])
    if (
        properties.get("CPUQuotaPerSecUSec") != "6s"
        or properties.get("MemoryHigh") != "18000000000"
        or properties.get("MemoryMax") != "20000000000"
        or properties.get("MemorySwapMax") != "0"
        or raw["cpu_max"] != "600000 100000"
        or abs(int(raw["memory_high"]) - 18_000_000_000) > 4096
        or abs(int(raw["memory_max"]) - 20_000_000_000) > 4096
        or int(raw["memory_swap_max"]) != 0
        or current >= int(raw["memory_high"])
        or fit_units
    ):
        raise ValueError(
            f"local boundary/fit gate failed: props={properties}, raw={raw}, fit_units={fit_units}"
        )
    return {
        "observed_at_utc": datetime.now(UTC).isoformat(timespec="milliseconds"),
        "source": "live systemd user slice properties, cgroup v2 files, and running unit list",
        "parent_slice": {
            "name": "wnn-campaign.slice",
            "control_group": control_group,
            "cpu_quota_per_sec_usec": properties["CPUQuotaPerSecUSec"],
            "cpu_max": raw["cpu_max"],
            "memory_high_bytes": int(raw["memory_high"]),
            "memory_max_bytes": int(raw["memory_max"]),
            "memory_swap_max_bytes": int(raw["memory_swap_max"]),
            "memory_current_bytes": current,
        },
        "active_bandit_services": model_units,
        "active_local_fit_services": fit_units,
        "gpu_fit_started": False,
        "support_process_cpu_quota": "100%",
        "support_process_memory_max_bytes": 1_500_000_000,
        "support_process_runtime_max_seconds": 120,
    }


def collect_quota() -> dict[str, Any]:
    """Use the existing read-only provider poller for fresh quota/status."""
    sys.path.insert(0, str(POLL_SCRIPT.parent))
    import poll_provider  # noqa: PLC0415

    snapshot = poll_provider.collect(QUOTA_TARGETS)
    poll_provider.append_jsonl(QUOTA_LOG, snapshot)
    gpu = snapshot["quota"]["gpu"]
    total = float(gpu["total_allowed_seconds"])
    used = float(gpu["used_seconds"])
    reserved = float(gpu["reserved_seconds"])
    reserve = total * ACCOUNT_RESERVE_FRACTION
    headroom_after_reserve = total - used - reserved - reserve
    campaign_headroom = float(gpu["campaign_headroom_seconds"])
    campaign_after_attempt = campaign_headroom - QUOTA_RESERVATION_S
    account_after_attempt = headroom_after_reserve - QUOTA_RESERVATION_S
    allocation = read_json(ALLOCATION)["successor_scope"]
    recovery_before = float(allocation["max_provisional_quota_seconds"])
    recovery_remaining_before = E520_RECOVERY_CAP_S - ROOT_PRIOR_RECOVERY_SPEND_S
    recovery_after = recovery_remaining_before - recovery_before
    if (
        total <= 0
        or account_after_attempt < 0
        or campaign_after_attempt < 0
        or recovery_after < 0
        or used < 0
        or reserved < 0
    ):
        raise ValueError(
            f"live quota/reserve gate failed: account_after={account_after_attempt}, "
            f"campaign_after={campaign_after_attempt}, recovery_after={recovery_after}"
        )
    return {
        "observed_at_utc": snapshot["observed_at_utc"],
        "source": snapshot["source"],
        "quota_refresh_utc": snapshot.get("quota_refresh_utc"),
        "gpu": gpu,
        "tpu": snapshot["quota"]["tpu"],
        "account_20_percent_reserve_seconds": reserve,
        "account_headroom_after_reserve_and_attempt_seconds": account_after_attempt,
        "campaign_headroom_after_attempt_seconds": campaign_after_attempt,
        "attempt_quota_reservation_seconds": QUOTA_RESERVATION_S,
        "e520_recovery_cap_seconds": E520_RECOVERY_CAP_S,
        "e520_prior_actual_recovery_spend_seconds": ROOT_PRIOR_RECOVERY_SPEND_S,
        "e520_remaining_before_attempt_seconds": recovery_remaining_before,
        "e520_remaining_after_attempt_reservation_seconds": recovery_after,
        "paid_compute": False,
        "provider_status": snapshot.get("status", []),
    }


def fresh_slug_absence() -> None:
    """Prove exact slug absence immediately before the one push."""
    command = [
        "kaggle",
        "kernels",
        "list",
        "--mine",
        "--search",
        "e520-profile-v10-live-uuid",
        "--format",
        "json",
    ]
    result = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, timeout=20, check=False
    )
    KERNEL_LIST.write_text(result.stdout, encoding="utf-8")
    KERNEL_LIST_STDERR.write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0 or result.stdout.strip() != "Not found" or result.stderr:
        raise RuntimeError(
            f"exact slug absence failed: rc={result.returncode}; stdout={result.stdout!r}; stderr={result.stderr!r}"
        )


def create_prepush_admission(
    allocation: dict[str, Any],
    source: dict[str, Any],
    transport: dict[str, Any],
    local: dict[str, Any],
    quota: dict[str, Any],
) -> dict[str, Any]:
    """Persist the live, hash-bound pre-push admission."""
    current_targets = read_json(TARGETS).get("targets", [])
    if [row for row in current_targets if row.get("kernel") == SLUG] != [PROVIDER_TARGET]:
        raise ValueError("durable V10 polling target missing or ambiguous")
    registration = read_json(REGISTRATION)
    if sha256_path(TARGETS) != registration.get("provider_targets_sha256_after"):
        raise ValueError("provider target file changed after V10 target registration")
    if ATTEMPT_MARKER.exists() or PUSH_RECEIPT.exists():
        raise ValueError("V10 attempt marker/receipt exists; inspect provider before any action")

    write_json(QUOTA_RECEIPT, quota)
    write_json(LOCAL_RECEIPT, local)
    quota_hash = sha256_path(QUOTA_RECEIPT)
    local_hash = sha256_path(LOCAL_RECEIPT)
    script_hash = sha256_path(Path(__file__))
    receipt = {
        "schema_version": 1,
        "receipt_id": "E520_V10_PREPUSH_ADMISSION_20261007_R1",
        "issued_at_utc": datetime.now(UTC).isoformat(),
        "status": "approved",
        "scope": "profile-only",
        "experiment_id": PACKET_ID,
        "kernel_id": SLUG,
        "authorization_basis": "Root V10 successor allocation, exact independent source PASS and static transport PASS, live provider quota/reserve, local parent resource gates, durable target registration and exact slug absence all pass.",
        "provider_push_now_authorized": True,
        "fit_or_score_admitted": False,
        "reviewer": {"id": "e520_cloud_delivery", "role": "independent-controller"},
        "root_allocation_path": str(ALLOCATION.relative_to(ROOT)),
        "root_allocation_sha256": EXPECTED["allocation"],
        "source_review_path": str(SOURCE_REVIEW.relative_to(ROOT)),
        "source_review_sha256": EXPECTED["source_review"],
        "source_review_map_sha256": EXPECTED["target_map"],
        "transport_review_path": str(TRANSPORT_REVIEW.relative_to(ROOT)),
        "transport_review_sha256": EXPECTED["transport"],
        "controller_admission_path": str(CONTROLLER.relative_to(ROOT)),
        "controller_admission_sha256": EXPECTED["controller"],
        "packet_assembly_path": str(ASSEMBLY.relative_to(ROOT)),
        "packet_assembly_sha256": EXPECTED["assembly"],
        "packet_manifest_path": str(MANIFEST.relative_to(ROOT)),
        "packet_manifest_sha256": EXPECTED["manifest"],
        "provider_target_registration_path": str(REGISTRATION.relative_to(ROOT)),
        "provider_target_registration_sha256": EXPECTED["registration"],
        "provider_targets_path": str(TARGETS.relative_to(ROOT)),
        "provider_targets_sha256": EXPECTED["provider_targets"],
        "guarded_push_script_path": str(Path(__file__).relative_to(ROOT)),
        "guarded_push_script_sha256": script_hash,
        "quota_admission": {
            "path": str(QUOTA_RECEIPT.relative_to(ROOT)),
            "sha256": quota_hash,
            "observed_at_utc": quota["observed_at_utc"],
            "gpu_account_used_seconds": quota["gpu"]["used_seconds"],
            "gpu_account_reserved_seconds": quota["gpu"]["reserved_seconds"],
            "gpu_account_total_seconds": quota["gpu"]["total_allowed_seconds"],
            "account_20_percent_reserve_seconds": quota["account_20_percent_reserve_seconds"],
            "account_headroom_after_reserve_and_attempt_seconds": quota[
                "account_headroom_after_reserve_and_attempt_seconds"
            ],
            "campaign_cap_seconds": quota["gpu"]["campaign_cap_seconds"],
            "campaign_headroom_seconds": quota["gpu"]["campaign_headroom_seconds"],
            "campaign_headroom_after_attempt_seconds": quota[
                "campaign_headroom_after_attempt_seconds"
            ],
            "attempt_quota_reservation_seconds": QUOTA_RESERVATION_S,
            "e520_remaining_before_attempt_seconds": quota["e520_remaining_before_attempt_seconds"],
            "e520_remaining_after_attempt_reservation_seconds": quota[
                "e520_remaining_after_attempt_reservation_seconds"
            ],
            "paid_compute": False,
            "source": quota["source"],
        },
        "local_resource": {
            "path": str(LOCAL_RECEIPT.relative_to(ROOT)),
            "sha256": local_hash,
            "observed_at_utc": local["observed_at_utc"],
            "parent_slice": local["parent_slice"],
            "active_bandit_services": local["active_bandit_services"],
            "active_local_fit_services": local["active_local_fit_services"],
            "gpu_fit_started": False,
            "support_cpu_quota": "100%",
            "support_memory_max_bytes": 1_500_000_000,
            "support_runtime_max_seconds": 120,
        },
        "provider_preflight": {
            "exact_slug_present_before_push": False,
            "kernel_id": SLUG,
            "kernel_list_command": [
                "kaggle",
                "kernels",
                "list",
                "--mine",
                "--search",
                "e520-profile-v10-live-uuid",
                "--format",
                "json",
            ],
            "kernel_list_result": "Not found",
            "kernel_list_stdout_path": str(KERNEL_LIST.relative_to(ROOT)),
            "kernel_list_stdout_sha256": sha256_path(KERNEL_LIST),
            "kernel_list_stderr_path": str(KERNEL_LIST_STDERR.relative_to(ROOT)),
            "kernel_list_stderr_sha256": sha256_path(KERNEL_LIST_STDERR),
            "target_registered": True,
        },
        "execution_scope": {
            "profile_only": True,
            "worker_wall_cap_seconds": PROFILE_WALL_CAP_S,
            "max_quota_reservation_seconds": QUOTA_RESERVATION_S,
            "automatic_retry": False,
            "fit": False,
            "prediction_export": False,
            "grading": False,
            "scoring": False,
            "paid_compute": False,
        },
        "push_action": "Run this exact guarded script once; it has performed the pure hash/scope checks, refreshed quota/resources and checked exact slug absence immediately before one kaggle kernels push.",
    }
    write_json(PREPUSH, receipt)
    return receipt


def validate_fresh_admission(receipt: dict[str, Any]) -> str:
    """Fail closed if any receipt, freshness or budget predicate changed."""
    admission_sha = sha256_path(PREPUSH)
    if (
        receipt.get("status") != "approved"
        or receipt.get("provider_push_now_authorized") is not True
        or receipt.get("scope") != "profile-only"
        or receipt.get("fit_or_score_admitted") is not False
        or receipt.get("kernel_id") != SLUG
        or receipt.get("experiment_id") != PACKET_ID
        or receipt.get("guarded_push_script_sha256") != sha256_path(Path(__file__))
    ):
        raise ValueError("pre-push admission/script scope check failed")
    for key, path in (
        ("root_allocation_sha256", ALLOCATION),
        ("source_review_sha256", SOURCE_REVIEW),
        ("transport_review_sha256", TRANSPORT_REVIEW),
        ("controller_admission_sha256", CONTROLLER),
        ("packet_assembly_sha256", ASSEMBLY),
        ("packet_manifest_sha256", MANIFEST),
        ("provider_target_registration_sha256", REGISTRATION),
        ("provider_targets_sha256", TARGETS),
    ):
        if receipt.get(key) != sha256_path(path):
            raise ValueError(f"admission binding mismatch: {key}")
    for section in ("quota_admission", "local_resource"):
        evidence = receipt[section]
        path = ROOT / evidence["path"]
        require_sha(path, evidence["sha256"], section)
        live_evidence = read_json(path)
        observed = datetime.fromisoformat(evidence["observed_at_utc"].replace("Z", "+00:00"))
        age = (datetime.now(UTC) - observed).total_seconds()
        if not 0 <= age <= 300:
            raise ValueError(f"{section} is stale or future dated: {age:.3f}s")
        if section == "quota_admission" and (
            live_evidence.get("observed_at_utc") != evidence.get("observed_at_utc")
            or float(live_evidence.get("gpu", {}).get("used_seconds", -1))
            != float(evidence.get("gpu_account_used_seconds", -2))
        ):
            raise ValueError("quota admission does not match its live provider receipt")
    if (
        receipt["quota_admission"]["account_headroom_after_reserve_and_attempt_seconds"] < 0
        or receipt["quota_admission"]["campaign_headroom_after_attempt_seconds"] < 0
        or receipt["quota_admission"]["e520_remaining_after_attempt_reservation_seconds"] < 0
        or receipt["local_resource"]["gpu_fit_started"] is not False
    ):
        raise ValueError("live quota/resource headroom check failed")
    if sha256_path(TARGETS) != EXPECTED["provider_targets"]:
        raise ValueError("durable polling target registry changed after admission")
    target_rows = read_json(TARGETS).get("targets", [])
    if [row for row in target_rows if row.get("kernel") == SLUG] != [PROVIDER_TARGET]:
        raise ValueError("registered target row changed or is ambiguous")
    if KERNEL_LIST.read_text(encoding="utf-8").strip() != "Not found":
        raise ValueError("exact slug absence receipt is invalid")
    if KERNEL_LIST_STDERR.read_text(encoding="utf-8") != "":
        raise ValueError("exact slug query wrote stderr")
    if ATTEMPT_MARKER.exists() or PUSH_RECEIPT.exists():
        raise ValueError("push marker/receipt already exists; inspect provider, no retry")
    return admission_sha


def main() -> int:
    """Verify all gates, persist admission, then invoke exactly one provider push."""
    if any(
        path.exists() for path in (PREPUSH, ATTEMPT_MARKER, PUSH_RECEIPT, PUSH_STDOUT, PUSH_STDERR)
    ):
        raise FileExistsError(
            "V10 pre-push or attempt artifact already exists; inspect before action"
        )
    allocation, source, transport = verify_frozen_evidence()
    local = capture_local_resource()
    quota = collect_quota()
    fresh_slug_absence()
    admission = create_prepush_admission(allocation, source, transport, local, quota)
    admission_sha = validate_fresh_admission(admission)

    started = datetime.now(UTC).isoformat()
    marker = {
        "receipt_id": "E520_V10_PUSH_ATTEMPT_STARTED_20261007_R1",
        "status": "provider_push_invoked_next",
        "attempt_id": PACKET_ID,
        "kernel_id": SLUG,
        "invoked_at_utc": started,
        "prepush_admission_sha256": admission_sha,
        "guarded_push_script_sha256": sha256_path(Path(__file__)),
        "automatic_retry": False,
    }
    with ATTEMPT_MARKER.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(marker, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())

    command = ["kaggle", "kernels", "push", "-p", str(PACKET), "-t", str(PROFILE_WALL_CAP_S)]
    timed_out = False
    try:
        result = subprocess.run(
            command, cwd=ROOT, capture_output=True, text=True, timeout=55, check=False
        )
        exit_code = result.returncode
        stdout = result.stdout
        stderr = result.stderr
    except subprocess.TimeoutExpired as error:
        timed_out = True
        exit_code = None
        stdout = (
            error.stdout.decode(errors="replace")
            if isinstance(error.stdout, bytes)
            else (error.stdout or "")
        )
        stderr = (
            error.stderr.decode(errors="replace")
            if isinstance(error.stderr, bytes)
            else (error.stderr or "")
        )
    PUSH_STDOUT.write_text(stdout, encoding="utf-8")
    PUSH_STDERR.write_text(stderr, encoding="utf-8")
    finished = datetime.now(UTC).isoformat()
    status = (
        "ambiguous_timeout_inspect_provider_before_any_retry"
        if timed_out
        else "cli_exit_zero_provider_status_verification_required"
        if exit_code == 0
        else "cli_error_inspect_provider_before_any_retry"
    )
    receipt = {
        "schema_version": 1,
        "receipt_id": "E520_V10_PUSH_RECEIPT_20261007_R1",
        "attempt_id": PACKET_ID,
        "kernel_id": SLUG,
        "push_command": command,
        "push_started_at_utc": started,
        "push_finished_at_utc": finished,
        "status": status,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "prepush_admission_path": str(PREPUSH.relative_to(ROOT)),
        "prepush_admission_sha256": admission_sha,
        "guarded_push_script_path": str(Path(__file__).relative_to(ROOT)),
        "guarded_push_script_sha256": sha256_path(Path(__file__)),
        "immediate_kernel_list_stdout_path": str(KERNEL_LIST.relative_to(ROOT)),
        "immediate_kernel_list_stdout_sha256": sha256_path(KERNEL_LIST),
        "immediate_kernel_list_stderr_path": str(KERNEL_LIST_STDERR.relative_to(ROOT)),
        "immediate_kernel_list_stderr_sha256": sha256_path(KERNEL_LIST_STDERR),
        "stdout_path": str(PUSH_STDOUT.relative_to(ROOT)),
        "stdout_sha256": sha256_path(PUSH_STDOUT),
        "stderr_path": str(PUSH_STDERR.relative_to(ROOT)),
        "stderr_sha256": sha256_path(PUSH_STDERR),
        "attempt_marker_path": str(ATTEMPT_MARKER.relative_to(ROOT)),
        "automatic_retry": False,
        "fit_or_score_run": False,
    }
    write_json(PUSH_RECEIPT, receipt)
    marker.update(
        {
            "status": status,
            "finished_at_utc": finished,
            "push_receipt_sha256": sha256_path(PUSH_RECEIPT),
        }
    )
    write_json(ATTEMPT_MARKER, marker)
    print(
        json.dumps({"prepush_admission_sha256": admission_sha, **receipt}, indent=2, sort_keys=True)
    )
    return 0 if exit_code == 0 and not timed_out else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(
            f"E520 V10 guarded push refused before or during provider call: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        raise SystemExit(1) from error
