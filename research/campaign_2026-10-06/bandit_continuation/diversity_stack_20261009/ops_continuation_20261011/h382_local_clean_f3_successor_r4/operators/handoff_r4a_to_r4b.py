"""Durably hand off the H382 fold-3 e3 sequence to its frozen e4 successor.

The operator waits for the already-running R4A sequence to finish, verifies its
exact fit/export records, performs the reviewed export-cache and fit-reserve
reclaims while the campaign GPU lock is free, then starts only the frozen R4B
sequence service. It never launches a training command directly.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
PACKET = Path(__file__).resolve().parent.parent
RUNTIME = ROOT / "research/campaign_2026-10-06/bandit_continuation/runtime"
PLAN_A = PACKET / "SEQUENCE_PLAN_R4A.json"
PLAN_B = PACKET / "SEQUENCE_PLAN_R4B.json"
PLAN_A_SHA256 = "acaae0562e261542ac36e58e514882385e082495f48ce2c0a513f31ce596f035"
PLAN_B_SHA256 = "1a8d8784b6989b006a2db8c942a36956aa23ceead2aea4f5228db86397830a06"
UNIT_A = "wnn-h382-local-f3-s3-e3-sequence-r4a.service"
UNIT_B = "wnn-h382-local-f3-s3-e4-sequence-r4b.service"
# The R4A transient unit used --collect and was garbage-collected after its
# successful sequence. Root observed this exact InvocationID while it ran.
UNIT_A_INVOCATION_ID = "1f4ba4e38bb946bc8cd065d083229d4b"
SEQUENCE_A_RECEIPT = RUNTIME / "sequence_h382_local_f3_e3_20261011_r4a.json"
SEQUENCE_B_RECEIPT = RUNTIME / "sequence_h382_local_f3_e4_20261011_r4b.json"
HANDOFF_RECEIPT = PACKET / "handoff_receipts/r4a_to_r4b_handoff_operator_r1.json"
EXPORT_RECLAIM_RECEIPT = PACKET / "handoff_receipts/r4a_e3_export_cache_reclaim_r1.json"
RESERVE_QUEUE = PACKET / "handoff_receipts/reserve_reclaim_queue"
RESERVE_RECLAIM_LOG = RESERVE_QUEUE / "runner_log.jsonl"
R4A_FIT = "h382-local-clean-f3-s3-e3-fit-r4"
R4A_EXPORT = "h382-local-clean-f3-s3-e3-export-r4"
R4B_FIT = "h382-local-clean-f3-s3-e4-fit-r4"
R4B_EXPORT = "h382-local-clean-f3-s3-e4-export-r4"
RESERVE_TARGET_BYTES = 11_000_000_000
POST_RECLAIM_LIMIT_BYTES = 6_500_000_000
POLL_SECONDS = 15
WAIT_SECONDS = 4 * 60 * 60

sys.path.insert(0, str(ROOT / "scripts"))
import bandit_resources  # noqa: E402
import local_queue_runner  # noqa: E402


class HandoffError(RuntimeError):
    """Raised when a handoff gate fails and no R4B launch is permitted."""


def sha256_file(path: Path) -> str:
    """Hash a file incrementally so checkpoint or prediction size is immaterial."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_receipt(path: Path, value: dict[str, Any]) -> None:
    """Atomically persist operator phase state without exposing command output."""
    bandit_resources.atomic_json(path, value)


def systemd_properties(unit: str) -> dict[str, str]:
    """Read exact properties for one named user service."""
    result = subprocess.run(
        [
            "systemctl",
            "--user",
            "show",
            unit,
            "--property=LoadState,ActiveState,SubState,Result,ExecMainStatus,MainPID,FragmentPath,ExecStart,InvocationID",
            "--no-pager",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode:
        return {}
    return dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)


def validate_packet() -> tuple[dict[str, Any], dict[str, Any]]:
    """Check the two frozen plans and their exact continuation IDs."""
    a_bytes = PLAN_A.read_bytes()
    b_bytes = PLAN_B.read_bytes()
    if hashlib.sha256(a_bytes).hexdigest() != PLAN_A_SHA256:
        raise HandoffError("R4A sequence plan hash differs from its sealed value")
    if hashlib.sha256(b_bytes).hexdigest() != PLAN_B_SHA256:
        raise HandoffError("R4B sequence plan hash differs from its sealed value")
    plan_a = json.loads(a_bytes)
    plan_b = json.loads(b_bytes)
    if [step["attempt"] for step in plan_a["steps"]] != [R4A_FIT, R4A_EXPORT]:
        raise HandoffError("R4A attempt order differs from the reviewed pair")
    if [step["attempt"] for step in plan_b["steps"]] != [R4B_FIT, R4B_EXPORT]:
        raise HandoffError("R4B attempt order differs from the reviewed pair")
    if len(plan_a["steps"]) != 2 or len(plan_b["steps"]) != 2:
        raise HandoffError("expected exactly two steps in each frozen sequence")
    if plan_a["runtime"] != str(RUNTIME) or plan_b["runtime"] != str(RUNTIME):
        raise HandoffError("sequence runtime path differs from the admitted runtime")
    if SEQUENCE_B_RECEIPT.exists():
        raise HandoffError("R4B sequence receipt already exists; refusing a duplicate")
    for attempt_id in (R4B_FIT, R4B_EXPORT):
        if bandit_resources.attempt_path(RUNTIME, attempt_id).exists():
            raise HandoffError(f"R4B attempt already has a durable record: {attempt_id}")
    for step in plan_b["steps"]:
        for output in step.get("must_not_exist", []):
            if Path(output).exists():
                raise HandoffError(f"R4B output collision: {output}")
    return plan_a, plan_b


def write_state(state: dict[str, Any], phase: str, **details: Any) -> None:
    """Append the current phase to a single atomic progress receipt."""
    state.update(details)
    state["phase"] = phase
    state["updated_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
    atomic_receipt(HANDOFF_RECEIPT, state)


def verify_a_terminal(plan_a: dict[str, Any]) -> dict[str, Any]:
    """Require successful service, sequence, fit and fixed-epoch export records."""
    service = systemd_properties(UNIT_A)
    if service.get("ActiveState") in {"active", "activating", "deactivating"}:
        return {"wait": True, "service": service}
    if service.get("LoadState") == "loaded":
        if service.get("InvocationID") != UNIT_A_INVOCATION_ID:
            raise HandoffError("loaded R4A service has a different invocation identity")
        if service.get("Result") != "success" or service.get("ExecMainStatus") != "0":
            raise HandoffError("R4A sequence service is terminal without rc=0")
    elif service.get("LoadState") != "not-found":
        raise HandoffError("R4A service state is neither loaded nor expected garbage-collected")
    if not SEQUENCE_A_RECEIPT.is_file():
        raise HandoffError("R4A sequence service ended without its sequence receipt")
    sequence = json.loads(SEQUENCE_A_RECEIPT.read_text())
    expected_hash = hashlib.sha256(PLAN_A.read_bytes()).hexdigest()
    if sequence.get("plan_sha256") != expected_hash or sequence.get("plan") != str(PLAN_A):
        raise HandoffError("R4A sequence receipt does not bind the frozen plan")
    if sequence.get("status") != "finished" or sequence.get("attempt") != R4A_EXPORT:
        raise HandoffError("R4A sequence receipt does not show both steps terminal-success")
    attempts: dict[str, dict[str, Any]] = {}
    for step in plan_a["steps"]:
        attempt_id = step["attempt"]
        path = bandit_resources.attempt_path(RUNTIME, attempt_id)
        if not path.is_file():
            raise HandoffError(f"R4A attempt receipt is missing: {attempt_id}")
        record = json.loads(path.read_text())
        if (
            record.get("status") != "finished"
            or record.get("exit_status") != 0
            or record.get("command") != step["expected_command"]
        ):
            raise HandoffError(f"R4A attempt is not the exact successful command: {attempt_id}")
        attempts[attempt_id] = {
            "receipt": str(path),
            "receipt_sha256": sha256_file(path),
            "status": record["status"],
            "exit_status": record["exit_status"],
        }
    output = ROOT / "cache/pred_h382_muonall256prop0_f3_s3_e3_valid.npy"
    sidecar = output.with_suffix(".json")
    if not output.is_file() or not sidecar.is_file():
        raise HandoffError("R4A fixed-epoch prediction or sidecar is missing")
    metadata = json.loads(sidecar.read_text())
    if metadata.get("run") != "h382_muonall256prop0_f3_s3_e3" or metadata.get("epoch") != 2:
        raise HandoffError("R4A prediction sidecar does not identify the e3 fixed CP2 export")
    if metadata.get("prediction_sha256") != sha256_file(output):
        raise HandoffError("R4A prediction bytes differ from their sidecar")
    return {
        "wait": False,
        "service": {
            "load_state": service.get("LoadState", "not-found"),
            "active_state": service.get("ActiveState"),
            "result": service.get("Result"),
            "exec_main_status": service.get("ExecMainStatus"),
            "invocation_id": service.get("InvocationID") or UNIT_A_INVOCATION_ID,
            "garbage_collected_after_success": service.get("LoadState") == "not-found",
        },
        "sequence_receipt_sha256": sha256_file(SEQUENCE_A_RECEIPT),
        "attempts": attempts,
        "prediction": {
            "path": str(output),
            "bytes": output.stat().st_size,
            "sha256": metadata["prediction_sha256"],
            "sidecar_sha256": sha256_file(sidecar),
            "epoch": metadata["epoch"],
        },
    }


def verify_parent_limits() -> tuple[Path, dict[str, int]]:
    """Read and verify the unchanged campaign cgroup before and after reclaim."""
    props = bandit_resources.parent_properties()
    group = bandit_resources.verify_parent(props)
    cgroup = bandit_resources.cgroup_dir(group)
    stat = dict(line.split() for line in (cgroup / "memory.stat").read_text().splitlines())
    snapshot = {
        "memory_current": int((cgroup / "memory.current").read_text()),
        "memory_file": int(stat["file"]),
        "memory_anon": int(stat["anon"]),
    }
    return cgroup, snapshot


def active_attempts() -> list[str]:
    """List admitted or unresolved ledger attempts that block quiescent reclaim."""
    return [
        str(item.get("id", "unknown"))
        for item in bandit_resources.job_records(RUNTIME)
        if item.get("status") in bandit_resources.ACTIVE | bandit_resources.UNRESOLVED
    ]


def run_export_reclaim() -> dict[str, Any]:
    """Call the existing predecessor- and GPU-lock-guarded export reclaim."""
    predecessor = bandit_resources.attempt_path(RUNTIME, R4A_EXPORT)
    command = [
        str(ROOT / "scripts/limited"),
        str(ROOT / ".venv/bin/python"),
        str(ROOT / "scripts/reclaim_bandit_export_cache.py"),
        "--runtime",
        str(RUNTIME),
        "--predecessor",
        str(predecessor),
        "--receipt",
        str(EXPORT_RECLAIM_RECEIPT),
    ]
    env = os.environ | {
        "WNN_SLICE": "wnn-campaign.slice",
        "WNN_CPU_QUOTA": "25%",
        "WNN_THREADS": "1",
        "WNN_MEM_MAX": "700000000",
        "WNN_RUNTIME_MAX": "120s",
    }
    result = subprocess.run(command, capture_output=True, text=True, timeout=150, env=env)
    return {
        "exit_code": result.returncode,
        "predecessor": str(predecessor),
        "predecessor_sha256": sha256_file(predecessor),
        "receipt": str(EXPORT_RECLAIM_RECEIPT) if EXPORT_RECLAIM_RECEIPT.exists() else None,
        "receipt_sha256": sha256_file(EXPORT_RECLAIM_RECEIPT)
        if EXPORT_RECLAIM_RECEIPT.exists()
        else None,
        "stderr_tail": result.stderr[-1000:],
    }


def acquire_gpu_lock(timeout_seconds: int):
    """Acquire the campaign GPU lock without blocking another healthy worker."""
    lock_path = RUNTIME / "gpu.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+")
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return handle
        except BlockingIOError:
            if time.monotonic() >= deadline:
                handle.close()
                raise HandoffError("campaign GPU lock remained occupied after R4A export") from None
            time.sleep(POLL_SECONDS)


def start_r4b_service() -> dict[str, Any]:
    """Start the installed frozen R4B service once; never directly launch a fit."""
    unit_file = PACKET / "units" / UNIT_B
    service = systemd_properties(UNIT_B)
    if service.get("LoadState") != "loaded":
        raise HandoffError("frozen R4B systemd service has not been installed/reloaded")
    fragment = Path(service.get("FragmentPath", ""))
    if not fragment.is_file() or sha256_file(fragment) != sha256_file(unit_file):
        raise HandoffError("installed R4B systemd unit differs from the reviewed template")
    if service.get("ActiveState") not in {"inactive", "failed"}:
        raise HandoffError("R4B systemd unit is not inactive; refusing a duplicate start")
    if service.get("ActiveState") == "failed":
        raise HandoffError("R4B systemd unit has a prior failed activation; preserve for review")
    result = subprocess.run(
        ["systemctl", "--user", "start", UNIT_B],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    after = systemd_properties(UNIT_B)
    if result.returncode != 0 or after.get("ActiveState") not in {"active", "activating"}:
        raise HandoffError("systemd did not confirm R4B sequence service active")
    return {
        "unit": UNIT_B,
        "unit_template_sha256": sha256_file(unit_file),
        "installed_fragment": str(fragment),
        "start_command_exit_code": result.returncode,
        "service_state": {
            key: after.get(key) for key in ("LoadState", "ActiveState", "SubState", "MainPID")
        },
    }


def execute() -> int:
    """Wait for R4A, reclaim only after its export, then start frozen R4B."""
    if HANDOFF_RECEIPT.exists():
        raise HandoffError("handoff receipt already exists; refusing to replay this operator")
    plan_a, plan_b = validate_packet()
    bandit_resources.verify_controller_membership()
    if (RUNTIME / "STOP_ADMISSION.json").exists():
        raise HandoffError("campaign admission is closed; preserve healthy work and stop here")
    state: dict[str, Any] = {
        "schema": "wnn.h382-local-f3-r4a-r4b-handoff.v1",
        "packet_directory": str(PACKET),
        "plan_r4a_sha256": hashlib.sha256(PLAN_A.read_bytes()).hexdigest(),
        "plan_r4b_sha256": hashlib.sha256(PLAN_B.read_bytes()).hexdigest(),
        "r4a_unit": UNIT_A,
        "r4b_unit": UNIT_B,
        "sequence_a_receipt": str(SEQUENCE_A_RECEIPT),
        "sequence_b_receipt": str(SEQUENCE_B_RECEIPT),
        "reserve_target_bytes": RESERVE_TARGET_BYTES,
        "post_reclaim_memory_limit_bytes": POST_RECLAIM_LIMIT_BYTES,
    }
    write_state(state, "waiting_for_exact_r4a_terminal")
    deadline = time.monotonic() + WAIT_SECONDS
    while True:
        if time.monotonic() >= deadline:
            raise HandoffError("R4A did not reach terminal success within the bounded wait")
        sequence = json.loads(SEQUENCE_A_RECEIPT.read_text()) if SEQUENCE_A_RECEIPT.exists() else {}
        if sequence.get("status") in {"needs_review", "drained"}:
            raise HandoffError(f"R4A sequence ended without success: {sequence.get('status')}")
        if sequence.get("status") == "running":
            heartbeat_text = sequence.get("heartbeat_utc")
            if heartbeat_text:
                heartbeat = datetime.fromisoformat(heartbeat_text)
                if (datetime.now(UTC) - heartbeat).total_seconds() > 180:
                    raise HandoffError("R4A sequence heartbeat is stale; no successor admission")
        verified = verify_a_terminal(plan_a)
        if not verified.get("wait"):
            break
        write_state(
            state,
            "waiting_for_exact_r4a_terminal",
            r4a_service_state=verified["service"].get("ActiveState"),
            r4a_sequence_status=sequence.get("status"),
            r4a_sequence_heartbeat_utc=sequence.get("heartbeat_utc"),
        )
        time.sleep(POLL_SECONDS)
    write_state(state, "r4a_terminal_verified", r4a_terminal=verified)
    if active_attempts():
        raise HandoffError(
            f"other active/unresolved campaign attempts block handoff: {active_attempts()}"
        )
    if EXPORT_RECLAIM_RECEIPT.exists() or RESERVE_RECLAIM_LOG.exists():
        raise HandoffError("a prior cache-handoff receipt exists; refusing to replay cleanup")
    reclaim = run_export_reclaim()
    write_state(state, "export_cache_reclaim_complete", export_cache_reclaim=reclaim)
    if reclaim["exit_code"] != 0 or reclaim["receipt"] is None:
        raise HandoffError("existing export reclaim did not complete successfully")
    reclaim_record = json.loads(EXPORT_RECLAIM_RECEIPT.read_text())
    if reclaim_record.get("status") != "ready":
        raise HandoffError("existing export reclaim did not report ready")

    RESERVE_QUEUE.mkdir(parents=True, exist_ok=True)
    cgroup, _ = verify_parent_limits()
    with acquire_gpu_lock(WAIT_SECONDS):
        blockers = active_attempts()
        if blockers:
            raise HandoffError(f"campaign attempt became active before reserve reclaim: {blockers}")
        _, reserve_before = verify_parent_limits()
        local_queue_runner.reclaim_file_cache_for(RESERVE_TARGET_BYTES, RESERVE_QUEUE)
        _, after = verify_parent_limits()
        log_hash = sha256_file(RESERVE_RECLAIM_LOG) if RESERVE_RECLAIM_LOG.exists() else None
        reserve_result = {
            "primitive": "scripts.local_queue_runner.reclaim_file_cache_for",
            "requested_reserve_bytes": RESERVE_TARGET_BYTES,
            "target_parent_memory_current_bytes": POST_RECLAIM_LIMIT_BYTES,
            "before": reserve_before,
            "after": after,
            "queue_log": str(RESERVE_RECLAIM_LOG),
            "queue_log_sha256": log_hash,
            "cgroup": str(cgroup),
            "gpu_lock_held": True,
            "active_campaign_attempts": blockers,
            "limits_changed": False,
        }
        write_state(state, "reserve_reclaim_complete", reserve_aware_reclaim=reserve_result)
        if after["memory_current"] > POST_RECLAIM_LIMIT_BYTES:
            raise HandoffError(
                f"reserve-aware reclaim left {after['memory_current']} bytes above the 6.5GB gate"
            )
        if active_attempts():
            raise HandoffError("campaign attempt became active during reclaim; no R4B start")
        if (RUNTIME / "STOP_ADMISSION.json").exists():
            raise HandoffError("campaign admission closed before R4B start")
        r4b_start = start_r4b_service()
        write_state(state, "r4b_service_started", r4b_service=r4b_start)
    write_state(
        state,
        "handoff_complete",
        r4b_sequence_owner="systemd user unit; sequence receipt bound above",
    )
    return 0


def main() -> int:
    """Run or statically validate the unique H382 R4 handoff packet."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-package-only", action="store_true")
    args = parser.parse_args()
    if args.validate_package_only:
        validate_packet()
        print(
            json.dumps(
                {
                    "status": "static_package_valid",
                    "r4a_plan_sha256": hashlib.sha256(PLAN_A.read_bytes()).hexdigest(),
                    "r4b_plan_sha256": hashlib.sha256(PLAN_B.read_bytes()).hexdigest(),
                    "packet_directory": str(PACKET),
                },
                sort_keys=True,
            )
        )
        return 0
    try:
        return execute()
    except (
        HandoffError,
        OSError,
        ValueError,
        KeyError,
        RuntimeError,
        subprocess.SubprocessError,
    ) as error:
        if HANDOFF_RECEIPT.exists():
            state = json.loads(HANDOFF_RECEIPT.read_text())
            write_state(state, "needs_review", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
