#!/usr/bin/env python3
"""Audit R3 and optionally hand off to the original conditional seed4 fit.

This wrapper cannot launch R3. Seed4 requires a separate explicit
`--launch-seed4` invocation after the R3 terminal audit has passed.
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import importlib.util
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

BASE = Path(__file__).resolve().parent
PACKET = BASE / "RECOVERY_GATE_PACKET_R3.json"
REQUEST = BASE / "RESUME_REQUEST_R3.json"
AUTHORITY = BASE / "ROOT_RESUME_AUTHORITY_R3.json"
AUDIT = BASE / "RESUME_TERMINAL_AUDIT_R3.json"
INTENT = BASE / "SEED4_AFTER_R3_RECEIPT_R1.json"
INTENT_R2 = BASE / "SEED4_AFTER_R3_RECEIPT_R2.json"
HANDOFF_R2 = BASE / "SEED4_AFTER_R3_CACHE_HANDOFF_R2.json"
STOP_BASENAME = "STOP_ADMISSION.json"
CAMPAIGN_SLICE = "wnn-campaign.slice"
PARENT_HIGH_BYTES = 18_000_000_000
PARENT_MAX_BYTES = 20_000_000_000
SEED4_RESERVE_BYTES = 9_000_000_000
HANDOFF_TARGET_BYTES = 8_500_000_000
MAX_RECLAIM_BYTES = 6_000_000_000
PRESERVED_FILE_CACHE_BYTES = 6_000_000_000


def sha256_file(path: Path) -> str:
    """Hash a small receipt or source file in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_successful_predecessor(
    audit: dict[str, Any] | None,
    attempt: dict[str, Any] | None,
    unit: dict[str, str] | None,
    *,
    attempt_id: str,
    authority_sha256: str,
    request_sha256: str,
    attempt_sha256: str,
    unit_name: str,
    recovered_state_sha256: str,
) -> None:
    """Require the exact R3 fit, authority, output state, and successful unit."""
    if not isinstance(audit, dict) or audit.get("status") != "recovery_fit_and_output_passed":
        raise ValueError("seed4 requires a successful R3 terminal audit")
    if audit.get("schema") != "wnn.fold1_gru384_resume_terminal_audit.r2":
        raise ValueError("R3 audit receipt schema differs from the frozen recovery gate")
    if not isinstance(attempt, dict) or not isinstance(unit, dict):
        raise ValueError("seed4 predecessor attempt and systemd unit receipts are required")
    expected_audit = {
        "attempt_id": attempt_id,
        "authority_sha256": authority_sha256,
        "request_sha256": request_sha256,
        "attempt_sha256": attempt_sha256,
        "recovered_training_state_sha256": recovered_state_sha256,
    }
    if any(audit.get(key) != value for key, value in expected_audit.items()):
        raise ValueError("R3 audit does not bind the exact predecessor attempt and state")
    expected_attempt = {
        "id": attempt_id,
        "kind": "gpu",
        "status": "finished",
        "exit_status": 0,
        "unit": unit_name,
        "contract_sha256": authority_sha256,
    }
    if any(attempt.get(key) != value for key, value in expected_attempt.items()):
        raise ValueError("R3 attempt is not the exact successful admitted predecessor")
    expected_unit = {
        "ActiveState": "inactive",
        "Result": "success",
        "ExecMainStatus": "0",
    }
    if any(unit.get(key) != value for key, value in expected_unit.items()):
        raise ValueError("R3 systemd unit is not a successful terminal unit")


def bounded_reclaim_bytes(current_bytes: int, file_bytes: int) -> int:
    """Bound cache reclaim to the 8.5 GB target and a 6 GB file-cache floor."""
    if type(current_bytes) is not int or type(file_bytes) is not int:
        raise ValueError("memory snapshot values must be exact integers")
    if current_bytes < 0 or file_bytes < 0:
        raise ValueError("memory snapshot values cannot be negative")
    return min(
        MAX_RECLAIM_BYTES,
        max(0, current_bytes - HANDOFF_TARGET_BYTES),
        max(0, file_bytes - PRESERVED_FILE_CACHE_BYTES),
    )


def memory_snapshot(cgroup: Path) -> dict[str, int]:
    """Read current, file-cache, and anonymous memory from one cgroup."""
    stat = dict(line.split() for line in (cgroup / "memory.stat").read_text().splitlines())
    snapshot = {
        "current": int((cgroup / "memory.current").read_text()),
        "file": int(stat["file"]),
        "anon": int(stat["anon"]),
    }
    if any(value < 0 for value in snapshot.values()):
        raise ValueError("campaign cgroup returned a negative memory counter")
    return snapshot


def campaign_parent_cgroup(
    attempt: dict[str, Any], cgroup_root: Path, cpu_cap: int
) -> tuple[Path, dict[str, int]]:
    """Bind the reclaim target to the recorded campaign parent without changing limits."""
    properties = attempt.get("parent_properties_at_start")
    group = properties.get("ControlGroup") if isinstance(properties, dict) else None
    if (
        attempt.get("parent_slice") != CAMPAIGN_SLICE
        or not isinstance(group, str)
        or not group.startswith("/")
        or not group.endswith("/" + CAMPAIGN_SLICE)
    ):
        raise ValueError("R3 attempt does not identify the authorized campaign parent cgroup")
    root = cgroup_root.resolve(strict=True)
    cgroup = (root / group.lstrip("/")).resolve(strict=True)
    try:
        cgroup.relative_to(root)
    except ValueError as error:
        raise ValueError("R3 parent cgroup escapes the cgroup root") from error
    if cgroup.name != CAMPAIGN_SLICE:
        raise ValueError("R3 parent cgroup is not the campaign slice")

    limits = {
        "memory_high": int((cgroup / "memory.high").read_text()),
        "memory_max": int((cgroup / "memory.max").read_text()),
        "memory_swap_max": int((cgroup / "memory.swap.max").read_text()),
    }
    expected_limits = {
        "memory_high": int(properties["MemoryHigh"]),
        "memory_max": int(properties["MemoryMax"]),
        "memory_swap_max": int(properties["MemorySwapMax"]),
    }
    if limits != expected_limits:
        raise ValueError("campaign cgroup limits changed since the R3 unit started")
    if (
        limits["memory_high"] > PARENT_HIGH_BYTES
        or limits["memory_max"] > PARENT_MAX_BYTES
        or limits["memory_swap_max"] != 0
    ):
        raise ValueError("campaign memory limits exceed the frozen no-swap envelope")
    quota, period = map(int, (cgroup / "cpu.max").read_text().split())
    if period <= 0 or quota > cpu_cap * period:
        raise ValueError("campaign CPU quota exceeds current authorization")
    return cgroup, limits


def load_gate(request: dict[str, Any]) -> Any:
    """Load the R3 gate only when its reviewed source hash still matches."""
    source = BASE / "fold1_gru384_recovery_gate_r3.py"
    if sha256_file(source) != request.get("recovery_gate_source_sha256"):
        raise ValueError("R3 gate source changed after root review")
    spec = importlib.util.spec_from_file_location("fold1_gru384_recovery_gate_r3", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import the reviewed R3 recovery gate")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(
    *, root: Path, packet_sha: str, authority_sha: str, check_only: bool, launch_seed4: bool
) -> dict[str, Any]:
    """Audit R3, reclaim bounded parent file cache, and optionally launch seed4."""
    root = root.resolve(strict=True)
    packet = json.loads(PACKET.read_text(encoding="utf-8"))
    request = json.loads(REQUEST.read_text(encoding="utf-8"))
    if not isinstance(packet, dict) or not isinstance(request, dict):
        raise ValueError("R3 packet/request must be JSON objects")
    gate = load_gate(request)
    validated_packet = gate.load_packet(root, PACKET, packet_sha)
    authority = json.loads(AUTHORITY.read_text(encoding="utf-8"))
    if not isinstance(authority, dict) or sha256_file(AUTHORITY) != authority_sha:
        raise ValueError("root R3 authority hash differs from the wrapper input")
    request_sha = sha256_file(REQUEST)
    gate.verify_authority(authority, authority_sha, request, request_sha)
    if request.get("packet_sha256") != packet_sha:
        raise ValueError("R3 request does not bind the current packet")
    if INTENT.exists() or INTENT_R2.exists() or HANDOFF_R2.exists():
        raise FileExistsError("R3-to-seed4 handoff already attempted; reconcile its receipts")
    if check_only:
        return {
            "status": "static_r3_authority_and_seed4_inputs_passed_no_audit_or_launch",
            "r3_request_sha256": request_sha,
            "r3_authority_sha256": authority_sha,
            "seed4_attempt_id": validated_packet["allocation"]["seed4_attempt_id"],
            "seed4_launch_requested": launch_seed4,
        }

    runtime = root / validated_packet["runtime_relative"]
    stop_marker = runtime / STOP_BASENAME
    if stop_marker.exists():
        raise RuntimeError("campaign admission barrier remains active; seed4 stays pending")
    seed4 = validated_packet["allocation"]
    attempt_path = runtime / "attempts" / f"{seed4['seed4_attempt_id']}.json"
    output_path = root / seed4["seed4_output_path"]
    r2_gate, r2_packet, _r2_request, _r2_authority, _queue, steps = gate.load_r2_inputs(
        root, validated_packet
    )
    old_gate = r2_gate.load_old_gate(root, r2_packet)
    r3_attempt_id = request["resume_attempt_id"]
    r3_unit_name = f"wnn-bandit-{r3_attempt_id}.service"
    r3_unit_before = old_gate.verify_unit(r3_unit_name)
    if AUDIT.exists():
        result = gate.read_json(AUDIT)
    else:
        result = gate.audit(
            argparse.Namespace(
                root=root,
                packet=PACKET,
                packet_sha256=packet_sha,
                request=REQUEST,
                authority=AUTHORITY,
                authority_sha256=authority_sha,
                output=AUDIT,
            )
        )
    if result.get("status") != "recovery_fit_and_output_passed":
        raise ValueError("R3 audit did not establish a complete successful recovery")
    r3_unit_after = old_gate.verify_unit(r3_unit_name)
    r3_attempt_path = runtime / "attempts" / f"{r3_attempt_id}.json"
    r3_attempt = gate.read_json(r3_attempt_path)
    recovered_state_path = root / validated_packet["resume_output_path"] / "training_state.pt"
    r3_state_sha = sha256_file(recovered_state_path)
    r3_attempt_sha = sha256_file(r3_attempt_path)
    audit_sha = sha256_file(AUDIT)
    require_successful_predecessor(
        result,
        r3_attempt,
        r3_unit_after,
        attempt_id=r3_attempt_id,
        authority_sha256=authority_sha,
        request_sha256=request_sha,
        attempt_sha256=r3_attempt_sha,
        unit_name=r3_unit_name,
        recovered_state_sha256=r3_state_sha,
    )
    if not launch_seed4:
        return {
            "status": "r3_audited_seed4_pending_explicit_handoff",
            "audit_sha256": audit_sha,
            "unit_before_audit": r3_unit_before,
            "unit_after_audit": r3_unit_after,
        }

    if stop_marker.exists():
        raise RuntimeError("campaign admission barrier remains active; seed4 stays pending")
    if attempt_path.exists() or output_path.exists():
        raise FileExistsError(
            "seed4 already has output/attempt evidence; refusing duplicate launch"
        )
    dropin = Path(seed4["seed4_unit_memory_high_dropin_path"])
    if not dropin.is_file() or sha256_file(dropin) != seed4["seed4_unit_memory_high_dropin_sha256"]:
        raise ValueError("seed4 exact-unit MemoryHigh drop-in is absent or changed")
    authority_path = root / seed4["authority_path"]
    if sha256_file(authority_path) != seed4["authority_sha256"]:
        raise ValueError("original conditional seed4 allocation changed")
    seed4_step = steps["seed4"]
    expected_seed4 = request["seed4_conditional_successor"]
    if (
        seed4_step["attempt_id"] != expected_seed4["attempt_id"]
        or seed4_step["run_name"] != expected_seed4["run_name"]
        or seed4_step["expected_output_path"] != expected_seed4["output_path"]
        or seed4_step["command_sha256_canonical_json"]
        != expected_seed4["command_sha256_canonical_json"]
        or seed4["authority_sha256"] != expected_seed4["original_authority_sha256"]
    ):
        raise ValueError("original conditional seed4 command or allocation identity changed")
    if shutil.disk_usage(root).free < int(validated_packet["minimum_free_bytes"]):
        raise RuntimeError("free disk fell below the original seed4 admission floor")

    resource_spec = importlib.util.spec_from_file_location(
        "seed4_bandit_resources", root / "scripts/bandit_resources.py"
    )
    if resource_spec is None or resource_spec.loader is None:
        raise RuntimeError("cannot load the current campaign resource policy")
    resources = importlib.util.module_from_spec(resource_spec)
    resource_spec.loader.exec_module(resources)
    cpu_cap = resources.authorized_cpu_cap()
    gpu_lock = runtime / "gpu.lock"
    if not gpu_lock.is_file():
        raise FileNotFoundError("campaign GPU lock is absent; cache handoff cannot proceed")
    with gpu_lock.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "campaign GPU lock is busy; no cache reclaim was attempted"
            ) from error
        if stop_marker.exists():
            raise RuntimeError("campaign STOP marker appeared before cache reclaim")
        unit_after_lock = old_gate.verify_unit(r3_unit_name)
        current_attempt = gate.read_json(r3_attempt_path)
        current_attempt_sha = sha256_file(r3_attempt_path)
        current_audit = gate.read_json(AUDIT)
        require_successful_predecessor(
            current_audit,
            current_attempt,
            unit_after_lock,
            attempt_id=r3_attempt_id,
            authority_sha256=authority_sha,
            request_sha256=request_sha,
            attempt_sha256=current_attempt_sha,
            unit_name=r3_unit_name,
            recovered_state_sha256=sha256_file(recovered_state_path),
        )
        cgroup, limits_before = campaign_parent_cgroup(
            current_attempt, Path("/sys/fs/cgroup"), cpu_cap
        )
        if not (cgroup / "memory.reclaim").exists():
            raise FileNotFoundError("campaign parent memory.reclaim is unavailable")
        before = memory_snapshot(cgroup)
        requested_bytes = bounded_reclaim_bytes(before["current"], before["file"])
        handoff_request = {
            "schema": "wnn.fold1_gru384_campaign_cache_handoff.r2",
            "status": "reclaim_requested",
            "observed_utc": datetime.now(UTC).isoformat(),
            "r3_audit_sha256": audit_sha,
            "r3_attempt_id": r3_attempt_id,
            "r3_attempt_sha256": r3_attempt_sha,
            "r3_unit": r3_unit_name,
            "r3_unit_terminal_state": r3_unit_after,
            "r3_recovered_training_state_sha256": r3_state_sha,
            "parent_cgroup": str(cgroup),
            "limits_before": limits_before,
            "before": before,
            "reclaim_request": {
                "bytes": requested_bytes,
                "swappiness": 0,
                "target_parent_current_bytes": HANDOFF_TARGET_BYTES,
                "maximum_request_bytes": MAX_RECLAIM_BYTES,
                "preserved_file_cache_floor_bytes": PRESERVED_FILE_CACHE_BYTES,
            },
            "seed4_reserve_bytes": SEED4_RESERVE_BYTES,
            "limits_changed": False,
            "global_drop_caches_used": False,
        }
        gate.write_exclusive_json(HANDOFF_R2, handoff_request)
        reclaim_warning = None
        reclaim_error = None
        if requested_bytes:
            try:
                (cgroup / "memory.reclaim").write_text(f"{requested_bytes} swappiness=0\n")
            except OSError as error:
                if error.errno == errno.EAGAIN:
                    reclaim_warning = str(error)
                else:
                    reclaim_error = str(error)
        after = memory_snapshot(cgroup)
        limits_after = {
            "memory_high": int((cgroup / "memory.high").read_text()),
            "memory_max": int((cgroup / "memory.max").read_text()),
            "memory_swap_max": int((cgroup / "memory.swap.max").read_text()),
        }
        ready = (
            reclaim_error is None
            and limits_after == limits_before
            and after["current"] <= HANDOFF_TARGET_BYTES
            and after["current"] + SEED4_RESERVE_BYTES <= limits_after["memory_high"]
        )
        handoff_receipt = {
            **handoff_request,
            "status": "ready" if ready else "needs_review",
            "after": after,
            "limits_after": limits_after,
            "kernel_partial_reclaim_warning": reclaim_warning,
            "reclaim_error": reclaim_error,
            "seed4_reserve_fits": after["current"] + SEED4_RESERVE_BYTES
            <= limits_after["memory_high"],
        }
        old_gate.atomic_json(HANDOFF_R2, handoff_receipt)
        if not ready:
            raise RuntimeError("campaign file-cache handoff did not meet the 8.5 GB target")

    if stop_marker.exists():
        raise RuntimeError("campaign STOP marker appeared after cache handoff; seed4 stays pending")
    if attempt_path.exists() or output_path.exists():
        raise FileExistsError("seed4 evidence appeared after cache handoff; refusing launch")
    handoff_sha = sha256_file(HANDOFF_R2)
    intent = {
        "schema": "wnn.fold1_gru384_seed4_after_r3_intent.r2",
        "status": "seed4_launch_intent",
        "r3_audit_sha256": audit_sha,
        "r3_attempt_sha256": r3_attempt_sha,
        "r3_unit": r3_unit_name,
        "r3_unit_terminal_state": r3_unit_after,
        "r3_recovered_training_state_sha256": r3_state_sha,
        "r3_request_sha256": request_sha,
        "r3_authority_sha256": authority_sha,
        "cache_handoff_sha256": handoff_sha,
        "seed4_attempt_id": seed4_step["attempt_id"],
        "seed4_command_sha256_canonical_json": seed4_step["command_sha256_canonical_json"],
        "original_allocation_sha256": seed4["authority_sha256"],
        "seed4_unit_memory_high_dropin_sha256": seed4["seed4_unit_memory_high_dropin_sha256"],
        "launch_requested_explicitly": True,
    }
    gate.write_exclusive_json(INTENT_R2, intent)
    launched = old_gate.launch_seed4(root, authority_path, runtime, seed4_step)
    unit = launched.get("unit")
    expected_unit = f"wnn-bandit-{seed4_step['attempt_id']}.service"
    if unit != expected_unit:
        receipt = {
            **intent,
            "status": "launch_unit_identity_mismatch_no_stop_attempted",
            "attempt": launched,
            "expected_unit": expected_unit,
        }
        old_gate.atomic_json(INTENT_R2, receipt)
        raise ValueError(f"launched unit identity is not the exact seed4 unit: {unit}")
    result = subprocess.run(
        [
            "systemctl",
            "--user",
            "show",
            unit,
            "--property=MemoryHigh,MemoryMax,MemorySwapMax,CPUQuotaPerSecUSec",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    properties = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    expected = {
        "MemoryHigh": "8000000000",
        "MemoryMax": "9000000000",
        "MemorySwapMax": "0",
        "CPUQuotaPerSecUSec": "2s",
    }
    if result.returncode != 0 or any(
        properties.get(key) != value for key, value in expected.items()
    ):
        launched_record = old_gate.read_json(attempt_path)
        if (
            launched_record.get("id") != seed4_step["attempt_id"]
            or launched_record.get("command") != seed4_step["command"]
            or launched_record.get("unit") != unit
        ):
            receipt = {
                **intent,
                "status": "resource_property_mismatch_identity_unverified_no_stop",
                "attempt": launched,
                "properties": properties,
            }
            old_gate.atomic_json(INTENT_R2, receipt)
            raise ValueError(
                "resource mismatch found, but exact seed4 unit identity could not be reverified"
            )
        stopped = subprocess.run(
            ["systemctl", "--user", "stop", unit],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
        terminal = subprocess.run(
            [
                "systemctl",
                "--user",
                "show",
                unit,
                "--property=ActiveState,SubState,Result,ExecMainStatus",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        terminal_state = dict(
            line.split("=", 1) for line in terminal.stdout.splitlines() if "=" in line
        )
        stopped_ok = (
            stopped.returncode == 0
            and terminal.returncode == 0
            and terminal_state.get("ActiveState") in {"inactive", "failed"}
        )
        receipt = {
            **intent,
            "status": "launch_properties_mismatch_stopped"
            if stopped_ok
            else "launch_properties_mismatch_stop_unverified",
            "attempt": launched,
            "properties": properties,
            "stop_returncode": stopped.returncode,
            "terminal_unit_state": terminal_state,
            "stop_stderr": stopped.stderr[-2000:],
        }
        old_gate.atomic_json(INTENT_R2, receipt)
        raise ValueError(
            f"seed4 resource properties failed; exact-unit stop_verified={stopped_ok}: {properties}"
        )
    receipt = {
        **intent,
        "status": "seed4_reserved",
        "attempt": launched,
        "unit_resource_properties": properties,
    }
    old_gate.atomic_json(INTENT_R2, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/home/vlad/development/wunder/wnn_connectome_starterpack"),
    )
    parser.add_argument("--packet-sha256", required=True)
    parser.add_argument("--authority-sha256", required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--launch-seed4", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                root=args.root,
                packet_sha=args.packet_sha256,
                authority_sha=args.authority_sha256,
                check_only=args.check_only,
                launch_seed4=args.launch_seed4,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
