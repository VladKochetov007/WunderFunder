#!/usr/bin/env python3
"""Audit R3 and optionally hand off to the original conditional seed4 fit.

This wrapper cannot launch R3. Seed4 requires a separate explicit
`--launch-seed4` invocation after the R3 terminal audit has passed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path
from typing import Any

BASE = Path(__file__).resolve().parent
PACKET = BASE / "RECOVERY_GATE_PACKET_R3.json"
REQUEST = BASE / "RESUME_REQUEST_R3.json"
AUTHORITY = BASE / "ROOT_RESUME_AUTHORITY_R3.json"
AUDIT = BASE / "RESUME_TERMINAL_AUDIT_R3.json"
INTENT = BASE / "SEED4_AFTER_R3_RECEIPT_R1.json"


def sha256_file(path: Path) -> str:
    """Hash a small receipt or source file in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    """Audit successful R3 and optionally dispatch the frozen original seed4 step."""
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
    if INTENT.exists():
        raise FileExistsError("R3-to-seed4 handoff already attempted; reconcile its intent receipt")
    if check_only:
        return {
            "status": "static_r3_authority_and_seed4_inputs_passed_no_audit_or_launch",
            "r3_request_sha256": request_sha,
            "r3_authority_sha256": authority_sha,
            "seed4_attempt_id": validated_packet["allocation"]["seed4_attempt_id"],
            "seed4_launch_requested": launch_seed4,
        }

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
    if not launch_seed4:
        return {"status": "r3_audited_seed4_pending_explicit_handoff", "audit": result}

    runtime = root / validated_packet["runtime_relative"]
    stop_marker = runtime / "STOP_ADMISSION.json"
    if stop_marker.exists():
        raise RuntimeError("campaign admission barrier remains active; seed4 stays pending")
    seed4 = validated_packet["allocation"]
    attempt_path = runtime / "attempts" / f"{seed4['seed4_attempt_id']}.json"
    output_path = root / seed4["seed4_output_path"]
    if attempt_path.exists() or output_path.exists():
        raise FileExistsError(
            "seed4 already has output/attempt evidence; refusing duplicate launch"
        )
    dropin = Path(seed4["seed4_unit_memory_high_dropin_path"])
    if not dropin.is_file() or sha256_file(dropin) != seed4["seed4_unit_memory_high_dropin_sha256"]:
        raise ValueError("seed4 exact-unit MemoryHigh drop-in is absent or changed")
    r2_gate, r2_packet, _r2_request, _r2_authority, _queue, steps = gate.load_r2_inputs(
        root, validated_packet
    )
    old_gate = r2_gate.load_old_gate(root, r2_packet)
    authority_path = root / seed4["authority_path"]
    if sha256_file(authority_path) != seed4["authority_sha256"]:
        raise ValueError("original conditional seed4 allocation changed")
    seed4_step = steps["seed4"]
    intent = {
        "schema": "wnn.fold1_gru384_seed4_after_r3_intent.r1",
        "status": "seed4_launch_intent",
        "r3_audit_sha256": sha256_file(AUDIT),
        "r3_request_sha256": request_sha,
        "r3_authority_sha256": authority_sha,
        "seed4_attempt_id": seed4_step["attempt_id"],
        "seed4_command_sha256_canonical_json": seed4_step["command_sha256_canonical_json"],
        "original_allocation_sha256": seed4["authority_sha256"],
        "seed4_unit_memory_high_dropin_sha256": seed4["seed4_unit_memory_high_dropin_sha256"],
        "launch_requested_explicitly": True,
    }
    gate.write_exclusive_json(INTENT, intent)
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
        old_gate.atomic_json(INTENT, receipt)
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
            old_gate.atomic_json(INTENT, receipt)
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
        old_gate.atomic_json(INTENT, receipt)
        raise ValueError(
            f"seed4 resource properties failed; exact-unit stop_verified={stopped_ok}: {properties}"
        )
    receipt = {
        **intent,
        "status": "seed4_reserved",
        "attempt": launched,
        "unit_resource_properties": properties,
    }
    old_gate.atomic_json(INTENT, receipt)
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
