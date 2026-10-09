"""Audit the exact recovered seed3 before launching the original admitted seed4."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
RECOVERY_GATE_SHA = "52565eb1c9841fad51812fe86654fd9284bf65a002be086654f5481654c960d4"
PACKET_SHA = "2546654dde9e2cd1385e8c83ebea1739c3adb1e5872cbb818068b59dc126c8a7"
AUTHORITY_SHA = "b56bf0df49779fb46aa9d70b2b3acf4558dd1c5a31c087694b9e9bc514afc489"
REQUEST_SHA = "46f4b21c26eee379fd4212fe53d022285ac9d7a40157e14de60ac03c070ae0a5"


def load_recovery_gate() -> ModuleType:
    """Load only the reviewed implementation and its exact root authority."""
    pins = {
        "fold1_gru384_recovery_gate.py": RECOVERY_GATE_SHA,
        "RECOVERY_GATE_PACKET_R2.json": PACKET_SHA,
        "ROOT_RESUME_AUTHORITY_R2.json": AUTHORITY_SHA,
        "RESUME_REQUEST_R2.json": REQUEST_SHA,
    }
    for name, expected in pins.items():
        if hashlib.sha256((BASE / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"recovery input changed: {name}")
    spec = importlib.util.spec_from_file_location(
        "reviewed_fold1_recovery", BASE / "fold1_gru384_recovery_gate.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load reviewed recovery gate")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(*, check_only: bool) -> None:
    """Fail closed on an incomplete recovery, changed source or duplicate launch."""
    gate = load_recovery_gate()
    packet_path = BASE / "RECOVERY_GATE_PACKET_R2.json"
    packet = gate.packet_for(ROOT, packet_path, PACKET_SHA)
    queue, _, _, seed4 = gate.load_original_inputs(ROOT, packet)
    old_gate = gate.load_old_gate(ROOT, packet)
    old_packet_path = ROOT / packet["original_postfit_gate_packet_path"]
    old_packet = gate.read_json(old_packet_path)
    old_gate.run(old_packet_path, packet["original_postfit_gate_packet_sha256"], True)
    request = gate.read_json(BASE / "RESUME_REQUEST_R2.json")
    authority_path = BASE / "ROOT_RESUME_AUTHORITY_R2.json"
    gate.verify_authority(gate.read_json(authority_path), AUTHORITY_SHA, request, REQUEST_SHA)
    runtime = ROOT / packet["runtime_relative"]
    intent_path = BASE / "SEED4_AFTER_RECOVERY_RECEIPT_R1.json"
    audit_path = BASE / "RESUME_TERMINAL_AUDIT_R2.json"
    if intent_path.exists() or audit_path.exists():
        raise FileExistsError("post-recovery audit/launch already attempted; reconcile receipts")
    if check_only:
        print(json.dumps({"status": "static_checks_passed_no_audit_or_launch"}))
        return
    unit_receipt = old_gate.verify_unit(f"wnn-bandit-{request['resume_attempt_id']}.service")
    result = gate.audit(
        argparse.Namespace(
            root=ROOT,
            packet=packet_path,
            packet_sha256=PACKET_SHA,
            request=BASE / "RESUME_REQUEST_R2.json",
            authority=authority_path,
            authority_sha256=AUTHORITY_SHA,
            output=audit_path,
        )
    )
    if result["status"] != "recovery_fit_and_output_passed":
        raise ValueError("recovery has no successful terminal output audit")
    # Persist intent before provider/controller mutation; never retry automatically.
    receipt = {
        "status": "seed4_launch_intent",
        "recovery_audit_sha256": gate.sha256_file(audit_path),
        "recovery_terminal_unit": unit_receipt,
        "seed4_attempt": seed4["attempt_id"],
        "seed4_command_sha256": seed4["command_sha256_canonical_json"],
        "seed4_authority_sha256": queue["root_authority_sha256"],
    }
    gate.write_exclusive_json(intent_path, receipt)
    launched = old_gate.launch_seed4(ROOT, ROOT / old_packet["authority_path"], runtime, seed4)
    old_gate.atomic_json(intent_path, {**receipt, "status": "seed4_reserved", "attempt": launched})
    print(json.dumps({"status": "seed4_reserved", "attempt": seed4["attempt_id"]}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    run(check_only=args.check_only)


if __name__ == "__main__":
    main()
