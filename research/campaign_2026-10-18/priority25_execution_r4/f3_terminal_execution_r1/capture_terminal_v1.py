from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import uuid

import terminal_gate as gate
from launch_terminal_v1 import (
    PACKET,
    REMOTE_PACKET,
    REMOTE_PYTHON,
    SSH,
    verify_local_packet,
    write_exclusive,
)


def expected_receipt_name(arm: str, operation: str) -> str:
    """Return the exact canonical helper output bound by the frozen plan."""
    suffix = "COLLECTION" if operation == "collector" else "VERIFICATION"
    return f"RANK01_{arm.upper()}_F3_S1001_TERMINAL_{suffix}.json"


def helper_receipt_status(
    raw: bytes, arm: str, operation: str
) -> tuple[dict[str, object], bool, str | None]:
    """Classify helper output while preserving failures as evidence."""
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {}, False, f"helper output is invalid JSON: {exc.msg}"
    if not isinstance(value, dict):
        return {}, False, "helper output JSON is not an object"
    if operation == "collector":
        attempt = value.get("attempt_evidence", {})
        if not isinstance(attempt, dict):
            attempt = {}
        valid = (
            value.get("schema") == "priority25.f3_checkpoint_collection.v1"
            and value.get("arm") == arm
            and value.get("phase") == "terminal"
            and value.get("plan_sha256") == gate.PLAN_SHA256
            and attempt.get("status") == "finished"
            and attempt.get("exit_status") == 0
            and value.get("source_snapshot_hash_match") is True
        )
    else:
        valid = (
            value.get("schema") == "priority25.f3_checkpoint_verification.v1"
            and value.get("status") == "passed"
            and value.get("arm") == arm
            and value.get("phase") == "terminal"
            and value.get("plan_sha256") == gate.PLAN_SHA256
            and value.get("fold3_manifest_sha256") == gate.FOLD_SHA256
            and value.get("next_epoch") == 6
        )
    return (
        value,
        valid,
        None if valid else "captured helper output does not satisfy its success contract",
    )


def main() -> None:
    """Capture raw B evidence locally, then request stop of only our exited unit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("wide", "deep"), required=True)
    parser.add_argument("--operation", choices=("collector", "verifier"), required=True)
    parser.add_argument("--invocation", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{7,63}", args.invocation):
        raise gate.GateError("invocation must be a unique lowercase alphanumeric slug")
    _, manifest_sha, review_sha = verify_local_packet()
    records = PACKET / "records"
    if records.is_symlink() or not records.is_dir():
        raise gate.GateError("local records root is missing or symlinked")
    local_dir = records / args.invocation
    if (
        local_dir.is_symlink()
        or not local_dir.is_dir()
        or local_dir.resolve().parent != records.resolve()
    ):
        raise gate.GateError("local invocation record directory is missing or redirected")
    intent_path = local_dir / "LOCAL_LAUNCH_INTENT.json"
    launch_path = local_dir / "REMOTE_LAUNCH_RESPONSE.json"
    if not intent_path.is_file() or not launch_path.is_file():
        raise gate.GateError("matching local launch records are required")
    intent = json.loads(intent_path.read_bytes())
    launch = json.loads(launch_path.read_bytes())
    if (intent.get("arm"), intent.get("operation"), intent.get("invocation")) != (
        args.arm,
        args.operation,
        args.invocation,
    ):
        raise gate.GateError("local launch intent does not match capture arguments")
    if intent.get("manifest_sha256") != manifest_sha or intent.get("review_sha256") != review_sha:
        raise gate.GateError("local launch intent binds a different reviewed packet")
    if launch.get("status") not in {"LAUNCHED", "LAUNCH_UNCONFIRMED"}:
        raise gate.GateError("local launch response does not record a support-unit launch attempt")
    expected_unit = f"computer-vision-homework-r01-f3-{args.arm}-terminal-{args.operation}-{args.invocation}.service"
    if (launch.get("invocation"), launch.get("unit")) != (args.invocation, expected_unit):
        raise gate.GateError("local launch response has a different invocation or unit")
    receipt_name = expected_receipt_name(args.arm, args.operation)
    local_outputs = {
        "REMOTE_CAPTURE_RESPONSE.json",
        "REMOTE_ADMISSION_V1.json",
        "REMOTE_LAUNCH_V1.json",
        f"REMOTE_{receipt_name}",
        "REMOTE_HELPER_RECEIPT.json",
        "LOCAL_EXECUTION_V1.json",
        "LOCAL_STOP_V1.json",
        "LOCAL_STOP_TRANSPORT_FAILURE.json",
        "REMOTE_CAPTURE_FAILURE.json",
    }
    if any(
        (local_dir / name).exists() or (local_dir / name).is_symlink() for name in local_outputs
    ):
        raise gate.GateError("capture output collision; preserve evidence and do not overwrite")
    remote_args = [
        REMOTE_PYTHON,
        "-B",
        str(REMOTE_PACKET / "terminal_remote_v1.py"),
        "capture",
        "--arm",
        args.arm,
        "--operation",
        args.operation,
        "--invocation",
        args.invocation,
        "--manifest-sha256",
        manifest_sha,
        "--review-sha256",
        review_sha,
        "--remote-sha256",
        json.loads((PACKET / "PREPARATION_MANIFEST.json").read_bytes())["runtime_files"][
            "terminal_remote_v1.py"
        ]["sha256"],
    ]
    try:
        result = subprocess.run(
            SSH + remote_args, capture_output=True, text=True, timeout=45, check=False
        )
    except subprocess.TimeoutExpired as exc:
        # The remote capture may already be durable. Preserve this transport
        # event separately and allow a later idempotent read of CAPTURE_V1.
        failure = {
            "status": "REMOTE_CAPTURE_OUTCOME_UNKNOWN_NO_STOP",
            "timeout_seconds": 45,
            "stdout": str(exc.stdout),
            "stderr": str(exc.stderr),
        }
        write_exclusive(
            local_dir / f"REMOTE_CAPTURE_TRANSPORT_FAILURE_{uuid.uuid4().hex}.json",
            json.dumps(failure, sort_keys=True, indent=2).encode() + b"\n",
        )
        raise gate.GateError(
            "remote capture timed out; no stop was requested and the exact capture is safe to read again"
        ) from exc
    if result.returncode != 0:
        failure = {
            "status": "REMOTE_CAPTURE_FAILED_NO_STOP",
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
        write_exclusive(
            local_dir / "REMOTE_CAPTURE_FAILURE.json",
            json.dumps(failure, sort_keys=True, indent=2).encode() + b"\n",
        )
        raise gate.GateError(f"remote capture failed ({result.returncode}); unit was not stopped")
    response_raw = result.stdout.encode()
    try:
        response = json.loads(response_raw)
    except json.JSONDecodeError as exc:
        failure = {
            "status": "REMOTE_CAPTURE_RESPONSE_INVALID_NO_STOP",
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "error": exc.msg,
        }
        write_exclusive(
            local_dir / "REMOTE_CAPTURE_FAILURE.json",
            json.dumps(failure, sort_keys=True, indent=2).encode() + b"\n",
        )
        raise gate.GateError(
            "remote capture response was invalid JSON; no stop was requested"
        ) from exc
    if response.get("status") == "WAITING_SUPPORT_EXIT":
        print(json.dumps(response, sort_keys=True))
        return
    if response.get("status") != "TERMINAL_CAPTURE":
        write_exclusive(
            local_dir / "REMOTE_CAPTURE_FAILURE.json",
            json.dumps(
                {
                    "status": "REMOTE_CAPTURE_UNEXPECTED_STATUS_NO_STOP",
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                },
                sort_keys=True,
                indent=2,
            ).encode()
            + b"\n",
        )
        raise gate.GateError("remote capture did not return a terminal record")
    write_exclusive(local_dir / "REMOTE_CAPTURE_RESPONSE.json", response_raw)
    if response.get("unit") != expected_unit:
        raise gate.GateError("remote capture unit identity differs; no stop requested")
    raw_files = gate.decode_raw_files(response.get("raw_files", {}))
    if set(raw_files) not in (
        {"ADMISSION_V1.json", "LAUNCH_V1.json"},
        {"ADMISSION_V1.json", "LAUNCH_V1.json", receipt_name},
    ):
        raise gate.GateError("captured raw file allowlist differs")
    for name, raw in raw_files.items():
        write_exclusive(local_dir / f"REMOTE_{name}", raw)
    admission = json.loads(raw_files["ADMISSION_V1.json"])
    remote_launch = json.loads(raw_files["LAUNCH_V1.json"])
    if (
        admission.get("invocation") != args.invocation
        or remote_launch.get("invocation") != args.invocation
    ):
        raise gate.GateError("captured remote invocation differs")
    helper_receipt: dict[str, object] = {}
    helper_success = False
    helper_error: str | None = "helper receipt was not written"
    if receipt_name in raw_files:
        write_exclusive(local_dir / "REMOTE_HELPER_RECEIPT.json", raw_files[receipt_name])
        helper_receipt, helper_success, helper_error = helper_receipt_status(
            raw_files[receipt_name], args.arm, args.operation
        )
    execution = {
        "schema": "priority25.f3_terminal_local_execution.v1",
        "captured_utc": dt.datetime.now(dt.UTC).isoformat(),
        "arm": args.arm,
        "operation": args.operation,
        "invocation": args.invocation,
        "unit": response.get("unit"),
        "unit_success": response.get("unit_success") is True,
        "helper_success": helper_success,
        "helper_validation_error": helper_error,
        "raw_file_sha256": {name: gate.sha256_bytes(raw) for name, raw in raw_files.items()},
        "journal_tail": response.get("journal_tail"),
        "parent_memory_events": response.get("parent_memory_events"),
        "populated_descendant_paths_before_stop": response.get(
            "populated_descendant_paths_before_stop"
        ),
        "capture_saved_before_stop": True,
    }
    write_exclusive(
        local_dir / "LOCAL_EXECUTION_V1.json",
        json.dumps(execution, sort_keys=True, indent=2).encode() + b"\n",
    )
    stop_args = [
        REMOTE_PYTHON,
        "-B",
        str(REMOTE_PACKET / "terminal_remote_v1.py"),
        "stop",
        "--arm",
        args.arm,
        "--operation",
        args.operation,
        "--invocation",
        args.invocation,
        "--manifest-sha256",
        manifest_sha,
        "--review-sha256",
        review_sha,
        "--remote-sha256",
        json.loads((PACKET / "PREPARATION_MANIFEST.json").read_bytes())["runtime_files"][
            "terminal_remote_v1.py"
        ]["sha256"],
    ]
    try:
        stopped = subprocess.run(
            SSH + stop_args, capture_output=True, text=True, timeout=35, check=False
        )
    except subprocess.TimeoutExpired as exc:
        failure = {
            "status": "UNKNOWN_STOP_OUTCOME_NO_RETRY",
            "stdout": str(exc.stdout),
            "stderr": str(exc.stderr),
            "unit": response.get("unit"),
        }
        write_exclusive(
            local_dir / "LOCAL_STOP_TRANSPORT_FAILURE.json",
            json.dumps(failure, sort_keys=True, indent=2).encode() + b"\n",
        )
        raise gate.GateError(
            "remote stop timed out; capture is durable, inspect unit before any further action"
        ) from exc
    stop_record = {
        "schema": "priority25.f3_terminal_local_stop.v1",
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "returncode": stopped.returncode,
        "stdout": stopped.stdout,
        "stderr": stopped.stderr,
        "capture_saved_before_stop": True,
        "stop_only_named_unit": response.get("unit"),
    }
    write_exclusive(
        local_dir / "LOCAL_STOP_V1.json",
        json.dumps(stop_record, sort_keys=True, indent=2).encode() + b"\n",
    )
    remote_stop: dict[str, object] = {}
    if stopped.returncode == 0:
        remote_stop = json.loads(stopped.stdout)
    if stopped.returncode != 0 or remote_stop.get("returncode") != 0:
        raise gate.GateError("stop command failed; preserve capture and inspect without retry")
    final_status = (
        "CAPTURED_AND_OWN_UNIT_STOPPED"
        if execution["unit_success"] and execution["helper_success"]
        else "FAILED_SUPPORT_CAPTURED_AND_OWN_UNIT_STOPPED"
    )
    print(
        json.dumps(
            {
                "status": final_status,
                "unit_success": execution["unit_success"],
                "helper_success": execution["helper_success"],
                "invocation": args.invocation,
                "local_record_dir": str(local_dir),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
