from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import subprocess
import uuid
from pathlib import Path

import terminal_gate as gate

PACKET = Path(__file__).resolve().parent
B = Path("/home/grinchenkoeg/computer-vision-homework-20261018")
REMOTE_PACKET = B / "research/campaign_2026-10-18/priority25_execution_r4/f3_terminal_execution_r1"
SSH = [
    "ssh",
    "-T",
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=10",
    "-o",
    "ConnectionAttempts=1",
    "beleriand",
]
REMOTE_PYTHON = "/home/grinchenkoeg/WunderFunder/.venv/bin/python"


def sha256(path: Path) -> str:
    """Return the hash of a local regular, non-symlink packet file."""
    if path.is_symlink() or not path.is_file():
        raise gate.GateError(f"packet file missing or symlinked: {path.name}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_local_packet() -> tuple[dict[str, object], str, str]:
    """Require byte-pinned local sources and an independent PASS before SSH."""
    manifest_path = PACKET / "PREPARATION_MANIFEST.json"
    review_path = PACKET / "INDEPENDENT_REVIEW_V1.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest_sha = sha256(manifest_path)
    review_sha = sha256(review_path)
    review = json.loads(review_path.read_bytes())
    if review.get("status") != "passed" or not str(review.get("decision", "")).startswith("PASS"):
        raise gate.GateError("independent static PASS review is required")
    if review.get("manifest_sha256") != manifest_sha or review.get("blockers") not in ([], None):
        raise gate.GateError("independent review does not bind this manifest cleanly")
    for name, item in manifest["artifacts"].items():
        if sha256(PACKET / name) != item["sha256"]:
            raise gate.GateError(f"packet artifact hash differs: {name}")
    return manifest, manifest_sha, review_sha


def write_exclusive(path: Path, raw: bytes) -> None:
    """Write one local record without replacing existing evidence."""
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()


def main() -> None:
    """Reserve local evidence, then request one fresh B root admission."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("wide", "deep"), required=True)
    parser.add_argument("--operation", choices=("collector", "verifier"), required=True)
    parser.add_argument("--invocation", help="unique lowercase slug; generated when omitted")
    args = parser.parse_args()
    manifest, manifest_sha, review_sha = verify_local_packet()
    invocation = (
        args.invocation
        or dt.datetime.now(dt.UTC).strftime("t%Y%m%d%H%M%S%f") + "-" + uuid.uuid4().hex[:8]
    )
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{7,63}", invocation):
        raise gate.GateError("invocation must be a unique lowercase alphanumeric slug")
    records = PACKET / "records"
    if records.is_symlink():
        raise gate.GateError("local records path is a symlink")
    records.mkdir(exist_ok=True)
    local_dir = records / invocation
    local_dir.mkdir(exist_ok=False)
    intent = {
        "schema": "priority25.f3_terminal_local_launch_intent.v1",
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "arm": args.arm,
        "operation": args.operation,
        "invocation": invocation,
        "plan_sha256": gate.PLAN_SHA256,
        "manifest_sha256": manifest_sha,
        "review_sha256": review_sha,
        "status": "TRANSPORT_PENDING_NO_RETRY_IF_OUTCOME_UNKNOWN",
    }
    write_exclusive(
        local_dir / "LOCAL_LAUNCH_INTENT.json",
        json.dumps(intent, sort_keys=True, indent=2).encode() + b"\n",
    )
    print(
        json.dumps(
            {
                "status": "ROOT_GATE_REQUESTED",
                "invocation": invocation,
                "local_record_dir": str(local_dir),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    remote = [
        REMOTE_PYTHON,
        "-B",
        str(REMOTE_PACKET / "terminal_remote_v1.py"),
        "launch",
        "--arm",
        args.arm,
        "--operation",
        args.operation,
        "--invocation",
        invocation,
        "--manifest-sha256",
        manifest_sha,
        "--review-sha256",
        review_sha,
        "--remote-sha256",
        manifest["runtime_files"]["terminal_remote_v1.py"]["sha256"],
    ]
    try:
        result = subprocess.run(
            SSH + remote, capture_output=True, text=True, timeout=90, check=False
        )
    except subprocess.TimeoutExpired as exc:
        failure = {
            "status": "UNKNOWN_REMOTE_LAUNCH_OUTCOME_NO_RETRY",
            "timeout_seconds": 90,
            "stdout": str(exc.stdout),
            "stderr": str(exc.stderr),
        }
        write_exclusive(
            local_dir / "LOCAL_LAUNCH_TRANSPORT_FAILURE.json",
            json.dumps(failure, sort_keys=True, indent=2).encode() + b"\n",
        )
        raise gate.GateError(
            "SSH launch timed out; preserve the record and inspect B before any further action"
        ) from exc
    response_raw = result.stdout.encode()
    response_path = local_dir / "REMOTE_LAUNCH_RESPONSE.json"
    write_exclusive(response_path, response_raw)
    if result.returncode != 0:
        failure = {
            "status": "REMOTE_LAUNCH_FAILED_OR_UNKNOWN",
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
        write_exclusive(
            local_dir / "LOCAL_LAUNCH_FAILURE.json",
            json.dumps(failure, sort_keys=True, indent=2).encode() + b"\n",
        )
        raise gate.GateError(
            f"remote preflight/launch failed ({result.returncode}); evidence is reserved"
        )
    response = json.loads(response_raw)
    if response.get("status") not in {"LAUNCHED", "LAUNCH_UNCONFIRMED"}:
        raise gate.GateError("remote response did not record the exact support-unit launch attempt")
    print(
        json.dumps(
            {
                "status": response["status"],
                "arm": args.arm,
                "operation": args.operation,
                "invocation": invocation,
                "local_record_dir": str(local_dir),
                "remote_record_dir": response["remote_record_dir"],
                "unit": response["unit"],
                "service_started": response.get("service_started"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
