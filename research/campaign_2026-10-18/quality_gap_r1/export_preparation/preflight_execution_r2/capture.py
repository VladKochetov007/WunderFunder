"""Capture the raw R2 preflight receipt without normalizing or overwriting it."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from bindings import B_PROJECT_ROOT, OVERLAY_REL, validate_preflight_receipt
from local_common import (
    get_record_dir,
    read_json,
    utc_now,
    write_bytes_exclusive,
    write_json_exclusive,
)

SSH_OPTIONS = ["-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "ConnectionAttempts=1"]
MAX_RECEIPT_BYTES = 1_000_000


def _validate_alias(alias: str) -> None:
    """Restrict SSH host arguments to an existing simple config alias."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", alias):
        raise ValueError("SSH host must be a simple configured alias")


def _remote_metadata(alias: str, receipt_path: Path) -> tuple[int, str, str]:
    """Read remote type, canonical path, size, and SHA for the small receipt."""
    _validate_alias(alias)
    quoted = str(receipt_path)
    command = (
        "set -eu; "
        f"test -f {quoted}; test ! -L {quoted}; "
        f"realpath -e -- {quoted}; "
        f"stat -c '%s' -- {quoted}; "
        f"sha256sum -- {quoted}"
    )
    result = subprocess.run(
        ["ssh", *SSH_OPTIONS, alias, command],
        capture_output=True,
        timeout=30,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "remote receipt identity probe failed: "
            + result.stderr.decode("utf-8", errors="replace")[-2000:]
        )
    lines = [
        line.strip()
        for line in result.stdout.decode("utf-8", errors="strict").splitlines()
        if line.strip()
    ]
    if len(lines) < 3:
        raise ValueError("remote receipt identity probe returned incomplete output")
    resolved_path = lines[-3]
    size = int(lines[-2])
    match = re.fullmatch(r"([0-9a-f]{64})\s+.+", lines[-1])
    if match is None:
        raise ValueError("remote receipt SHA output is malformed")
    if resolved_path != str(receipt_path):
        raise ValueError("remote receipt resolved path differs from the exact staged output path")
    if size <= 0 or size > MAX_RECEIPT_BYTES:
        raise ValueError(f"remote receipt size is outside the safe capture bound: {size}")
    return size, match.group(1), resolved_path


def _copy_receipt(alias: str, receipt_path: Path, transport_dir: Path) -> bytes:
    """Copy receipt bytes into a newly created local transport directory."""
    _validate_alias(alias)
    if transport_dir.exists() or transport_dir.is_symlink():
        raise FileExistsError("capture transport directory already exists")
    transport_dir.mkdir(mode=0o700)
    target = transport_dir / receipt_path.name
    command = [
        "scp",
        "-q",
        "-p",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "ConnectionAttempts=1",
        f"{alias}:{receipt_path}",
        str(transport_dir),
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        timeout=60,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "receipt scp failed: " + result.stderr.decode("utf-8", errors="replace")[-2000:]
        )
    if target.is_symlink() or not target.is_file():
        raise ValueError("scp did not create a regular local receipt file")
    data = target.read_bytes()
    if len(data) > MAX_RECEIPT_BYTES:
        raise ValueError("captured receipt exceeded the safe byte limit")
    return data


def capture(attempt_id: str, alias: str) -> Path:
    """Preserve remote receipt bytes, compare transfer digest, and check bindings."""
    record_dir = get_record_dir(attempt_id)
    execution = read_json(record_dir / "EXECUTION.json")
    if (
        execution.get("attempt_id") != attempt_id
        or execution.get("returncode") != 0
        or execution.get("preflight_summary", {}).get("status") != "PASS_CPU_LOADER_PREFLIGHT"
    ):
        raise ValueError("this attempt has no successful CPU preflight execution")
    receipt_path = B_PROJECT_ROOT / OVERLAY_REL / "COMPATIBILITY_RECEIPT_R2.json"
    local_raw_path = record_dir / "COMPATIBILITY_RECEIPT_R2.json"
    if local_raw_path.exists() or local_raw_path.is_symlink():
        raise FileExistsError("local raw preflight receipt already exists")
    remote_size, remote_sha, remote_resolved = _remote_metadata(alias, receipt_path)
    data = _copy_receipt(alias, receipt_path, record_dir / "capture_transport")
    digest = hashlib.sha256(data).hexdigest()
    write_bytes_exclusive(local_raw_path, data)
    validation_error: str | None = None
    try:
        if len(data) != remote_size or digest != remote_sha:
            raise ValueError("captured raw bytes differ from the remote size/SHA observation")
        payload = json.loads(data)
        if not isinstance(payload, dict):
            raise ValueError("preflight receipt is not a JSON object")
        validate_preflight_receipt(payload)
    except Exception as exc:
        validation_error = str(exc)
    capture_record: dict[str, Any] = {
        "schema": "quality_gap.preflight_receipt_capture.r2",
        "status": "CAPTURED_AND_VALIDATED"
        if validation_error is None
        else "CAPTURED_VALIDATION_FAILED",
        "attempt_id": attempt_id,
        "captured_utc": utc_now(),
        "remote_receipt_path": str(receipt_path),
        "remote_resolved_path": remote_resolved,
        "remote_bytes": remote_size,
        "remote_sha256": remote_sha,
        "captured_bytes": len(data),
        "captured_sha256": digest,
        "receipt_sha256_was_predetermined": False,
        "validation_error": validation_error,
        "root_admission": "NOT_ADMITTED",
    }
    write_json_exclusive(record_dir / "CAPTURE.json", capture_record)
    if validation_error is not None:
        raise ValueError(validation_error)
    return local_raw_path


def main() -> int:
    """Copy one finished receipt to the existing local attempt record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssh-alias", default="beleriand")
    parser.add_argument("--attempt-id", required=True)
    args = parser.parse_args()
    result = capture(args.attempt_id, args.ssh_alias)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
