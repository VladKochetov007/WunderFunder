"""Local-only evidence and SSH helpers shared by the R2 staging utilities."""

from __future__ import annotations

import json
import re
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EXECUTION_DIR = Path(__file__).resolve().parent
ATTEMPT_RE = re.compile(r"^preflight-r2-[a-z0-9-]{8,70}$")
SSH_OPTIONS = ["-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "ConnectionAttempts=1"]


def utc_now() -> str:
    """Return a timezone-aware UTC timestamp for a local evidence record."""
    return datetime.now(timezone.utc).isoformat()  # noqa: UP017 - Python 3.9 support


def validate_attempt_id(attempt_id: str) -> None:
    """Reject attempt IDs that could alter paths or systemd unit names."""
    if not ATTEMPT_RE.fullmatch(attempt_id):
        raise ValueError("attempt id must match preflight-r2-<lowercase-id>")


def create_record_dir(attempt_id: str) -> Path:
    """Create one unique local evidence directory, never reusing an attempt."""
    validate_attempt_id(attempt_id)
    records = EXECUTION_DIR / "records"
    records.mkdir(exist_ok=True)
    path = records / attempt_id
    path.mkdir(mode=0o700, exist_ok=False)
    return path


def get_record_dir(attempt_id: str) -> Path:
    """Return an existing attempt directory after checking its path shape."""
    validate_attempt_id(attempt_id)
    path = EXECUTION_DIR / "records" / attempt_id
    if path.is_symlink() or not path.is_dir():
        raise FileNotFoundError(f"local preflight attempt does not exist: {attempt_id}")
    return path


def write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    """Create a JSON evidence file without replacing an existing record."""
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def write_bytes_exclusive(path: Path, payload: bytes) -> None:
    """Create a raw evidence file without replacing prior bytes."""
    with path.open("xb") as stream:
        stream.write(payload)


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object from a local attempt record."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def parse_remote_json(stdout: bytes) -> dict[str, Any]:
    """Parse the last JSON object line while tolerating login banners."""
    for line in reversed(stdout.decode("utf-8", errors="replace").splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("remote helper did not return a JSON object")


def ssh_argv(alias: str, remote_args: list[str]) -> list[str]:
    """Build a noninteractive SSH call using an existing configured alias."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", alias):
        raise ValueError("SSH host must be a simple configured alias")
    return ["ssh", *SSH_OPTIONS, alias, shlex.join(remote_args)]


def run_ssh(
    alias: str,
    remote_args: list[str],
    *,
    timeout: int,
    input_bytes: bytes | None = None,
) -> subprocess.CompletedProcess[bytes]:
    """Run one bounded SSH command without a PTY or local shell."""
    return subprocess.run(
        ssh_argv(alias, remote_args),
        input=input_bytes,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def verify_remote_file_sha256(alias: str, path: Path, expected_sha256: str) -> None:
    """Check a staged helper's digest before invoking its Python interpreter."""
    result = run_ssh(alias, ["sha256sum", "--", str(path)], timeout=30)
    if result.returncode != 0:
        raise RuntimeError(
            "remote helper hash check failed: "
            + result.stderr.decode("utf-8", errors="replace")[-2000:]
        )
    lines = result.stdout.decode("utf-8", errors="strict").splitlines()
    matches = [
        line.split(maxsplit=1)[0]
        for line in lines
        if re.fullmatch(r"[0-9a-f]{64}\s+.+", line.strip())
    ]
    if not matches or matches[-1] != expected_sha256:
        raise ValueError(f"staged helper digest changed: {path}")
