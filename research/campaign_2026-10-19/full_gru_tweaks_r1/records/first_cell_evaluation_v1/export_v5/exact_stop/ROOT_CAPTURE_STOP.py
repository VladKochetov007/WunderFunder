"""Send one exact reviewed exited-export stop and preserve raw bytes before parsing."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

D = Path(
    "/Users/nb/Documents/wunder/WunderFunder-maintenance/research/campaign_2026-10-19/full_gru_tweaks_r1/records/first_cell_evaluation_v1/export_v5/exact_stop"
)
BODY_PIN = (18079, "a0664d02eb40f30fce479aae36e6354d8f639afb0d7910bb38fe1867c1ee6bc1")
REVIEW_PIN = (4031, "d1e02fb08b6583cbb462530c42929c9fc6ff6482d4d82e9e51ec16f2498ecbb2")
ACCEPTANCE_PIN = (8833, "5a9a99121f80eb7fa07e4591318bea81078ca8b2db1032f150cc33afcc728d0d")
UNIT = "computer-vision-homework-fg19-first-cell-export-v3.service"
INVOCATION = "78fa25d0abc44d199e3696d172dbc4f9"


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read_pin(path: Path, expected: tuple[int, str]) -> bytes:
    if path.is_symlink() or not path.is_file() or path.stat().st_size != expected[0]:
        raise ValueError(f"wrong regular metadata input: {path}")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        raw = os.read(fd, expected[0] + 1)
    finally:
        os.close(fd)
    if (len(raw), digest(raw)) != expected:
        raise ValueError(f"input byte pin differs: {path}")
    return raw


def fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write(name: str, raw: bytes) -> None:
    with (D / name).open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    fsync_directory(D)


def main() -> None:
    if D.is_symlink() or not D.is_dir():
        raise ValueError("exact private stop directory is absent or a symlink")
    for name in ("STDOUT.raw.json", "STDERR.raw.txt", "TRANSPORT.json", "EXACT_STOP.json"):
        if (D / name).exists() or (D / name).is_symlink():
            raise ValueError("prior stop outcome exists; inspect it, never retry")
    body = read_pin(D / "REMOTE.py", BODY_PIN)
    read_pin(D / "SOURCE_REVIEW.json", REVIEW_PIN)
    read_pin(D.parent / "EXPORT_ACCEPTANCE.json", ACCEPTANCE_PIN)
    started, timed_out = datetime.now(UTC).isoformat(), False
    try:
        result = subprocess.run(
            [
                "ssh",
                "-T",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=10",
                "-o",
                "ConnectionAttempts=1",
                "beleriand",
                "python3 -E -B -",
            ],
            input=body,
            capture_output=True,
            timeout=60,
            check=False,
        )
        stdout, stderr, returncode = result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired as error:
        stdout, stderr, returncode = error.stdout or b"", error.stderr or b"", None
        timed_out = True
    finished = datetime.now(UTC).isoformat()
    write("STDOUT.raw.json", stdout)
    write("STDERR.raw.txt", stderr)
    transport = {
        "schema": "fg19.first-cell-export-exact-stop-root-transport.v1",
        "started_utc": started,
        "finished_utc": finished,
        "body_bytes": len(body),
        "body_sha256": digest(body),
        "source_review_bytes": REVIEW_PIN[0],
        "source_review_sha256": REVIEW_PIN[1],
        "ssh_returncode": returncode,
        "timed_out": timed_out,
        "stdout_bytes": len(stdout),
        "stdout_sha256": digest(stdout),
        "stderr_bytes": len(stderr),
        "stderr_sha256": digest(stderr),
        "persisted_before_parse": True,
        "retry_performed": False,
    }
    write("TRANSPORT.json", (json.dumps(transport, indent=2) + "\n").encode())
    if returncode != 0 or timed_out or stderr or len(stdout) > 128000:
        raise RuntimeError("exact raw stop outcome preserved; inspect, never retry")
    stopped = json.loads(stdout)
    expected_keys = {
        "schema",
        "captured_utc",
        "unit",
        "invocation_id",
        "acceptance_sha256",
        "before",
        "after",
        "populated_before",
        "populated_after",
        "stop_returncode",
        "stop_stdout",
        "stop_stderr",
        "scope",
    }
    if set(stopped) != expected_keys or (
        stopped["schema"],
        stopped["unit"],
        stopped["invocation_id"],
        stopped["acceptance_sha256"],
    ) != ("fg19-first-cell-export-exact-owned-stop.v1", UNIT, INVOCATION, ACCEPTANCE_PIN[1]):
        raise ValueError("exact stop receipt shape/identity differs; raw preserved")
    before, after = stopped["before"], stopped["after"]
    if tuple(
        before.get(k)
        for k in (
            "Id",
            "InvocationID",
            "ActiveState",
            "SubState",
            "MainPID",
            "ExecMainCode",
            "ExecMainStatus",
            "Result",
        )
    ) != (UNIT, INVOCATION, "active", "exited", "0", "1", "0", "success"):
        raise ValueError("exact stop before-state differs")
    if (
        tuple(
            after.get(k)
            for k in ("Id", "LoadState", "ActiveState", "SubState", "MainPID", "InvocationID")
        )
        != (UNIT, "not-found", "inactive", "dead", "0", "")
        or type(stopped["stop_returncode"]) is not int
        or stopped["stop_returncode"] != 0
        or stopped["populated_before"]
        or stopped["populated_after"]
        or stopped["stop_stdout"]
        or stopped["stop_stderr"]
    ):
        raise ValueError("exact stop after-state differs; raw preserved")
    write("EXACT_STOP.json", stdout)
    print(
        json.dumps(
            {
                "status": "EXACT_OWNED_SUCCESSFUL_EXPORT_STOP_CAPTURED",
                "bytes": len(stdout),
                "sha256": digest(stdout),
                "transport": transport,
            }
        )
    )


if __name__ == "__main__":
    main()
