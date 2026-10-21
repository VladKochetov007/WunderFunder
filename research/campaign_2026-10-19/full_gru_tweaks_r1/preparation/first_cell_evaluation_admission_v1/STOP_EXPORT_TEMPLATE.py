from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

UNIT = "computer-vision-homework-fg19-first-cell-export-v1.service"
INVOCATION = "UNBOUND_ACTUAL_EXPORT_INVOCATION"
ACCEPTANCE_BYTES = 0
ACCEPTANCE_SHA256 = "UNBOUND_ROOT_EXPORT_ACCEPTANCE"
ACCEPTANCE_BASE64 = "UNBOUND_ROOT_EXPORT_ACCEPTANCE_BASE64"


def show(unit: str) -> dict[str, str]:
    """Inspect only the new exact export unit or its campaign resource slice."""
    if unit not in (UNIT, "wnn-campaign.slice"):
        raise ValueError("unit outside exact export stop scope")
    result = subprocess.run(
        [
            "systemctl",
            "--user",
            "show",
            "--all",
            unit,
            "-p",
            "Id",
            "-p",
            "LoadState",
            "-p",
            "ActiveState",
            "-p",
            "SubState",
            "-p",
            "MainPID",
            "-p",
            "InvocationID",
            "-p",
            "ExecMainCode",
            "-p",
            "ExecMainStatus",
            "-p",
            "Result",
            "-p",
            "ControlGroup",
        ],
        capture_output=True,
        text=True,
        timeout=12,
        check=False,
    )
    if result.returncode:
        raise RuntimeError("exact stop unit query failed")
    return dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)


def populated() -> dict[str, str]:
    """Bound the parent descendant inventory and reject any populated process."""
    parent = show("wnn-campaign.slice")
    group = parent.get("ControlGroup", "")
    if group in ("", "/") or not group.endswith("/wnn-campaign.slice"):
        raise ValueError("unscoped parent cgroup")
    root = Path("/sys/fs/cgroup") / group.lstrip("/")
    if root.resolve(strict=True) != root:
        raise ValueError("parent cgroup redirected")
    stack, visited, total = [root], 0, 0
    result = {}
    while stack:
        current = stack.pop()
        visited += 1
        if visited > 128:
            raise ValueError("cgroup directory cap exceeded")
        with os.scandir(current) as entries:
            for index, entry in enumerate(entries):
                if index >= 256:
                    raise ValueError("cgroup per-directory entry cap exceeded")
                if entry.is_symlink():
                    raise ValueError("cgroup directory contains symlink")
                if entry.is_dir(follow_symlinks=False):
                    stack.append(Path(entry.path))
                    if visited + len(stack) > 128:
                        raise ValueError("cgroup directory cap exceeded")
        fd = os.open(current / "cgroup.procs", os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        with os.fdopen(fd, "rb") as stream:
            raw = stream.read(32769)
        total += len(raw)
        if len(raw) > 32768 or total > 262144:
            raise ValueError("cgroup process bytes exceeded cap")
        if raw.strip():
            result[str(current)] = raw.decode("ascii")
    return result


def main() -> None:
    """Stop exactly one successful exited export only after ROOT acceptance is durable."""
    if len(INVOCATION) != 32 or any(c not in "0123456789abcdef" for c in INVOCATION):
        raise ValueError("actual export invocation unbound")
    raw = base64.b64decode(ACCEPTANCE_BASE64, validate=True)
    if (
        type(ACCEPTANCE_BYTES) is not int
        or ACCEPTANCE_BYTES <= 0
        or (len(raw), hashlib.sha256(raw).hexdigest()) != (ACCEPTANCE_BYTES, ACCEPTANCE_SHA256)
    ):
        raise ValueError("exact ROOT acceptance byte binding differs")
    acceptance = json.loads(raw)
    if (
        acceptance.get("schema"),
        acceptance.get("status"),
        acceptance.get("unit"),
        acceptance.get("invocation_id"),
        acceptance.get("quality_result"),
    ) != (
        "fg19.first-cell-export-acceptance.v1",
        "ACCEPTED_FIXED_FIVE_PREDICTION_EXPORTS_NOT_SCORED",
        UNIT,
        INVOCATION,
        None,
    ):
        raise ValueError("ROOT acceptance semantics differ")
    before = show(UNIT)
    if (
        before.get("Id"),
        before.get("InvocationID"),
        before.get("ActiveState"),
        before.get("SubState"),
        before.get("MainPID"),
        before.get("ExecMainCode"),
        before.get("ExecMainStatus"),
        before.get("Result"),
    ) != (UNIT, INVOCATION, "active", "exited", "0", "1", "0", "success"):
        raise ValueError("exact successful export is not exited before stop")
    populated_before = populated()
    if populated_before:
        raise ValueError("campaign has populated descendants before stop")
    stop = subprocess.run(
        ["systemctl", "--user", "stop", UNIT],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    after = show(UNIT)
    populated_after = populated()
    if (
        stop.returncode != 0
        or populated_after
        or (
            after.get("LoadState"),
            after.get("ActiveState"),
            after.get("SubState"),
            after.get("MainPID"),
            after.get("InvocationID"),
        )
        != ("not-found", "inactive", "dead", "0", "")
    ):
        raise RuntimeError("exact stop outcome requires inspection; never retry")
    print(
        json.dumps(
            {
                "schema": "fg19-first-cell-export-exact-owned-stop.v1",
                "captured_utc": datetime.now(UTC).isoformat(),
                "unit": UNIT,
                "invocation_id": INVOCATION,
                "acceptance_sha256": ACCEPTANCE_SHA256,
                "before": before,
                "after": after,
                "populated_before": populated_before,
                "populated_after": populated_after,
                "stop_returncode": stop.returncode,
                "stop_stdout": stop.stdout,
                "stop_stderr": stop.stderr,
                "scope": "Only the one successful exited export unit. All old native/support and foreign units excluded; no file or payload mutation.",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
