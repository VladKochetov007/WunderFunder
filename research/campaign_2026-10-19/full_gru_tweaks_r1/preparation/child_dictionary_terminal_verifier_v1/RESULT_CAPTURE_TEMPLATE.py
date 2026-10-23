"""Bounded read-only capture for the dictionary terminal-verifier support unit.

Source adaptation: records/child_ema_fit_v1/terminal_verification_v3/capture/REMOTE.py
(2,890 bytes, SHA-256 b38f2bece666bcea634dc74776e88d3da6b364b55a3870469385fed43acc62a2).
Only capture identity and the guarded code closure are substituted here. This
file is inert locally and deliberately refuses to run until root binds the
actual support admission/launch receipt sizes, hashes, and invocation ID.
"""

from __future__ import annotations

import base64
import datetime as dt
import hashlib
import json
import os
import re
import stat
import subprocess
from pathlib import Path

B = Path("/home/grinchenkoeg/computer-vision-homework-20261018")
D = B / "research_tmp/fg19_child_dictionary_verify_v1"
SUPPORT_UNIT = "computer-vision-homework-fg19-child-dictionary-verify-v1.service"
NATIVE_UNIT = "computer-vision-homework-fg19-child-dictionary-fit-v1.service"
NATIVE_INVOCATION_ID = "73b10b699e034a1b9952b2377b845095"

# Root must fill these only from the actual support admission/launch receipts.
# They intentionally fail closed while this local template remains unbound.
SUPPORT_ADMISSION_BYTES = None
SUPPORT_ADMISSION_SHA256 = "UNBOUND_ROOT_AFTER_ACTUAL_SUPPORT_ADMISSION"
SUPPORT_LAUNCH_BYTES = None
SUPPORT_LAUNCH_SHA256 = "UNBOUND_ROOT_AFTER_ACTUAL_SUPPORT_ADMISSION"
SUPPORT_INVOCATION_ID = "UNBOUND_ROOT_AFTER_ACTUAL_SUPPORT_ADMISSION"

MAX_CAPTURE_FILE_BYTES = 1_000_000
MAX_CAPTURE_AGGREGATE_BYTES = 4_000_000
CGROUP_ROOT = Path("/sys/fs/cgroup")

CODE_PINS = {
    "child_dictionary_terminal_check_v1.py": (
        34_164,
        "d06818648362d571e3b06637bc026b97190476a6b59474f4a57ec84c349e4d82",
    ),
    "child_control_terminal_check_v3.py": (
        79_972,
        "618b9b0fe7c0cf85e94848b24a837a63e27df46f81e35f2f8536c049072e5167",
    ),
    "parent_fit_terminal_check_v3.py": (
        64_550,
        "f2b819c1e105171f0093fae433aec301347645f8f64717effa16917005dbfe41",
    ),
    "propensity_terminal_support.py": (
        13_107,
        "958d973b17a2180c954ca7e0953ce20d7b5a3120124cd3cf964a97606604f409",
    ),
    "teacher_s5_terminal_check_v1.py": (
        37_448,
        "cb7b482cc0821421b3c62f87fea1724b4148e8e85f468ae1d65eb6ccf5db41a5",
    ),
    "child_ema_terminal_check_v3.py": (
        38_267,
        "676d548c1cd25c3c1379548999cca1300ef135dd41b4a8f73689ef8e59de3b5d",
    ),
}

UNIT_PROPERTIES = (
    "LoadState",
    "ActiveState",
    "SubState",
    "MainPID",
    "InvocationID",
    "Result",
    "ExecMainCode",
    "ExecMainStatus",
    "ControlGroup",
    "CPUUsageNSec",
    "MemoryCurrent",
    "MemoryPeak",
    "MemoryHigh",
    "MemoryMax",
    "MemorySwapMax",
    "CPUQuotaPerSecUSec",
    "ExecMainStartTimestamp",
    "ExecMainExitTimestamp",
    "ExecMainStartTimestampMonotonic",
    "ExecMainExitTimestampMonotonic",
)


def _require_bound_pins() -> None:
    """Prevent any remote query until root binds the actual support receipts."""
    for label, byte_count, digest in (
        ("admission", SUPPORT_ADMISSION_BYTES, SUPPORT_ADMISSION_SHA256),
        ("launch", SUPPORT_LAUNCH_BYTES, SUPPORT_LAUNCH_SHA256),
    ):
        if type(byte_count) is not int or byte_count <= 0:
            raise RuntimeError(f"root has not bound exact support {label} bytes")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise RuntimeError(f"root has not bound exact support {label} SHA-256")
    if re.fullmatch(r"[0-9a-f]{32}", SUPPORT_INVOCATION_ID) is None:
        raise RuntimeError("root has not bound the actual support invocation ID")


def _read_regular(path: Path, byte_cap: int) -> bytes:
    """Read one bounded regular file without following a final symlink."""
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > byte_cap:
        raise ValueError(f"unsafe or oversized capture input: {path}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) != identity:
            raise ValueError(f"capture input changed before read: {path}")
        parts: list[bytes] = []
        remaining = byte_cap + 1
        while remaining:
            chunk = os.read(descriptor, min(65_536, remaining))
            if not chunk:
                break
            parts.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(parts)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if len(raw) != before.st_size or len(raw) > byte_cap:
        raise ValueError(f"capture input changed size during read: {path}")
    if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != identity:
        raise ValueError(f"capture input changed during read: {path}")
    return raw


def _read_pinned_receipt(name: str, byte_count: int, digest: str) -> bytes:
    raw = _read_regular(D / name, MAX_CAPTURE_FILE_BYTES)
    if len(raw) != byte_count or hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError(f"support receipt bytes/SHA-256 mismatch: {name}")
    return raw


def _show(unit: str) -> dict[str, str]:
    command = ["systemctl", "--user", "show", "--all", unit]
    for key in UNIT_PROPERTIES:
        command.extend(("-p", key))
    result = subprocess.run(command, capture_output=True, text=True, timeout=10)
    if result.returncode != 0:
        raise RuntimeError(f"systemctl show failed for exact unit {unit}")
    return dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)


def _checked_cgroup(control_group: str) -> Path:
    if not control_group.startswith("/") or ".." in Path(control_group).parts:
        raise ValueError("invalid systemd ControlGroup path")
    group = CGROUP_ROOT / control_group.lstrip("/")
    if group == CGROUP_ROOT or not group.is_dir():
        raise ValueError("systemd ControlGroup is absent or resolves to cgroup root")
    group.relative_to(CGROUP_ROOT)
    return group


def _cgroup_populated(group: Path) -> dict[str, str]:
    paths = [group / "cgroup.procs"]
    for path in group.rglob("cgroup.procs"):
        if path != paths[0]:
            paths.append(path)
            if len(paths) >= 128:
                raise ValueError("parent cgroup process-file scan exceeded bound")
    result: dict[str, str] = {}
    for path in paths:
        if path.is_file():
            result[str(path)] = path.read_text().strip()
    return result


def _cgroup_text(group: Path, names: tuple[str, ...]) -> dict[str, str]:
    values: dict[str, str] = {}
    for name in names:
        path = group / name
        if path.exists():
            value = path.read_text()
            if len(value) > 4096:
                raise ValueError(f"oversized cgroup metadata: {name}")
            values[name] = value
    return values


def _capture_file(files: dict[str, dict[str, object]], name: str) -> None:
    raw = _read_regular(D / name, MAX_CAPTURE_FILE_BYTES)
    files[name] = {
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "base64": base64.b64encode(raw).decode("ascii"),
    }


def main() -> None:
    _require_bound_pins()
    directory_stat = D.lstat()
    if not stat.S_ISDIR(directory_stat.st_mode):
        raise ValueError("dictionary verifier output is not a real directory")

    admission_raw = _read_pinned_receipt(
        "VERIFIER_ADMISSION.json", SUPPORT_ADMISSION_BYTES, SUPPORT_ADMISSION_SHA256
    )
    launch_raw = _read_pinned_receipt(
        "VERIFIER_LAUNCH.json", SUPPORT_LAUNCH_BYTES, SUPPORT_LAUNCH_SHA256
    )
    admission = json.loads(admission_raw)
    launch = json.loads(launch_raw)
    if not isinstance(admission, dict) or not isinstance(launch, dict):
        raise ValueError("support receipts must be JSON objects")
    if (
        admission.get("schema") != "fg19.child-dictionary-terminal-verifier-admission.v1"
        or admission.get("status") != "FRESH_GATES_PASSED_SUPPORT_SUBMISSION_PENDING"
        or admission.get("run_name") != "fg19_child_dictionary_f0_s1001_fit_v1"
        or admission.get("native_unit") != NATIVE_UNIT
        or admission.get("native_invocation_id") != NATIVE_INVOCATION_ID
    ):
        raise ValueError("support admission identity/schema mismatch")
    if (
        launch.get("schema") != "fg19.child-dictionary-terminal-verifier-launch.v1"
        or launch.get("unit") != SUPPORT_UNIT
        or launch.get("invocation_id") != SUPPORT_INVOCATION_ID
        or launch.get("native_unit") != NATIVE_UNIT
        or launch.get("native_invocation_id") != NATIVE_INVOCATION_ID
    ):
        raise ValueError("support launch identity/schema mismatch")

    unit = _show(SUPPORT_UNIT)
    if unit.get("InvocationID") != SUPPORT_INVOCATION_ID:
        raise ValueError("live support unit invocation differs from pinned launch")
    parent = _show("wnn-campaign.slice")
    parent_control_group = parent.get("ControlGroup", "")
    if not parent_control_group.endswith("/wnn-campaign.slice"):
        raise ValueError("campaign parent slice ControlGroup mismatch")
    parent_group = _checked_cgroup(parent_control_group)
    populated = _cgroup_populated(parent_group)
    parent_events = _cgroup_text(
        parent_group, ("memory.current", "memory.events", "memory.events.local")
    )

    own = None
    unit_control_group = unit.get("ControlGroup", "")
    if unit_control_group not in ("", "/"):
        if not unit_control_group.startswith(parent_control_group.rstrip("/") + "/"):
            raise ValueError("support unit cgroup is not inside the campaign slice")
        own_group = _checked_cgroup(unit_control_group)
        own = _cgroup_text(
            own_group,
            ("memory.current", "memory.peak", "memory.events", "memory.events.local"),
        )
    # An unavailable own cgroup remains null; there is deliberately no host-root fallback.

    code_pins: dict[str, dict[str, object]] = {}
    for name, (expected_bytes, expected_sha256) in CODE_PINS.items():
        raw = _read_regular(D / name, MAX_CAPTURE_FILE_BYTES)
        actual_sha256 = hashlib.sha256(raw).hexdigest()
        if len(raw) != expected_bytes or actual_sha256 != expected_sha256:
            raise ValueError(f"private checker closure byte/SHA mismatch: {name}")
        code_pins[name] = {"bytes": len(raw), "sha256": actual_sha256}

    files: dict[str, dict[str, object]] = {}
    for name in ("VERIFIER_ADMISSION.json", "VERIFIER_LAUNCH.json", "unit.log"):
        _capture_file(files, name)

    terminal_success = (
        unit.get("ActiveState"),
        unit.get("SubState"),
        unit.get("MainPID"),
        unit.get("ExecMainCode"),
        unit.get("ExecMainStatus"),
    ) == ("active", "exited", "0", "1", "0")
    if terminal_success:
        _capture_file(files, "VERIFICATION.json")
    if sum(int(item["bytes"]) for item in files.values()) > MAX_CAPTURE_AGGREGATE_BYTES:
        raise ValueError("raw capture aggregate exceeds 4 MB")

    response = {
        "utc": dt.datetime.now(dt.UTC).isoformat(),
        "unit": unit,
        "support_unit": SUPPORT_UNIT,
        "support_invocation_id": SUPPORT_INVOCATION_ID,
        "terminal_success": terminal_success,
        "parent": parent,
        "parent_events": parent_events,
        "own_files": own,
        "populated": populated,
        "files": files,
        "code_pins": code_pins,
        "scope": "Bounded dictionary terminal-verifier metadata/log capture only; six exact private code pins and exact root-bound support receipts; no model, state, source, dictionary, cache-array, dataset, or trainer payload read/hash; empty own cgroup is unavailable and never host root.",
    }
    print(json.dumps(response, sort_keys=True))


if __name__ == "__main__":
    main()
