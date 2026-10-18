"""Stable Linux /proc identities for exact old-unit process reconciliation."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProcessIdentity:
    """Kernel identity fields that distinguish a PID from its later reuse."""

    pid: int
    start_ticks: int
    process_group: int
    parent_pid: int
    state: str


@dataclass(frozen=True)
class ProcSnapshot:
    """Read-only command and cgroup state for one Linux process."""

    identity: ProcessIdentity
    argv: tuple[str, ...]
    cmdline_sha256: str
    cgroup_path: str
    environment: dict[str, str]
    cwd: str


def _stat_fields(path: Path) -> tuple[int, int, int, str, int]:
    """Parse PID, PPID, process-group, state and start ticks from proc stat."""
    raw = path.read_text(encoding="ascii")
    close = raw.rfind(")")
    if close < 0:
        raise ValueError(f"malformed proc stat record: {path}")
    pid = int(raw[: raw.find(" ")])
    fields = raw[close + 2 :].split()
    if len(fields) < 20:
        raise ValueError(f"truncated proc stat record: {path}")
    return pid, int(fields[1]), int(fields[2]), fields[0], int(fields[19])


def _terminal_snapshot(process: Path, before: tuple[int, int, int, str, int]) -> ProcSnapshot:
    """Keep stable kernel identity when a zombie exposes no command metadata."""
    cgroup_path = ""
    try:
        cgroups = (process / "cgroup").read_text(encoding="ascii").splitlines()
        cgroup_path = next(
            (line.split(":", 2)[2] for line in cgroups if line.startswith("0::")), ""
        )
    except OSError:
        pass
    try:
        after = _stat_fields(process / "stat")
    except FileNotFoundError as error:
        raise ProcessLookupError(f"terminal process {before[0]} disappeared") from error
    if (
        before[0] != after[0]
        or before[4] != after[4]
        or before[2] != after[2]
        or after[3] not in {"Z", "X"}
    ):
        raise ProcessLookupError(f"terminal process identity changed for PID {before[0]}")
    identity = ProcessIdentity(before[0], before[4], before[2], after[1], after[3])
    return ProcSnapshot(
        identity,
        (),
        hashlib.sha256(b"").hexdigest(),
        cgroup_path,
        {},
        "",
    )


def _terminal_after_read_error(
    process: Path,
    before: tuple[int, int, int, str, int],
    error: OSError | ValueError,
) -> ProcSnapshot:
    """Downgrade a metadata read failure only if the same PID is now terminal."""
    try:
        after = _stat_fields(process / "stat")
    except FileNotFoundError:
        raise ProcessLookupError(f"process {before[0]} disappeared during metadata read") from error
    except OSError:
        raise error from None
    if before[0] != after[0] or before[4] != after[4] or before[2] != after[2]:
        raise ProcessLookupError(
            f"process identity changed while reading PID {before[0]}"
        ) from error
    if after[3] not in {"Z", "X"}:
        raise error
    return _terminal_snapshot(process, after)


def read_process(pid: int, proc_root: Path = Path("/proc")) -> ProcSnapshot:
    """Read one process twice, allowing state transitions but rejecting PID reuse."""
    process = proc_root / str(pid)
    before = _stat_fields(process / "stat")
    if before[3] in {"Z", "X"}:
        return _terminal_snapshot(process, before)
    try:
        raw_cmdline = (process / "cmdline").read_bytes()
        argv = tuple(
            token.decode("utf-8", errors="surrogateescape")
            for token in raw_cmdline.split(b"\0")
            if token
        )
        cgroups = (process / "cgroup").read_text(encoding="ascii").splitlines()
        unified = next((line.split(":", 2)[2] for line in cgroups if line.startswith("0::")), None)
        if unified is None:
            raise ValueError(f"process {pid} is not in a readable unified cgroup")
        raw_environment = (process / "environ").read_bytes()
        environment = {}
        for token in raw_environment.split(b"\0"):
            if b"=" in token:
                key, value = token.split(b"=", 1)
                environment[key.decode("ascii", errors="ignore")] = value.decode(
                    "utf-8", errors="surrogateescape"
                )
        cwd = os.readlink(process / "cwd")
        after = _stat_fields(process / "stat")
        cgroups_after = (process / "cgroup").read_text(encoding="ascii").splitlines()
        if (
            before[0] != after[0]
            or before[4] != after[4]
            or before[2] != after[2]
            or cgroups != cgroups_after
        ):
            raise ProcessLookupError(f"process identity changed while reading PID {pid}")
    except (OSError, ValueError) as error:
        return _terminal_after_read_error(process, before, error)
    identity = ProcessIdentity(pid, before[4], before[2], before[1], before[3])
    return ProcSnapshot(
        identity, argv, hashlib.sha256(raw_cmdline).hexdigest(), unified, environment, cwd
    )


def _unit_component(unit: str) -> str:
    """Normalize a systemd service name without allowing substring matches."""
    if not unit or "/" in unit or unit in {".", ".."}:
        raise ValueError("systemd unit must be one nonempty path component")
    return unit if unit.endswith(".service") else f"{unit}.service"


def unit_members(unit: str, proc_root: Path = Path("/proc")) -> tuple[ProcSnapshot, ...]:
    """Return exact-unit and nested-descendant cgroup members, excluding lookalikes."""
    component = _unit_component(unit)
    members: list[ProcSnapshot] = []
    for item in proc_root.iterdir():
        if not item.name.isdigit():
            continue
        try:
            snapshot = read_process(int(item.name), proc_root)
        except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
            continue
        if component in snapshot.cgroup_path.split("/"):
            members.append(snapshot)
    return tuple(members)


def grouped_members(snapshots: tuple[ProcSnapshot, ...]) -> dict[int, set[int]]:
    """Group unit members by kernel process-group ID."""
    groups: dict[int, set[int]] = {}
    for snapshot in snapshots:
        groups.setdefault(snapshot.identity.process_group, set()).add(snapshot.identity.pid)
    return groups


def require_controller_identity(
    snapshot: ProcSnapshot,
    *,
    expected_pid: int,
    expected_start_ticks: int,
    expected_unit: str,
    command_marker: str,
    must_be_stopped: bool,
) -> None:
    """Require the exact old controller before a freeze or after its SIGSTOP."""
    if (
        snapshot.identity.pid != expected_pid
        or snapshot.identity.start_ticks != expected_start_ticks
    ):
        raise ValueError("old controller PID/start-time identity changed")
    if _unit_component(expected_unit) not in snapshot.cgroup_path.split("/"):
        raise ValueError("old controller is outside the exact systemd unit cgroup")
    if not snapshot.argv or command_marker not in "\0".join(snapshot.argv):
        raise ValueError("old controller command line does not contain its authorized marker")
    stopped = snapshot.identity.state in {"T", "t"}
    if must_be_stopped != stopped:
        expectation = "stopped" if must_be_stopped else "running"
        raise ValueError(f"old controller must be {expectation} at this phase")
