from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from handoff_core import ProcIdentity, parse_proc_stat, sha256_bytes


@dataclass(frozen=True)
class ProcSnapshot:
    """Read-only identity and ownership view of one process in the old unit."""

    identity: ProcIdentity
    cgroup_path: str
    argv: tuple[str, ...]
    cmdline_sha256: str


def cgroup_path(pid: int, proc_root: Path = Path("/proc")) -> str:
    """Return one process's unified cgroup path or reject ambiguous cgroup data."""
    records = (proc_root / str(pid) / "cgroup").read_text().splitlines()
    unified = [line.partition("::")[2] for line in records if line.startswith("0::")]
    if len(unified) != 1 or not unified[0].startswith("/"):
        raise ValueError(f"PID {pid} has no unique cgroup-v2 path")
    return unified[0]


def _argv(pid: int, proc_root: Path) -> tuple[tuple[str, ...], bytes]:
    """Read command arguments without decoding arbitrary bytes lossily."""
    raw = (proc_root / str(pid) / "cmdline").read_bytes()
    return tuple(part.decode("utf-8", errors="replace") for part in raw.split(b"\0") if part), raw


def snapshot_pid(pid: int, proc_root: Path = Path("/proc")) -> ProcSnapshot:
    """Take a stable double-read snapshot for PID-reuse-safe reconciliation."""
    process_dir = proc_root / str(pid)
    first = parse_proc_stat((process_dir / "stat").read_text(), pid)
    argv, raw_cmdline = _argv(pid, proc_root)
    owner = cgroup_path(pid, proc_root)
    last = parse_proc_stat((process_dir / "stat").read_text(), pid)
    if first.start_ticks != last.start_ticks:
        raise ProcessLookupError(f"PID {pid} changed during /proc inspection")
    return ProcSnapshot(last, owner, argv, sha256_bytes(raw_cmdline))


def unit_members(unit_name: str, proc_root: Path = Path("/proc")) -> tuple[ProcSnapshot, ...]:
    """Enumerate current processes in one exact systemd unit, including zombies."""
    target = f"{unit_name}.service"
    snapshots: list[ProcSnapshot] = []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            item = snapshot_pid(int(entry.name), proc_root)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        path_parts = item.cgroup_path.rstrip("/").split("/")
        if path_parts[-1:] == [target]:
            snapshots.append(item)
    return tuple(sorted(snapshots, key=lambda item: item.identity.pid))


def parse_worker_args(argv: tuple[str, ...]) -> dict[str, str]:
    """Parse the exact PR15 worker CLI fields used to recover an unjournaled spawn."""
    script_indices = [i for i, token in enumerate(argv) if token.endswith("/worker.py")]
    if len(script_indices) != 1:
        raise ValueError("process is not one unique PR15 worker.py invocation")
    values: dict[str, str] = {}
    tokens = argv[script_indices[0] + 1 :]
    for index, token in enumerate(tokens[:-1]):
        if token.startswith("--"):
            if token in values:
                raise ValueError(f"duplicate worker argument: {token}")
            values[token] = tokens[index + 1]
    required = ("--root", "--name", "--fold", "--seed", "--control", "--gpu", "--cpus", "--work", "--mode")
    if set(values) != set(required) or values["--mode"] != "fit":
        raise ValueError("worker command is incomplete or is not a full fit")
    return values


def worker_job_id(snapshot: ProcSnapshot, unit_name: str) -> str | None:
    """Identify a live full-fit worker in the exact old unit; ignore all others."""
    if snapshot.cgroup_path.rstrip("/").split("/")[-1:] != [f"{unit_name}.service"]:
        return None
    if not snapshot.argv:
        return None
    try:
        return parse_worker_args(snapshot.argv)["--name"]
    except ValueError:
        return None


def require_controller_identity(
    snapshot: ProcSnapshot,
    *,
    expected_pid: int,
    expected_start_ticks: int,
    expected_unit: str,
    command_marker: str,
    must_be_stopped: bool = False,
) -> None:
    """Guard a future SIGSTOP/retirement action with exact PID, start and unit."""
    identity = snapshot.identity
    if identity.pid != expected_pid or identity.start_ticks != expected_start_ticks:
        raise ValueError("controller PID was reused or its start identity changed")
    if snapshot.cgroup_path.rstrip("/").split("/")[-1:] != [f"{expected_unit}.service"]:
        raise ValueError("controller PID is outside the frozen PR15 systemd unit")
    if not any(command_marker in token for token in snapshot.argv):
        raise ValueError("controller command line does not match the frozen controller")
    if must_be_stopped and identity.state not in {"T", "t"}:
        raise ValueError("old controller is not confirmed stopped")


def grouped_members(snapshots: tuple[ProcSnapshot, ...]) -> dict[int, tuple[int, ...]]:
    """Group a unit's PIDs by process group so no worker descendant is orphaned."""
    groups: dict[int, list[int]] = {}
    for snapshot in snapshots:
        groups.setdefault(snapshot.identity.process_group, []).append(snapshot.identity.pid)
    return {group: tuple(sorted(pids)) for group, pids in sorted(groups.items())}
