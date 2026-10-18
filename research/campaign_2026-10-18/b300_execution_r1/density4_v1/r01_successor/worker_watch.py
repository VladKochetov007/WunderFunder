"""PID-reuse-safe process-group observation for adopted and fresh fit workers."""

from __future__ import annotations

import os
import select
from pathlib import Path

from proc_snapshot import ProcSnapshot, _stat_fields, read_process


def pidfd_for(pid: int, start_ticks: int) -> int | None:
    """Open a pollable reference only after confirming the PID/start-tick pair."""
    if not hasattr(os, "pidfd_open"):
        return None
    current = read_process(pid)
    if current.identity.start_ticks != start_ticks:
        raise ProcessLookupError(f"PID {pid} was reused before pidfd acquisition")
    return os.pidfd_open(pid, 0)


def pidfd_exited(pidfd: int) -> bool:
    """Return whether the pinned process identity has exited."""
    poller = select.poll()
    poller.register(pidfd, select.POLLIN | select.POLLHUP | select.POLLERR)
    return bool(poller.poll(0))


def process_group_members(pgid: int, proc_root: Path = Path("/proc")) -> tuple[ProcSnapshot, ...]:
    """Read only members whose kernel process-group ID matches the requested group."""
    members = []
    for process in proc_root.iterdir():
        if not process.name.isdigit():
            continue
        pid = int(process.name)
        try:
            stat = _stat_fields(process / "stat")
            if stat[2] == pgid:
                members.append(read_process(pid, proc_root))
        except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
            continue
    return tuple(members)


def group_has_live_members(pgid: int, proc_root: Path = Path("/proc")) -> bool:
    """Treat an all-zombie group as terminal only after every live member is gone."""
    members = process_group_members(pgid, proc_root)
    return any(item.identity.state not in {"Z", "X"} for item in members)


def require_same_process(
    pid: int, start_ticks: int, pgid: int, proc_root: Path = Path("/proc")
) -> ProcSnapshot:
    """Reject a process-group leader that disappeared or was replaced by PID reuse."""
    snapshot = read_process(pid, proc_root)
    if snapshot.identity.start_ticks != start_ticks or snapshot.identity.process_group != pgid:
        raise ProcessLookupError(f"process identity changed for PID {pid}")
    return snapshot
