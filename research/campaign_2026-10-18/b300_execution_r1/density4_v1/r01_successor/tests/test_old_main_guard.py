"""CPU tests for the successor's exact stopped-controller guard."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

CONTROL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CONTROL))

import successor_runner  # noqa: E402
from proc_snapshot import ProcessIdentity, ProcSnapshot  # noqa: E402


def _authority_and_snapshot(*, state: str = "T") -> tuple[dict[str, object], ProcSnapshot]:
    pid = 321
    start_ticks = 654
    cgroup = "/user.slice/user-1.slice/tabdata-r01-execution-v4.service"
    packet_root = "/data/epanov_s/tabdata/runtime/b300-r01-v1"
    marker = str(Path(packet_root) / "execute.py")
    cmdline_sha256 = "a" * 64
    snapshot = ProcSnapshot(
        identity=ProcessIdentity(pid, start_ticks, pid, 1, state),
        argv=("python", marker),
        cmdline_sha256=cmdline_sha256,
        cgroup_path=cgroup,
        environment={},
        cwd="/data/epanov_s/tabdata",
    )
    authority: dict[str, object] = {
        "old_main_pid": pid,
        "old_main_start_ticks": start_ticks,
        "old_unit": "tabdata-r01-execution-v4",
        "old_main_cgroup_path": cgroup,
        "main_cmdline_sha256": cmdline_sha256,
        "packet_root": packet_root,
    }
    return authority, snapshot


def test_old_main_guard_accepts_only_exact_stopped_controller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority, snapshot = _authority_and_snapshot()
    monkeypatch.setattr(successor_runner, "read_process", lambda _pid: snapshot)

    successor_runner._require_old_main_stopped(authority)


@pytest.mark.parametrize("state", ["R", "S", "D"])
def test_old_main_guard_rejects_running_controller(
    monkeypatch: pytest.MonkeyPatch, state: str
) -> None:
    authority, snapshot = _authority_and_snapshot(state=state)
    monkeypatch.setattr(successor_runner, "read_process", lambda _pid: snapshot)

    with pytest.raises(ValueError, match="must be stopped"):
        successor_runner._require_old_main_stopped(authority)


@pytest.mark.parametrize("field", ["cgroup_path", "cmdline_sha256"])
def test_old_main_guard_rejects_identity_drift(monkeypatch: pytest.MonkeyPatch, field: str) -> None:
    authority, snapshot = _authority_and_snapshot()
    changed = dict(snapshot.__dict__)
    changed[field] = (
        "/other/user.slice/tabdata-r01-execution-v4.service" if field == "cgroup_path" else "b" * 64
    )
    drifted = ProcSnapshot(**changed)
    monkeypatch.setattr(successor_runner, "read_process", lambda _pid: drifted)

    with pytest.raises(ValueError, match="cgroup or command hash changed"):
        successor_runner._require_old_main_stopped(authority)
