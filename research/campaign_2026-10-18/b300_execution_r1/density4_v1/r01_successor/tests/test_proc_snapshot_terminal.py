"""Terminal process metadata races must not hide or invent live process state."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

CONTROL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CONTROL))

import proc_snapshot  # noqa: E402
from proc_snapshot import unit_members  # noqa: E402


def _fake_process(proc_root: Path, *, state: str, pid: int = 1234) -> Path:
    process = proc_root / str(pid)
    process.mkdir(parents=True)
    fields = ["0"] * 20
    fields[0] = state
    fields[1] = "1"
    fields[2] = str(pid)
    fields[19] = "7654321"
    (process / "stat").write_text(f"{pid} (fake-worker) {' '.join(fields)}")
    (process / "cgroup").write_text("0::/user.slice/tabdata-r01-execution-v4.service\n")
    (process / "cmdline").write_bytes(b"python\0worker.py\0")
    (process / "environ").write_bytes(b"THREADS=2\0")
    return process


def test_initial_zombie_returns_identity_with_explicit_empty_metadata(tmp_path: Path) -> None:
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    process = _fake_process(proc_root, state="Z")
    (process / "cmdline").unlink()
    (process / "environ").unlink()

    snapshot = proc_snapshot.read_process(1234, proc_root)

    assert snapshot.identity.state == "Z"
    assert snapshot.identity.start_ticks == 7654321
    assert snapshot.identity.process_group == 1234
    assert snapshot.argv == ()
    assert snapshot.environment == {}
    assert snapshot.cwd == ""
    assert snapshot.cgroup_path == "/user.slice/tabdata-r01-execution-v4.service"
    assert unit_members("tabdata-r01-execution-v4", proc_root) == (snapshot,)


def test_permission_race_downgrades_only_after_same_process_becomes_zombie(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    process = _fake_process(proc_root, state="S")
    original_read_bytes = Path.read_bytes

    def read_bytes(path: Path) -> bytes:
        if path.name == "environ":
            (process / "stat").write_text(
                f"1234 (fake-worker) {' '.join(_stat_fields_for_test('Z', 1234))}"
            )
            raise PermissionError("zombie environ is not readable")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)

    snapshot = proc_snapshot.read_process(1234, proc_root)

    assert snapshot.identity.state == "Z"
    assert snapshot.argv == ()
    assert snapshot.environment == {}
    assert snapshot.cwd == ""


def test_permission_error_on_still_live_process_is_not_suppressed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    _fake_process(proc_root, state="S")
    original_read_bytes = Path.read_bytes

    def read_bytes(path: Path) -> bytes:
        if path.name == "environ":
            raise PermissionError("live environ is not readable")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)

    with pytest.raises(PermissionError, match="live environ is not readable"):
        proc_snapshot.read_process(1234, proc_root)


def _stat_fields_for_test(state: str, pid: int) -> list[str]:
    fields = ["0"] * 20
    fields[0] = state
    fields[1] = "1"
    fields[2] = str(pid)
    fields[19] = "7654321"
    return fields
