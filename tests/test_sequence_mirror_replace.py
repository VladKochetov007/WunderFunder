from __future__ import annotations

import errno
import json
from pathlib import Path

import pytest

from scripts import sequence_monitor_mirror_io as mirror
from scripts.collect_sequence_monitor_r2 import collect_runs


def _windows_lock() -> PermissionError:
    error = PermissionError(errno.EACCES, "fixture Windows reader lock")
    error.winerror = 5
    return error


def test_transient_windows_lock_retries_same_atomic_state_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "state.json"
    path.write_bytes(b'{"old":true}\n')
    replace = mirror.os.replace
    calls, sleeps = [], []

    def locked(source: Path, destination: Path) -> None:
        calls.append((source, destination, source.read_bytes()))
        if len(calls) < 3:
            assert path.read_bytes() == b'{"old":true}\n'
            raise _windows_lock()
        replace(source, destination)

    monkeypatch.setattr(mirror.os, "replace", locked)
    monkeypatch.setattr(mirror.time, "sleep", sleeps.append)
    mirror.atomic_json(path, {"cursors": {"run/file": {"offset": 7, "marker": "abcd"}}})
    assert len(calls) == 3 and len(sleeps) == 2
    assert len({source for source, _, _ in calls}) == 1
    assert len({payload for _, _, payload in calls}) == 1
    assert json.loads(path.read_bytes())["cursors"]["run/file"] == {"offset": 7, "marker": "abcd"}
    assert not list(tmp_path.glob(".state.json.*"))


def test_permanent_windows_lock_raises_within_one_second_and_keeps_old_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "state.json"
    path.write_bytes(b"original-state")
    clock, attempts = [0.0], []

    def locked(source: Path, destination: Path) -> None:
        attempts.append((source, destination))
        raise _windows_lock()

    monkeypatch.setattr(mirror.os, "replace", locked)
    monkeypatch.setattr(mirror.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        mirror.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds)
    )
    with pytest.raises(PermissionError):
        mirror.atomic_bytes(path, b"new-state")
    assert clock[0] == 1.0 and len(attempts) > 1
    assert path.read_bytes() == b"original-state"
    assert not list(tmp_path.glob(".state.json.*"))


def test_other_io_errors_and_symlink_targets_are_not_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "state.json"
    path.write_bytes(b"preserved")
    calls = []

    def denied(source: Path, destination: Path) -> None:
        calls.append(destination)
        raise PermissionError(errno.EACCES, "fixture ordinary permission refusal")

    monkeypatch.setattr(mirror.os, "replace", denied)
    monkeypatch.setattr(mirror.time, "sleep", lambda _: pytest.fail("unexpected retry"))
    with pytest.raises(PermissionError):
        mirror.atomic_bytes(path, b"replacement")
    assert calls == [path] and path.read_bytes() == b"preserved"
    actual_is_symlink = Path.is_symlink
    monkeypatch.setattr(
        Path, "is_symlink", lambda candidate: candidate == path or actual_is_symlink(candidate)
    )
    with pytest.raises(ValueError, match="symlink"):
        mirror.atomic_bytes(path, b"replacement")
    assert calls == [path] and path.read_bytes() == b"preserved"


def test_corrupt_snapshot_still_refuses_publication_and_cursor_advance(tmp_path: Path) -> None:
    run = tmp_path / "remote/cell/fit"
    run.mkdir(parents=True)
    (run / "run_metadata.json").write_text('{"version":1}')
    snapshot = collect_runs(str(tmp_path / "remote"), "l4")
    native, view = tmp_path / "native", tmp_path / "view"
    cursors = mirror.apply_snapshot(snapshot, view, native, {}, "l4")
    original = (native / "l4-cell-fit/run_metadata.json").read_bytes()
    (run / "run_metadata.json").write_text('{"version":2}')
    broken = collect_runs(str(tmp_path / "remote"), "l4", cursors)
    broken["runs"][0]["files"][0]["sha256"] = "0" * 64
    before = json.dumps(cursors, sort_keys=True)
    with pytest.raises(ValueError, match="digest mismatch"):
        mirror.apply_snapshot(broken, view, native, cursors, "l4")
    assert json.dumps(cursors, sort_keys=True) == before
    assert (native / "l4-cell-fit/run_metadata.json").read_bytes() == original
    assert (view / "l4-cell-fit/run_metadata.json").read_bytes() == original
