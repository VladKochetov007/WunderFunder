from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

pytest.importorskip("fcntl", reason="controller uses Linux systemd/process APIs")
CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))

import feature_controller as controller  # noqa: E402


def test_resource_snapshot_is_authoritative_only_after_good_receipt_is_durable(
    tmp_path: Path, monkeypatch
) -> None:
    resource = {"verified": True, "time_unix": 123.0}
    last_good, receipt_sha = controller.persist_resource_snapshot(tmp_path, resource)
    assert last_good == resource
    assert receipt_sha == controller.sha(tmp_path / "RESOURCE_GOOD.json")

    original_write = controller.write

    def fail_good_write(path: Path, value: dict, **kwargs) -> None:
        if path.name == "RESOURCE_GOOD.json":
            raise OSError("injected disk failure")
        original_write(path, value, **kwargs)

    monkeypatch.setattr(controller, "write", fail_good_write)
    with pytest.raises(OSError, match="injected disk failure"):
        controller.persist_resource_snapshot(tmp_path, {"verified": True, "time_unix": 124.0})


def test_unverified_resource_snapshot_never_returns_an_admission_receipt(tmp_path: Path) -> None:
    resource, receipt_sha = controller.persist_resource_snapshot(
        tmp_path, {"verified": False, "errors": ["missing counter"]}
    )
    assert resource is None
    assert receipt_sha is None
    assert (tmp_path / "RESOURCE.json").is_file()
    assert not (tmp_path / "RESOURCE_GOOD.json").exists()


def test_directory_fsync_failure_does_not_authorize_the_resource_snapshot(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        controller,
        "fsync_directory",
        lambda path: (_ for _ in ()).throw(OSError("injected directory fsync failure")),
    )
    with pytest.raises(OSError, match="directory fsync failure"):
        controller.persist_resource_snapshot(tmp_path, {"verified": True, "time_unix": 123.0})


def test_grade_launch_failure_is_recorded_without_unwinding_active_fits(
    tmp_path: Path, monkeypatch
) -> None:
    active_fits = {"healthy-cell": object()}
    failed: dict[str, str] = {}
    attempted: set[str] = set()
    before = dict(active_fits)
    monkeypatch.setattr(
        controller, "spawn", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("no fd"))
    )
    monkeypatch.setattr(controller, "event", lambda *args, **kwargs: None)

    result = controller.spawn_grade(
        ["grade"], [160, 162], tmp_path / "grade.log", {}, "pair-f0-s1", tmp_path, failed, attempted
    )
    assert result is None
    assert active_fits == before
    assert "grade:pair-f0-s1" in failed
    assert attempted == {"pair-f0-s1"}


def test_spawn_closes_log_when_process_creation_fails(tmp_path: Path, monkeypatch) -> None:
    handle = io.StringIO()
    monkeypatch.setattr(Path, "open", lambda *args, **kwargs: handle)
    monkeypatch.setattr(
        controller.subprocess,
        "Popen",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("injected spawn failure")),
    )
    with pytest.raises(OSError, match="injected spawn failure"):
        controller.spawn(["worker"], [80, 82], tmp_path / "worker.log", {})
    assert handle.closed
