from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture
def collector(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Load the narrow operational helper without importing Torch or the trainer."""
    root = Path(__file__).resolve().parents[1]
    path = root / "research/campaign_2026-10-18/priority25_execution_r4/receipts/checks"
    monkeypatch.syspath_prepend(str(path))
    return importlib.import_module("collect_terminal_checkpoint")


def make_endpoint(base: Path) -> tuple[Path, Path]:
    """Provide the actual terminal JSON shapes with small opaque weight payloads."""
    run = base / "data/artifacts/runs/rr18_r01_wide_f0_s1001_e6"
    run.mkdir(parents=True)
    for name in ("model_ep5.pt", "model.pt", "training_state.pt"):
        (run / name).write_bytes(name.encode())
    provenance = {
        "cumulative_updates": {"optimizer_completed_updates": 89_520},
        "checkpoint_sha256": {
            name: hashlib.sha256((run / name).read_bytes()).hexdigest()
            for name in ("model_ep5.pt", "model.pt")
        },
        "resumed_from": {"path": "stage1/training_state.pt", "sha256": "parent", "next_epoch": 2},
    }
    (run / "provenance.json").write_text(json.dumps(provenance))
    (run / "history.json").write_text(json.dumps([{"epoch": i} for i in range(6)]))
    attempt = "computer-vision-homework-r01-wide-f0s1001-ep25-v1"
    attempt_path = base / "runtime/attempts" / f"{attempt}.json"
    attempt_path.parent.mkdir(parents=True)
    attempt_path.write_text(json.dumps({"id": attempt, "status": "finished", "exit_status": 0}))
    log = base / "runtime/logs" / f"{attempt}.log"
    log.parent.mkdir(parents=True)
    log.write_text("SELECTED_FIXED 5\n")
    return run, attempt_path


def test_collection_copies_terminal_outputs_and_evidence(
    collector: ModuleType, tmp_path: Path
) -> None:
    """All deliverables and execution evidence receive independently checked hashes."""
    make_endpoint(tmp_path)
    receipt = collector.archive_terminal(tmp_path, "wide", 3_000_000_000, "/test")
    assert set(receipt["artifacts"]) == set(collector.ARTIFACT_NAMES)
    for item in [
        *receipt["artifacts"].values(),
        receipt["attempt_evidence"],
        receipt["log_evidence"],
    ]:
        payload = Path(item["path"]).read_bytes()
        assert len(payload) == item["bytes"]
        assert hashlib.sha256(payload).hexdigest() == item["sha256"]
    assert receipt["source_stats_before"] == receipt["source_stats_after"]
    assert receipt["attempt_evidence"]["status"] == "finished"
    assert receipt["attempt_evidence"]["exit_status"] == 0


@pytest.mark.parametrize(
    "status,exit_status", [("running", None), ("finished", None), ("finished", 1)]
)
def test_non_success_never_creates_archive(
    collector: ModuleType, tmp_path: Path, status: str, exit_status: int | None
) -> None:
    """Absent exit status must not be silently interpreted as successful completion."""
    _, path = make_endpoint(tmp_path)
    value = json.loads(path.read_text())
    value["status"] = status
    value.pop("exit_status")
    if exit_status is not None:
        value["exit_status"] = exit_status
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="successfully terminal"):
        collector.archive_terminal(tmp_path, "wide", 3_000_000_000, "/test")
    assert not (tmp_path / "checkpoint_snapshots").exists()


def test_incomplete_finalization_does_not_archive(collector: ModuleType, tmp_path: Path) -> None:
    """Final history is required even if an attempt record claims success."""
    run, _ = make_endpoint(tmp_path)
    (run / "history.json").write_text(json.dumps([{"epoch": i} for i in range(5)]))
    with pytest.raises(ValueError, match="final history"):
        collector.archive_terminal(tmp_path, "wide", 3_000_000_000, "/test")
    assert not (tmp_path / "checkpoint_snapshots").exists()


def test_existing_archive_is_never_overwritten(collector: ModuleType, tmp_path: Path) -> None:
    """A second call must fail rather than replace previously preserved evidence."""
    make_endpoint(tmp_path)
    first = collector.archive_terminal(tmp_path, "wide", 3_000_000_000, "/test")
    with pytest.raises(FileExistsError):
        collector.archive_terminal(tmp_path, "wide", 3_000_000_000, "/test")
    path = Path(first["attempt_evidence"]["path"])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == first["attempt_evidence"]["sha256"]


def test_headroom_rejection_precedes_collection(collector: ModuleType, tmp_path: Path) -> None:
    """The active GPU worker reservation stays covered by the support-task gate."""
    make_endpoint(tmp_path)
    with pytest.raises(ValueError, match="memory headroom"):
        collector.archive_terminal(tmp_path, "wide", 9_000_000_000, "/test")
    assert not (tmp_path / "checkpoint_snapshots").exists()


def test_changed_dependency_rejected_before_writes(
    collector: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An edited dependency must not silently replace the reviewed collector bytes."""
    make_endpoint(tmp_path)
    changed = tmp_path / "changed_helper.py"
    changed.write_text("changed\n")
    monkeypatch.setattr(collector.checkpoint, "__file__", str(changed))
    with pytest.raises(ValueError, match="dependency hash"):
        collector.archive_terminal(tmp_path, "wide", 3_000_000_000, "/test")
    assert not (tmp_path / "checkpoint_snapshots").exists()


def test_source_change_during_copy_is_not_accepted(
    collector: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A late source rewrite invalidates collection even after an earlier hash matched."""
    run, _ = make_endpoint(tmp_path)
    original_copy = collector.shutil.copy2

    def copy_with_concurrent_rewrite(source: Path, destination: Path) -> str:
        copied = original_copy(source, destination)
        if source.name == "history.json":
            provenance = run / "provenance.json"
            provenance.write_text(provenance.read_text() + " ")
        return str(copied)

    monkeypatch.setattr(collector.shutil, "copy2", copy_with_concurrent_rewrite)
    with pytest.raises(ValueError, match="generation changed"):
        collector.archive_terminal(tmp_path, "wide", 3_000_000_000, "/test")
