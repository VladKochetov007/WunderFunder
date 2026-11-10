from __future__ import annotations

import json
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

import pytest

from wnn.training.sequence_monitor_progress import MonitorProgress, record_successful_update
from wnn.training.sequence_observer import SequenceObserver
from wnn.training.sequence_observer_resources import (
    ProcessReading,
    gpu_resources,
    read_process,
    tracked_processes,
)


def _process(pid: int, parent: int = 1, started: int = 100, state: str = "S") -> ProcessReading:
    return ProcessReading(pid, parent, started, state, 120, 4096)


def test_process_tree_binds_start_ticks_and_excludes_reused_pids() -> None:
    """Children remain observable after a leader exits; reused PIDs are never adopted."""
    identities = {10: 100}
    table = {10: _process(10), 11: _process(11, 10, 110), 12: _process(12, 11, 120)}
    assert set(tracked_processes(table, identities)) == {10, 11, 12}
    table = {10: _process(10, 1, 999), 11: _process(11, 1, 110), 13: _process(13, 10, 1000)}
    assert set(tracked_processes(table, identities)) == {11}
    assert 13 not in identities
    assert not _process(12, state="Z").is_alive


def test_proc_stat_parses_parenthesized_names_without_signals(tmp_path: Path) -> None:
    """Start-time and CPU fields retain Linux's field numbering despite comm spaces."""
    (tmp_path / "10").mkdir()
    fields = ["0"] * 22
    fields[0], fields[1], fields[11], fields[12], fields[19], fields[21] = (
        "S",
        "1",
        "7",
        "8",
        "100",
        "3",
    )
    (tmp_path / "10" / "stat").write_text("10 (python (worker)) " + " ".join(fields))
    with patch(
        "wnn.training.sequence_observer_resources.os.sysconf", return_value=4096, create=True
    ):
        reading = read_process(10, tmp_path)
    assert reading == ProcessReading(10, 1, 100, "S", 15, 12288)


def test_gpu_telemetry_covers_all_devices_and_preserves_foreign_pids() -> None:
    """A foreign GPU process is labelled and never signalled or adopted."""
    devices = "0, GPU-one, L4, 80, 100, 24000, 50, 65\n1, GPU-two, B300, 90, 200, 288000, 600, 70\n"
    applications = "GPU-one, 10, 100\nGPU-two, 999, 200\n"
    with patch("wnn.training.sequence_observer_resources.subprocess.run") as run:
        run.side_effect = [CompletedProcess([], 0, devices), CompletedProcess([], 0, applications)]
        metrics, context = gpu_resources({10})
    assert metrics["gpu/device_0/utilization_pct"] == 80
    assert metrics["gpu/device_1/memory_used_bytes"] == 200 * 1048576
    assert context["devices"][1]["uuid"] == "GPU-two"
    assert context["gpu_processes"][1] == {
        "gpu_uuid": "GPU-two",
        "pid": 999,
        "tracked": False,
        "memory_used_bytes": 200 * 1048576,
    }
    assert all("--id" not in call.args[0] for call in run.call_args_list)


def test_heartbeat_runs_while_training_does_not_advance(tmp_path: Path) -> None:
    """Independent snapshots advance heartbeat time while evaluation retains its phase."""
    with patch.dict("os.environ", {"WNN_MONITOR_HEARTBEAT_SECONDS": "1"}):
        progress = MonitorProgress(tmp_path)
    try:
        record_successful_update(type("Monitor", (), {"progress": progress})(), 7, 0.5, 1234)
        progress.phase("evaluation", 7, 1)
        before = json.loads(progress.path.read_text())
        threading.Event().wait(1.15)
        after = json.loads(progress.path.read_text())
        assert after["heartbeat_at"] > before["heartbeat_at"]
        assert after["phase"] == "evaluation"
        assert after["last_successful_update"] == 7
        assert after["last_successful_update_at"] == before["last_successful_update_at"]
    finally:
        progress.close()


def _old_train_event() -> dict[str, object]:
    return {
        "schema": "wnn.monitor.v1",
        "kind": "train_step",
        "step": 7,
        "epoch": 0.5,
        "wall_time": (datetime.now(UTC) - timedelta(minutes=10)).isoformat(),
        "metrics": {"train/loss": 0.9},
        "context": {},
    }


def test_observer_preserves_independent_liveness_during_long_evaluation(tmp_path: Path) -> None:
    """An old optimizer event plus real live processes remains live during evaluation."""
    output = tmp_path / "run-one"
    output.mkdir()
    (output / "run_metadata.json").write_text(json.dumps({"source": "fixture", "config": {}}))
    (output / "monitor_events.jsonl").write_text(json.dumps(_old_train_event()) + "\n")
    (output / "monitor_progress.json").write_text(
        json.dumps(
            {
                "pid": 11,
                "phase": "evaluation",
                "step": 7,
                "epoch": 1,
                "heartbeat_at": datetime.now(UTC).isoformat(),
                "last_successful_update": 7,
            }
        )
    )
    table = {10: _process(10), 11: _process(11, 10, 110)}
    with (
        patch("wnn.training.sequence_observer.read_process", return_value=table[10]),
        patch("wnn.training.sequence_observer.process_table", return_value=table),
        patch(
            "wnn.training.sequence_observer.gpu_resources",
            return_value=({"gpu/device_1/utilization_pct": 85}, {}),
        ),
        patch("wnn.training.sequence_observer.host_resources", return_value={}),
    ):
        observer = SequenceObserver(output, 10)
        event = observer.sample()
    assert event["metrics"]["observer/process_alive_count"] == 2
    assert event["metrics"]["observer/event_age_seconds"] >= 590
    assert event["context"]["phase"] == "evaluation"
    assert event["metrics"]["observer/last_successful_update"] == 7
    assert "gpu/device_1/utilization_pct" in event["metrics"]
    with (
        patch("wnn.training.sequence_observer.process_table", return_value={}),
        patch("wnn.training.sequence_observer.gpu_resources", return_value=({}, {})),
        patch("wnn.training.sequence_observer.host_resources", return_value={}),
    ):
        completed = observer.sample(returncode=0)
    assert completed["context"]["phase"] == "completed"
    assert completed["metrics"]["observer/process_alive_count"] == 0


def test_observer_does_not_create_trainer_output_or_infer_liveness_from_logs(
    tmp_path: Path,
) -> None:
    """Missing output is allowed during startup; an unreadable PID is rejected."""
    output = tmp_path / "not-created"
    with (
        patch("wnn.training.sequence_observer.read_process", return_value=_process(10)),
        patch("wnn.training.sequence_observer.process_table", return_value={10: _process(10)}),
        patch("wnn.training.sequence_observer.gpu_resources", return_value=({}, {})),
        patch("wnn.training.sequence_observer.host_resources", return_value={}),
    ):
        observer = SequenceObserver(output, 10)
        assert observer.sample()["metrics"]["observer/process_alive_count"] == 1
    assert not output.exists()
    with (
        patch("wnn.training.sequence_observer.read_process", return_value=None),
        pytest.raises(ValueError, match="live readable"),
    ):
        SequenceObserver(tmp_path, 10)
