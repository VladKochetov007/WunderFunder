"""CPU tests for exclusive rolling admission, draining, and real callback supervision."""

from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import rolling
from journal import Journal
from process_start import Running, start_owned

import runtime


def exercise_queue(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, fail_first: bool
) -> tuple:
    panel = json.loads((Path(__file__).parents[1] / "PANEL.json").read_text())
    for job in panel["jobs"]:
        job["run_dir"] = str(tmp_path / "runs" / job["id"])
    lanes = {}
    starts = []
    grades = []
    clock = [0.0]
    results = {}
    monkeypatch.setattr(rolling, "_runtime_paths", lambda *a: {})
    monkeypatch.setattr(rolling, "_validate_fit_receipt", lambda job: {})
    monkeypatch.setattr(rolling, "sha256_file", lambda path: "test-hash")
    monkeypatch.setattr(rolling, "_fit_binding", lambda root, job: {"job": job})
    monkeypatch.setattr(
        rolling,
        "_export_one",
        lambda root, paths, fit, deadline: {
            "resource_admitted": True,
            "status": "completed_validated",
            "output_root": tmp_path,
        },
    )

    def grade(root, paths, fold, fits, exports, deadline):
        key = (fold, fits["wide"]["job"]["seed"])
        assert key not in grades
        grades.append(key)
        return {"status": "graded", "fold": fold, "seed": key[1]}

    monkeypatch.setattr(rolling, "_grade_pair", grade)
    monkeypatch.setattr(
        rolling, "cgroup_snapshot", lambda: {"memory_events": {}, "memory_current": 0}
    )
    monkeypatch.setattr(rolling, "require_gpus", lambda *a: None)
    monkeypatch.setattr(
        rolling,
        "actor_environment",
        lambda root, run, uuid: {"CUDA_VISIBLE_DEVICES": uuid},
    )

    def launch(**kwargs):
        uuid = kwargs["environment"]["CUDA_VISIBLE_DEVICES"]
        assert uuid not in lanes, "Two active fits admitted to same GPU"
        duration = 1.0 if "_wide_" in kwargs["name"] else 2.0
        task = Running(
            kwargs["name"],
            SimpleNamespace(pid=500 + len(starts), returncode=0),
            io.BytesIO(),
            clock[0],
            clock[0] + 100,
            gpu_uuid=uuid,
        )
        lanes[uuid] = (clock[0] + duration, task)
        starts.append((task.name, clock[0]))
        return task

    monkeypatch.setattr(rolling, "start_owned", launch)

    def monitor(active, journal, root, deadline, baseline, paths, *, advance):
        while lanes:
            uuid, (end, task) = min(lanes.items(), key=lambda item: item[1][0])
            clock[0] = end
            del lanes[uuid]
            failed = fail_first and not results
            task.status = "failed" if failed else "completed"
            task.log.close()
            results[task.name] = {
                "status": task.status,
                "returncode": 1 if failed else 0,
            }
            advance(active, results)
            assert len(lanes) <= 8
        return results

    monkeypatch.setattr(rolling, "_monitor", monitor)
    result = rolling.run_rolling(
        panel,
        tmp_path,
        tmp_path / "state",
        Journal(tmp_path / "events.jsonl"),
        time.monotonic() + 100,
    )
    return result, starts, grades


def test_queue_fills_free_gpu_before_slower_initial_jobs_finish(
    monkeypatch, tmp_path
) -> None:
    result, starts, grades = exercise_queue(monkeypatch, tmp_path, fail_first=False)
    assert result["status"] == "complete"
    assert len(starts) == len({name for name, _ in starts}) == 60
    assert len(grades) == 30
    assert starts[8][1] == 1.0  # Deep initial fits are still running until time2.
    assert result["remaining_not_admitted"] == 0


def test_failure_stops_new_admission_and_drains_healthy_initial_fits(
    monkeypatch, tmp_path
) -> None:
    result, starts, _ = exercise_queue(monkeypatch, tmp_path, fail_first=True)
    assert result["status"] == "partial"
    assert len(starts) == 8
    assert result["remaining_not_admitted"] == 52
    assert (
        sum(r["status"] == "fixed_epoch5_receipt_verified" for r in result["fits"]) == 7
    )


def test_real_process_monitor_replenishes_on_completion(monkeypatch, tmp_path) -> None:
    snapshot = {"memory_events": {}, "memory_current": 0, "memory_peak": 0}
    monkeypatch.setattr(runtime, "cgroup_snapshot", lambda: snapshot)
    cpu = min(__import__("os").sched_getaffinity(0))
    active = []

    def launch(name):
        return start_owned(
            name=name,
            argv=[sys.executable, "-c", "print('owned process completed')"],
            environment={"PATH": "/usr/bin:/bin"},
            cpus=(cpu,),
            log_path=tmp_path / f"{name}.log",
            timeout_seconds=20,
        )

    active.append(launch("first"))

    def advance(tasks, results):
        if "first" in results and len(tasks) == 1:
            tasks.append(launch("second"))

    result = runtime._monitor(
        active,
        Journal(tmp_path / "events.jsonl"),
        tmp_path,
        time.monotonic() + 30,
        {},
        [tmp_path],
        advance=advance,
    )
    assert set(result) == {"first", "second"}
    assert all(r["returncode"] == 0 for r in result.values())
