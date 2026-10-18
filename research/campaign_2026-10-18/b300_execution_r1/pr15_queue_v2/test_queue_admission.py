"""The queue must drain an existing healthy fit after a later spawn fails."""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import execute as driver
import pytest


def test_spawn_failure_does_not_cancel_healthy_fit(monkeypatch, tmp_path) -> None:
    """Exercise the real admission loop with an injected second-launch failure."""
    state = tmp_path / "state"
    monkeypatch.setattr(driver, "STATE", state)
    monkeypatch.setattr(driver, "ROOT", tmp_path)
    monkeypatch.setattr(driver, "cgroup_snapshot", lambda: {"memory_events": {}})
    launches = []
    cancellations = []

    def launch(job: dict) -> SimpleNamespace:
        launches.append(job["id"])
        if len(launches) == 2:
            raise OSError("injected process creation failure")
        return SimpleNamespace(name=job["id"], process=SimpleNamespace(pid=99999), status="running")

    def monitor(active: list, *args, **kwargs) -> dict:
        assert len(active) == 1 and active[0].name == "healthy"
        events = [json.loads(line) for line in (state / "events.jsonl").read_text().splitlines()]
        assert any(row["event"] == "admission_failed" for row in events)
        active[0].status = "completed"
        return {"healthy": {"status": "completed", "returncode": 0}}

    monkeypatch.setattr(driver, "launch", launch)
    monkeypatch.setattr(driver, "_monitor", monitor)
    monkeypatch.setattr(driver, "terminate_owned", cancellations.append)
    jobs = [
        {"id": "healthy", "gpu_index": 0, "lane": 0},
        {"id": "spawn_fails", "gpu_index": 0, "lane": 1},
    ]
    with pytest.raises(RuntimeError, match="Successor incomplete"):
        driver.run(jobs, jobs, time.time() + 24 * 3600)
    assert launches == ["healthy", "spawn_fails"]
    assert cancellations == []
