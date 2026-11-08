from __future__ import annotations

import json
from pathlib import Path

from scripts import tl3_dashboard_bridge as bridge
from scripts.l4_teacher_capacity_20261104.live_observability import (
    _append_watch_events,
    _latest_non_resource_step,
)


def _event(kind: str, step: int, epoch: int) -> dict[str, object]:
    return {
        "schema": bridge.EVENT_SCHEMA,
        "kind": kind,
        "step": step,
        "epoch": epoch,
        "wall_time": "2026-11-04T00:00:00Z",
        "metrics": {},
        "context": {},
    }


def test_latest_non_resource_step_ignores_old_observer_sample(tmp_path: Path) -> None:
    event_path = tmp_path / "monitor_events.jsonl"
    rows = [
        _event("train_step", 400, 1),
        _event("train_step", 500, 2),
        _event("resources", 1_000_000_000, 2),
    ]
    event_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    assert _latest_non_resource_step(event_path) == 500


def test_watch_resource_event_uses_real_step_and_keeps_sample_context(tmp_path: Path) -> None:
    event_path = tmp_path / "monitor_events.jsonl"
    event_path.write_text(json.dumps(_event("train_step", 500, 2)) + "\n", encoding="utf-8")
    record = {
        "sample_id": 17,
        "observed_at_utc": "2026-11-04T00:01:00Z",
        "latest_epoch": 2.0,
        "cells": {"cell-a": {"disk_free_bytes": 1234}},
    }

    assert _append_watch_events(tmp_path, "cell-a", "run-a", "a" * 64, [record]) == 1
    written = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines()]
    resource = written[-1]
    assert resource["kind"] == "resources"
    assert resource["step"] == 500
    assert resource["context"]["sample_id"] == 17
