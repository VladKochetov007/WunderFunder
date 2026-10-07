from __future__ import annotations

import fcntl
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.reclaim_bandit_export_cache import reclaim_completed_export


@pytest.fixture
def cache_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    cgroup = tmp_path / "wnn-campaign.slice"
    cgroup.mkdir()
    for name, value in {
        "memory.max": "20000000000",
        "memory.high": "18000000000",
        "memory.swap.max": "0",
        "cpu.max": "600000 100000",
        "memory.reclaim": "",
    }.items():
        (cgroup / name).write_text(value)
    predecessor = tmp_path / "predecessor.json"
    predecessor.write_text(
        json.dumps(
            {
                "status": "finished",
                "exit_status": 0,
                "kind": "gpu",
                "command": ["python", "scripts/predict_run.py", "teacher", "train"],
                "parent_slice": "wnn-campaign.slice",
                "parent_properties_at_start": {"ControlGroup": "/wnn-campaign.slice"},
            }
        )
    )
    return tmp_path, predecessor, tmp_path / "receipt.json", cgroup


def test_reclaim_keeps_limits_and_uses_file_only(cache_fixture: tuple[Path, ...]) -> None:
    runtime, predecessor, receipt, cgroup = cache_fixture
    before = {"current": 14_000_000_000, "file": 13_000_000_000, "anon": 500_000_000}
    after = {"current": 8_500_000_000, "file": 7_500_000_000, "anon": 500_000_000}
    with patch("scripts.reclaim_bandit_export_cache.memory_snapshot", side_effect=[before, after]):
        reclaim_completed_export(runtime, predecessor, receipt, runtime)
    assert (cgroup / "memory.reclaim").read_text() == "5500000000 swappiness=0\n"
    assert (cgroup / "memory.max").read_text() == "20000000000"
    assert json.loads(receipt.read_text())["status"] == "ready"


def test_active_gpu_prevents_reclaim(cache_fixture: tuple[Path, ...]) -> None:
    runtime, predecessor, receipt, cgroup = cache_fixture
    with (runtime / "gpu.lock").open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            reclaim_completed_export(runtime, predecessor, receipt, runtime)
    assert (cgroup / "memory.reclaim").read_text() == ""


def test_failed_export_prevents_reclaim(cache_fixture: tuple[Path, ...]) -> None:
    runtime, predecessor, receipt, cgroup = cache_fixture
    record = json.loads(predecessor.read_text())
    record["status"] = "failed"
    predecessor.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="successfully completed"):
        reclaim_completed_export(runtime, predecessor, receipt, runtime)
    assert (cgroup / "memory.reclaim").read_text() == ""


def test_insufficient_headroom_does_not_approve(cache_fixture: tuple[Path, ...]) -> None:
    runtime, predecessor, receipt, _ = cache_fixture
    observation = {"current": 12_000_000_000, "file": 2_000_000_000, "anon": 10_000_000_000}
    with (
        patch("scripts.reclaim_bandit_export_cache.memory_snapshot", return_value=observation),
        pytest.raises(RuntimeError, match="headroom"),
    ):
        reclaim_completed_export(runtime, predecessor, receipt, runtime)
    assert json.loads(receipt.read_text())["status"] == "needs_review"
