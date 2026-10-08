"""Unit checks for the bounded smoke3 kernel's startup and live guards."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

KERNEL_SCRIPT = (
    Path(__file__).parent.parent
    / "research/campaign_2026-10-06/bandit_continuation/kaggle_warm_cohort/smoke3_20261008/kernel"
    / "wnn-warm-cohort-smoke-20261008-v3.py"
)


def load_kernel() -> object:
    """Import the standalone Kaggle script without running its entrypoint."""
    spec = importlib.util.spec_from_file_location("smoke3_kernel_driver", KERNEL_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_kernel_wall_limit_is_hard_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WARM_MAX_WALL_SECONDS", "9000")
    assert load_kernel().MAX_WALL == 7800

    monkeypatch.setenv("WARM_MAX_WALL_SECONDS", "600")
    assert load_kernel().MAX_WALL == 600


def test_resource_snapshot_records_worker_rss_and_gpu_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel = load_kernel()

    class FakeProcess:
        pid = 100

        @staticmethod
        def poll() -> None:
            return None

    monkeypatch.setattr(kernel, "_host_available_bytes", lambda: 30_000_000_000)
    monkeypatch.setattr(kernel, "_cgroup_available_bytes", lambda: 28_000_000_000)
    monkeypatch.setattr(
        kernel,
        "_process_memory_table",
        lambda: {100: (1, 100), 200: (100, 200), 300: (200, 300)},
    )
    monkeypatch.setattr(
        kernel,
        "gpu_inventory",
        lambda: [
            {"index": 0, "memory_used_mib": 100, "memory_total_mib": 16000},
            {"index": 1, "memory_used_mib": 200, "memory_total_mib": 16000},
        ],
    )

    snapshot = kernel.resource_snapshot({0: FakeProcess()})

    assert snapshot["host_available_bytes"] == 30_000_000_000
    assert snapshot["effective_available_bytes"] == 28_000_000_000
    assert snapshot["worker_rss_bytes_by_gpu"] == {"0": 600}
    assert snapshot["worker_aggregate_rss_bytes"] == 600
    assert snapshot["gpu_count"] == 2
    assert len(snapshot["gpus"]) == 2


@pytest.mark.parametrize(
    ("available", "gpu_count", "expected_issue"),
    [
        (19_999_999_999, 2, "below 20 GB"),
        (20_000_000_000, 1, "fewer than 2 GPUs"),
        (20_000_000_000, 2, None),
    ],
)
def test_startup_resource_gate_fails_closed(
    available: int, gpu_count: int, expected_issue: str | None
) -> None:
    kernel = load_kernel()
    issues = kernel._startup_resource_issues(
        {"effective_available_bytes": available, "gpu_count": gpu_count}
    )
    if expected_issue is None:
        assert issues == []
    else:
        assert any(expected_issue in issue for issue in issues)


def test_runtime_resource_gate_stops_on_low_memory_or_rss() -> None:
    kernel = load_kernel()
    base = {
        "effective_available_bytes": 3_000_000_000,
        "worker_aggregate_rss_bytes": 24_000_000_000,
        "gpu_count": 2,
    }
    assert kernel._runtime_resource_issues(base) == []
    assert "effective available memory is below 2 GB" in kernel._runtime_resource_issues(
        {**base, "effective_available_bytes": 1_999_999_999}
    )
    assert "worker-tree RSS exceeds 24 GB or is unavailable" in kernel._runtime_resource_issues(
        {**base, "worker_aggregate_rss_bytes": 24_000_000_001}
    )


def test_supervisor_uses_entry_deadline_and_emits_resource_heartbeats(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    kernel = load_kernel()
    kernel.WORK = tmp_path
    kernel.MAX_WALL = 100
    kernel.SUPERVISOR_SHUTDOWN_RESERVE_SECONDS = 60
    kernel.SUPERVISOR_HEARTBEAT_SECONDS = 30

    class FakeClock:
        now = 0.0

        @classmethod
        def monotonic(cls) -> float:
            return cls.now

        @classmethod
        def sleep(cls, seconds: float) -> None:
            cls.now += seconds

    class FakeProcess:
        pid = 123
        returncode: int | None = None

        def poll(self) -> int | None:
            return self.returncode

    process = FakeProcess()
    records: list[tuple[str, dict[str, object]]] = []
    stop_reasons: list[str] = []
    monkeypatch.setattr(kernel, "time", FakeClock)
    monkeypatch.setattr(kernel, "phase", lambda name, **fields: records.append((name, fields)))
    monkeypatch.setattr(
        kernel,
        "resource_snapshot",
        lambda workers: {
            "effective_available_bytes": 3_000_000_000,
            "worker_aggregate_rss_bytes": 1_000_000_000,
            "gpu_count": 2,
            "gpus": [],
        },
    )

    def stop_workers(workers: dict[int, object], reason: str) -> None:
        stop_reasons.append(reason)
        process.returncode = -15

    monkeypatch.setattr(kernel, "_terminate_workers", stop_workers)
    kernel.supervise({0: process}, started_monotonic=0)

    heartbeats = [fields for name, fields in records if name == "heartbeat"]
    assert len(heartbeats) == 2
    assert all("resources" in heartbeat for heartbeat in heartbeats)
    assert stop_reasons == ["hard_wall_limit"]
    assert any(name == "wall_limit_reached" for name, _ in records)


def test_process_descendant_scan_tracks_nested_worker_children() -> None:
    kernel = load_kernel()
    table = {10: (1, 100), 20: (10, 200), 30: (20, 300), 40: (1, 400)}

    assert kernel._descendants(10, table) == {10, 20, 30}
