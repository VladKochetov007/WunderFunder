from __future__ import annotations

import sys
from pathlib import Path

CHECKS = Path(__file__).resolve().parents[2] / "checks"
sys.path.insert(0, str(CHECKS))

from monitor_jobs import alerts  # noqa: E402


def _snapshot(parent_events: dict[str, int | str]) -> dict[str, object]:
    return {
        "provider": {"status_checks_deferred": 0, "kernels": {}},
        "local": {
            "attempts_requiring_attention": [],
            "slice_properties": {
                "MemoryCurrent": "0",
                "MemoryHigh": "36000000000",
                "MemoryMax": "40000000000",
                "CPUQuotaPerSecUSec": "12s",
                "MemorySwapMax": "0",
            },
            "parent_memory_events": parent_events,
            "services": {"stdout": "", "exit": 0},
            "gpu": {"stdout": "0, 0", "exit": 0},
        },
    }


def test_unreadable_parent_memory_events_are_alerted() -> None:
    result = alerts(_snapshot({"observation_error": "permission denied"}), 0.0)
    assert any("parent memory.events unavailable" in item for item in result)


def test_missing_parent_oom_counter_is_alerted() -> None:
    result = alerts(_snapshot({"oom": 0, "oom_kill": 0}), 0.0)
    assert any("missing OOM counters: oom_group_kill" in item for item in result)


def test_observed_parent_oom_counters_do_not_raise_observation_alert() -> None:
    result = alerts(
        _snapshot({"oom": 0, "oom_kill": 0, "oom_group_kill": 0}), 0.0
    )
    assert not any("parent memory.events" in item for item in result)
