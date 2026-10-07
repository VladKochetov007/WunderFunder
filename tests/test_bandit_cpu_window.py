from datetime import datetime, timedelta

import pytest

from scripts import bandit_resources as resources


def test_temporary_cpu_window_has_exact_boundaries() -> None:
    start, end = resources.TEMPORARY_CPU_START, resources.TEMPORARY_CPU_END
    assert resources.authorized_cpu_cap(start - timedelta(microseconds=1)) == 6
    assert resources.authorized_cpu_cap(start) == 12
    assert resources.authorized_cpu_cap(end - timedelta(microseconds=1)) == 12
    assert resources.authorized_cpu_cap(end) == 6
    assert resources.authorized_cpu_cap(end + timedelta(days=1)) == 6
    with pytest.raises(ValueError, match="timezone"):
        resources.authorized_cpu_cap(datetime(2026, 10, 8))


def test_twelve_core_parent_still_requires_original_memory_caps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(resources, "authorized_cpu_cap", lambda: 12)
    properties = {
        "CPUQuotaPerSecUSec": "12s",
        "MemoryHigh": "18000000000",
        "MemoryMax": "20000000000",
        "MemorySwapMax": "0",
        "ControlGroup": "/user.slice/user-1000.slice/wnn-campaign.slice",
    }
    assert resources.verify_parent(properties) == properties["ControlGroup"]
    for changed in (
        {"CPUQuotaPerSecUSec": "16s"},
        {"MemoryMax": "24000000000"},
        {"MemorySwapMax": "1000000"},
    ):
        with pytest.raises(RuntimeError):
            resources.verify_parent({**properties, **changed})
