from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

PACKET = Path(__file__).resolve().parents[1] / "packet"
sys.path.insert(0, str(PACKET))

from cache_pressure_recovery import (  # noqa: E402
    MAX_ADVICE_BYTES,
    MAX_RELIEF_EVENTS,
    MAX_TOTAL_ADVICE_BYTES,
    MIN_RELIEF_INTERVAL_SECONDS,
    ReliefBudget,
    advise_recent_rowgroups,
    has_reclaimable_file_pressure,
    host_headroom_passes,
    memory_snapshot_matches,
    relief_target_bytes,
)


def _stat(
    *,
    anon: int = 400_000_000,
    inactive: int = 900_000_000,
    active: int = 300_000_000,
    refault: int = 20,
) -> dict[str, int]:
    return {
        "anon": anon,
        "file": inactive + active + 50,
        "inactive_file": inactive,
        "active_file": active,
        "shmem": 50,
        "workingset_refault_file": refault,
    }


def _bindings(paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    result = {}
    for split, path in paths.items():
        stat = path.stat()
        result[split] = {
            "canonical_path": str(path.resolve()),
            "device": stat.st_dev,
            "inode": stat.st_ino,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "ctime_ns": stat.st_ctime_ns,
            "mode": stat.st_mode,
        }
    return result


def _raw_files(tmp_path: Path) -> dict[str, Path]:
    paths = {}
    for split in ("train", "valid"):
        path = tmp_path / f"{split}.parquet"
        path.write_bytes(bytes(300_000))
        paths[split] = path
    return paths


def _fake_fadvise(calls: list[tuple[str, int, int, int]]) -> Any:
    def advise(descriptor: int, offset: int, length: int, advice: int) -> None:
        target = os.readlink(f"/proc/self/fd/{descriptor}")
        calls.append((target, offset, length, advice))

    return advise


def test_cache_evidence_requires_inactive_file_and_stable_anon_and_working_set() -> None:
    previous = _stat()
    current = _stat(refault=30)
    assert has_reclaimable_file_pressure(current, previous, page_size=4_096) == (True, [])

    active_dominates = _stat(inactive=900_000_000, active=4_000_000_000)
    assert not has_reclaimable_file_pressure(active_dominates, previous, page_size=4_096)[0]

    active_growth = _stat(inactive=1_200_000_000, active=600_000_000)
    active_growth_previous = _stat(inactive=900_000_000, active=300_000_000)
    assert (
        "active_file grew too quickly"
        in has_reclaimable_file_pressure(active_growth, active_growth_previous, page_size=4_096)[1]
    )

    anon_growth = _stat(anon=600_000_000)
    previous_anon = _stat(anon=100_000_000)
    assert (
        "anonymous memory grew too quickly"
        in has_reclaimable_file_pressure(anon_growth, previous_anon, page_size=4_096)[1]
    )

    refault_spike = _stat(refault=20 + 20_000)
    assert (
        "file working-set refault rate is high"
        in has_reclaimable_file_pressure(refault_spike, previous, page_size=4_096)[1]
    )

    little_inactive = _stat(inactive=100_000_000)
    assert not has_reclaimable_file_pressure(little_inactive, previous, page_size=4_096)[0]


def test_relief_hints_only_recent_identity_bound_rowgroups_within_budget(tmp_path: Path) -> None:
    paths = _raw_files(tmp_path)
    bindings = _bindings(paths)
    trace = tmp_path / "reads.tsv"
    trace.write_text(
        "train\t2\t8192\t12288\nvalid\t3\t16384\t20480\ntrain\t2\t8192\t12288\n",
        encoding="ascii",
    )
    calls: list[tuple[str, int, int, int]] = []
    result = advise_recent_rowgroups(
        paths,
        bindings,
        trace,
        page_size=4_096,
        fadvise=_fake_fadvise(calls),
        max_bytes=20_000,
    )

    assert result["status"] == "advised"
    assert 0 < result["requested_bytes"] <= 20_000
    assert {Path(call[0]) for call in calls} == set(paths.values())
    assert all(offset % 4_096 == 0 and length % 4_096 == 0 for _, offset, length, _ in calls)
    assert all(advice == os.POSIX_FADV_DONTNEED for _, _, _, advice in calls)


def test_changed_raw_input_identity_fails_closed_without_advising(tmp_path: Path) -> None:
    paths = _raw_files(tmp_path)
    bindings = _bindings(paths)
    bindings["train"]["size"] += 1
    trace = tmp_path / "reads.tsv"
    trace.write_text("train\t2\t8192\t12288\n", encoding="ascii")
    calls: list[tuple[str, int, int, int]] = []
    result = advise_recent_rowgroups(
        paths,
        bindings,
        trace,
        page_size=4_096,
        fadvise=_fake_fadvise(calls),
    )
    assert result["status"] == "failed_closed"
    assert "identity changed" in result["diagnostic"]
    assert calls == []


def test_unsupported_advice_and_invalid_or_empty_trace_fail_closed(tmp_path: Path) -> None:
    paths = _raw_files(tmp_path)
    bindings = _bindings(paths)
    trace = tmp_path / "reads.tsv"
    trace.write_text("train\t2\t8192\t12288\n", encoding="ascii")
    unsupported = advise_recent_rowgroups(
        paths,
        bindings,
        trace,
        page_size=4_096,
        fadvise=object(),  # type: ignore[arg-type]
    )
    assert unsupported["status"] == "unsupported"

    trace.write_text("unknown\t2\t8192\t12288\n", encoding="ascii")
    malformed = advise_recent_rowgroups(
        paths,
        bindings,
        trace,
        page_size=4_096,
        fadvise=_fake_fadvise([]),
    )
    assert malformed["status"] == "failed_closed"

    trace.write_text("", encoding="ascii")
    empty = advise_recent_rowgroups(
        paths,
        bindings,
        trace,
        page_size=4_096,
        fadvise=_fake_fadvise([]),
    )
    assert empty["status"] == "no_recent_rowgroups"

    trace.write_text("train\t2\t8192\t300001\n", encoding="ascii")
    out_of_bounds = advise_recent_rowgroups(
        paths,
        bindings,
        trace,
        page_size=4_096,
        fadvise=_fake_fadvise([]),
    )
    assert out_of_bounds["status"] == "failed_closed"
    assert "outside" in out_of_bounds["diagnostic"]


def test_snapshot_consistency_recovery_margin_and_repeated_attempt_budget() -> None:
    snapshot = {
        "cgroup_memory_stat_status": "available",
        "cgroup_memory_stat_path": "/sys/fs/cgroup/worker",
        "cgroup_available_bytes": 8_000,
        "cgroup_memory_available_bytes": 8_000,
        "cgroup_limit_bytes": 32_000,
        "cgroup_memory_limit_bytes": 32_000,
    }
    assert memory_snapshot_matches(snapshot)
    snapshot["cgroup_memory_available_bytes"] = 7_999
    assert not memory_snapshot_matches(snapshot)
    assert relief_target_bytes(32_212_254_720, 6_500_000_000, 0.25) == 8_565_063_680
    assert host_headroom_passes(6_500, 20_000, minimum_bytes=6_500, minimum_fraction=0.25)
    assert not host_headroom_passes(6_499, 20_000, minimum_bytes=6_500, minimum_fraction=0.25)
    assert not host_headroom_passes(6_500, 40_000, minimum_bytes=6_500, minimum_fraction=0.25)


def test_relief_budget_has_read_horizon_bound_rate_cap_and_failure_ceiling() -> None:
    assert MAX_RELIEF_EVENTS * MAX_ADVICE_BYTES == MAX_TOTAL_ADVICE_BYTES
    assert MAX_RELIEF_EVENTS == 320
    assert MAX_TOTAL_ADVICE_BYTES == 245_760_000_000
    assert MIN_RELIEF_INTERVAL_SECONDS == 20.0

    budget = ReliefBudget(
        maximum_events=2,
        maximum_episode_bytes=768,
        maximum_total_bytes=1_536,
        minimum_interval_seconds=20,
    )
    assert budget.reserve(now=10.0, requested_bytes=768) == (True, None)
    assert budget.reserve(now=29.9, requested_bytes=768) == (False, "minimum_interval")
    assert budget.snapshot()["events"] == 1
    assert budget.reserve(now=30.0, requested_bytes=768) == (True, None)
    assert budget.reserve(now=50.0, requested_bytes=768) == (False, "event_limit")

    total_bytes_limited = ReliefBudget(
        maximum_events=3,
        maximum_episode_bytes=768,
        maximum_total_bytes=768,
        minimum_interval_seconds=0,
    )
    assert total_bytes_limited.reserve(now=0, requested_bytes=768) == (True, None)
    assert total_bytes_limited.reserve(now=1, requested_bytes=768) == (False, "total_byte_limit")
    assert total_bytes_limited.reserve(now=2, requested_bytes=769) == (False, "episode_byte_limit")
