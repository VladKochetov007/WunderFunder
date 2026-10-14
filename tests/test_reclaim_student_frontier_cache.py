from __future__ import annotations

import fcntl
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts import reclaim_student_frontier_cache as handoff


def make_attempts(runtime: Path) -> Path:
    attempts = runtime / "attempts"
    attempts.mkdir(parents=True)
    self_record = {
        "id": "handoff-test",
        "kind": "cpu",
        "status": "running",
        "command": ["python", "/work/scripts/reclaim_student_frontier_cache.py"],
        "reservation": {"cpus": 1, "ram_bytes": 400_000_000},
    }
    (attempts / "handoff-test.json").write_text(json.dumps(self_record))
    return attempts


def make_cgroup(cgroup_root: Path) -> tuple[Path, str]:
    parent_group = "/user.slice/test.slice/wnn-campaign.slice"
    cgroup = cgroup_root / parent_group.lstrip("/")
    cgroup.mkdir(parents=True)
    for name, value in {
        "cpu.max": "500000 100000\n",
        "memory.current": "14000000000\n",
        "memory.high": "17999998976\n",
        "memory.max": "19999997952\n",
        "memory.swap.max": "0\n",
        "memory.reclaim": "",
        "memory.stat": "anon 1000000000\nfile 13000000000\n",
    }.items():
        (cgroup / name).write_text(value)
    return cgroup, parent_group


def test_profile_binding_rejects_the_legacy_upstream_predictor(tmp_path: Path) -> None:
    copied = tmp_path / "predict_student_frontier_f0_frozen.py"
    copied.write_text("# frozen student predictor\n")
    upstream = tmp_path / "predict_run.py"
    upstream.write_text("# upstream predictor\n")
    digest = handoff.sha256_file(copied)
    profile = {
        "source": {
            "profiled_predictor_copy": str(copied),
            "profiled_predictor_copy_sha256": digest,
            "upstream_predictor": str(upstream),
        }
    }
    record = {
        "working_directory": str(tmp_path),
        "source_sha256": {copied.name: digest},
    }
    command = [str(copied), "--campaign-profile-first-batch"]
    assert handoff.validate_pinned_predictor(profile, command, record) == (copied, digest)
    with pytest.raises(ValueError, match="pinned student predictor"):
        handoff.validate_pinned_predictor(profile, [str(upstream)], record)


def test_gpu_terminal_gate_checks_reservation_observers_and_oom() -> None:
    zero_oom = {name: 0 for name in handoff.OOM_FIELDS}
    record = {
        "reservation": {"ram_bytes": 6_000_000_000, "cpus": 1},
        "peak_ram_bytes": 5_500_000_000,
        "peak_observed_vram_bytes": 2_000_000_000,
        "observer_status": {name: "ok" for name in handoff.REQUIRED_PROFILE_OBSERVERS},
        "observer_events": [],
        **{
            field: zero_oom.copy()
            for field in (
                "memory_event_deltas",
                "final_memory_event_deltas",
                "parent_memory_event_deltas",
                "final_parent_memory_event_deltas",
                "parent_hierarchical_memory_event_deltas",
                "final_parent_hierarchical_memory_event_deltas",
            )
        },
    }
    assert handoff.terminal_gpu_resources(record, 6_000_000_000)
    assert not handoff.terminal_gpu_resources(record, 5_000_000_000)
    record["final_memory_event_deltas"]["oom_kill"] = 1
    assert not handoff.terminal_gpu_resources(record, 6_000_000_000)


def test_file_cache_handoff_is_file_only_and_keeps_limits(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    make_attempts(runtime)
    cgroup_root = tmp_path / "cgroup"
    cgroup, parent_group = make_cgroup(cgroup_root)
    properties = {
        "CPUQuotaPerSecUSec": "5s",
        "MemoryHigh": "18000000000",
        "MemoryMax": "20000000000",
        "MemorySwapMax": "0",
        "ControlGroup": parent_group,
    }
    before = {"current": 14_000_000_000, "file": 13_000_000_000, "anon": 1_000_000_000}
    after = {"current": 8_500_000_000, "file": 7_500_000_000, "anon": 1_000_000_000}
    with (
        patch.object(handoff, "systemd_parent_properties", return_value=properties),
        patch.object(handoff, "read_memory_snapshot", side_effect=[before, after]),
    ):
        receipt = handoff.reclaim_cache(
            runtime=runtime,
            self_attempt="handoff-test",
            parent_group=parent_group,
            cgroup_root=cgroup_root,
            next_reserve_bytes=handoff.PROFILE_NEXT_RESERVE_BYTES,
        )
    assert receipt["status"] == "ready"
    assert receipt["limits_changed"] is False
    assert receipt["ledger_lock_check"]["active_gpu_attempts"] == []
    assert (cgroup / "memory.reclaim").read_text() == "5500000000 swappiness=0\n"
    assert (cgroup / "cpu.max").read_text() == "500000 100000\n"
    assert (cgroup / "memory.max").read_text() == "19999997952\n"
    assert (cgroup / "memory.high").read_text() == "17999998976\n"
    assert (cgroup / "memory.swap.max").read_text() == "0\n"


def test_active_gpu_attempt_blocks_reclaim(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    attempts = make_attempts(runtime)
    (attempts / "gpu-active.json").write_text(
        json.dumps({"id": "gpu-active", "kind": "gpu", "status": "running"})
    )
    cgroup_root = tmp_path / "cgroup"
    cgroup, parent_group = make_cgroup(cgroup_root)
    properties = {
        "CPUQuotaPerSecUSec": "5s",
        "MemoryHigh": "18000000000",
        "MemoryMax": "20000000000",
        "MemorySwapMax": "0",
        "ControlGroup": parent_group,
    }
    with (
        patch.object(handoff, "systemd_parent_properties", return_value=properties),
        patch.object(handoff, "read_memory_snapshot") as memory_snapshot,
        pytest.raises(RuntimeError, match="active_gpu"),
    ):
        handoff.reclaim_cache(
            runtime=runtime,
            self_attempt="handoff-test",
            parent_group=parent_group,
            cgroup_root=cgroup_root,
            next_reserve_bytes=handoff.FIT_NEXT_RESERVE_BYTES,
        )
    memory_snapshot.assert_not_called()
    assert (cgroup / "memory.reclaim").read_text() == ""


def test_gpu_lock_blocks_reclaim(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    make_attempts(runtime)
    cgroup_root = tmp_path / "cgroup"
    cgroup, parent_group = make_cgroup(cgroup_root)
    properties = {
        "CPUQuotaPerSecUSec": "5s",
        "MemoryHigh": "18000000000",
        "MemoryMax": "20000000000",
        "MemorySwapMax": "0",
        "ControlGroup": parent_group,
    }
    with (runtime / "gpu.lock").open("a+") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (
            patch.object(handoff, "systemd_parent_properties", return_value=properties),
            pytest.raises(BlockingIOError),
        ):
            handoff.reclaim_cache(
                runtime=runtime,
                self_attempt="handoff-test",
                parent_group=parent_group,
                cgroup_root=cgroup_root,
                next_reserve_bytes=handoff.PROFILE_NEXT_RESERVE_BYTES,
            )
    assert (cgroup / "memory.reclaim").read_text() == ""
