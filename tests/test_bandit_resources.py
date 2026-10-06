from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import pytest

from scripts import bandit_resources as resources


def parent_properties() -> dict[str, str]:
    return {
        "CPUQuotaPerSecUSec": "10s",
        "MemoryHigh": "18000000000",
        "MemoryMax": "20000000000",
        "MemorySwapMax": "0",
        "ControlGroup": "/user.slice/user-1000.slice/wnn-campaign.slice",
    }


def attempt_record(attempt: str, *, cpus: int, ram_bytes: int) -> dict[str, object]:
    return {
        "id": attempt,
        "status": "running",
        "reservation": {"cpus": cpus, "ram_bytes": ram_bytes},
    }


def worker_workspace(
    tmp_path: Path, kind: str = "gpu"
) -> tuple[Path, Path, str, dict[str, object]]:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "src/wnn").mkdir(parents=True)
    state = tmp_path / "runtime"
    group = parent_properties()["ControlGroup"] + "/wnn-bandit-a1.service"
    record: dict[str, object] = {
        "id": "a1",
        "unit": "wnn-bandit-a1.service",
        "kind": kind,
        "command": ["fake-worker"],
        "contract": str(tmp_path / "contract.md"),
        "source_sha256": {},
        "reservation": {"cpus": 1, "ram_bytes": 2_000_000_000},
        "status": "reserved",
        "peak_ram_bytes": 0,
        "peak_parent_ram_bytes": 2_000_000_000,
        "peak_observed_vram_bytes": 0,
        "observer_events": [],
    }
    (tmp_path / "contract.md").write_text("test contract")
    resources.atomic_json(resources.attempt_path(state, "a1"), record)
    return root, state, group, record


def prepare_worker_observers(
    monkeypatch: pytest.MonkeyPatch, group: str, *, gpu_reads: list[int | Exception] | None = None
) -> None:
    parent = parent_properties()
    monkeypatch.setattr(resources, "parent_properties", lambda: parent)
    monkeypatch.setattr(resources, "current_cgroup", lambda _pid="self": group)
    monkeypatch.setattr(resources, "verify_parent_membership", lambda *_args: None)
    monkeypatch.setattr(
        resources,
        "systemd_properties",
        lambda _unit: {
            "ControlGroup": group,
            "ActiveState": "active",
            "CPUQuotaPerSecUSec": "1s",
            "MemoryMax": "2000000000",
            "MemorySwapMax": "0",
            "OOMPolicy": "continue",
            "KillMode": "control-group",
        },
    )
    monkeypatch.setattr(resources, "source_manifest", lambda *_args: {})
    monkeypatch.setattr(
        resources,
        "memory_current",
        lambda path: 100_000_000 if path.endswith("wnn-bandit-a1.service") else 2_000_000_000,
    )
    monkeypatch.setattr(
        resources,
        "memory_events",
        lambda _path: {"oom": 0, "oom_kill": 0, "oom_group_kill": 0},
    )
    monkeypatch.setattr(resources, "support_cpu_reservations", lambda *_args: {})
    monkeypatch.setattr(resources.time, "sleep", lambda _seconds: None)
    if gpu_reads is not None:
        results = iter(gpu_reads)

        def gpu_memory() -> int:
            value = next(results)
            if isinstance(value, Exception):
                raise value
            return value

        monkeypatch.setattr(resources, "gpu_memory_bytes", gpu_memory)


def test_parent_requires_exact_aggregate_caps() -> None:
    assert resources.verify_parent(parent_properties()).endswith("/wnn-campaign.slice")
    changed = {**parent_properties(), "MemoryMax": "24000000000"}
    with pytest.raises(RuntimeError, match="not verified"):
        resources.verify_parent(changed)


def test_admission_obeys_aggregate_cpu_ram_and_measured_headroom(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = tmp_path / "state"
    resources.atomic_json(
        resources.attempt_path(state, "existing"),
        attempt_record("existing", cpus=8, ram_bytes=12_000_000_000),
    )
    monkeypatch.setattr(resources, "support_cpu_reservations", lambda *_args: {})
    monkeypatch.setattr(resources, "memory_current", lambda _group: 2_000_000_000)
    with pytest.raises(RuntimeError, match="CPU reservation"):
        resources.verify_capacity(state, "/test/parent", 3, 1_000_000_000)
    with pytest.raises(RuntimeError, match="reservations"):
        resources.verify_capacity(state, "/test/parent", 1, 7_000_000_000)
    admitted = resources.verify_capacity(state, "/test/parent", 1, 5_000_000_000)
    assert admitted["parent_memory_current_bytes"] == 2_000_000_000
    monkeypatch.setattr(resources, "memory_current", lambda _group: 14_000_000_000)
    with pytest.raises(RuntimeError, match="measured parent RAM"):
        resources.verify_capacity(state, "/test/parent", 1, 5_000_000_000)


def test_launch_preflight_excludes_its_own_unresolved_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    (root / "scripts" / "limited").write_text("#!/bin/sh\n", encoding="utf-8")
    contract = tmp_path / "CONTRACT.md"
    contract.write_text("frozen test contract\n", encoding="utf-8")
    state = tmp_path / "runtime"
    monkeypatch.setattr(resources, "verify_controller_membership", lambda: None)
    monkeypatch.setattr(resources, "parent_properties", parent_properties)
    monkeypatch.setattr(resources, "verify_parent_membership", lambda *_args: None)
    monkeypatch.setattr(resources, "memory_current", lambda _group: 1_000_000_000)
    monkeypatch.setattr(
        resources,
        "memory_events",
        lambda _group: {"oom": 0, "oom_kill": 0, "oom_group_kill": 0},
    )
    monkeypatch.setattr(resources, "support_cpu_reservations", lambda *_args: {})
    monkeypatch.setattr(
        resources.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(command, 0, "", ""),
    )
    args = argparse.Namespace(
        attempt="launch-v2",
        kind="cpu",
        cpus=1,
        reserve_bytes=1_000_000_000,
        admission_timeout=5,
        command=["python", "-c", "pass"],
        contract=contract,
        state=state,
        root=root,
        python="python",
        profile="smoke",
    )

    launched = resources.launch(args)

    assert launched["status"] == "reserved"
    record = resources.read_json(resources.attempt_path(state, "launch-v2"))
    assert record["status"] == "reserved"
    assert record["admission_measurement"]["committed_cpu_cores"] == 0


def test_another_unresolved_attempt_still_blocks_admission(tmp_path: Path) -> None:
    state = tmp_path / "state"
    own = attempt_record("own", cpus=1, ram_bytes=1_000_000_000)
    own["status"] = "preflight"
    blocked = attempt_record("blocked", cpus=1, ram_bytes=1_000_000_000)
    blocked["status"] = "launch_uncertain"
    resources.atomic_json(resources.attempt_path(state, "own"), own)
    resources.atomic_json(resources.attempt_path(state, "blocked"), blocked)

    with pytest.raises(RuntimeError, match="blocked"):
        resources.active_reservations(state, exclude="own")


def test_active_support_units_are_reserved_from_parent_cgroup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    group = tmp_path / "wnn-campaign.slice"
    group.mkdir()
    for name, cpu_max in (
        ("marimo.service", "200000 100000"),
        ("freebuff.scope", "100000 100000"),
        ("watcher.service", "5000 100000"),
    ):
        child = group / name
        child.mkdir()
        (child / "cgroup.events").write_text("populated 1\n")
        (child / "cpu.max").write_text(f"{cpu_max}\n")
    monkeypatch.setattr(resources, "cgroup_dir", lambda _path: group)
    support = resources.support_cpu_reservations(tmp_path / "state", "/test/parent")
    assert support == {"freebuff.scope": 1.0, "marimo.service": 2.0, "watcher.service": 0.05}
    monkeypatch.setattr(resources, "memory_current", lambda _path: 1_000_000_000)
    admitted = resources.verify_capacity(tmp_path / "state", "/test/parent", 6, 1_000_000_000)
    assert admitted["support_cpu_reserve_cores"] == 3.05
    with pytest.raises(RuntimeError, match="CPU reservation"):
        resources.verify_capacity(tmp_path / "state", "/test/parent", 7, 1_000_000_000)


def test_gpu_service_is_directly_bounded_under_campaign_parent(tmp_path: Path) -> None:
    command = resources.service_command(
        tmp_path, tmp_path / "runtime", "fit_s3", "gpu", 5, 12_000_000_000, "python", 900
    )

    assert "--slice=wnn-campaign.slice" in command
    assert "CPUQuota=500%" in command
    assert "MemoryMax=12000000000" in command
    assert "MemorySwapMax=0" in command
    assert "OOMPolicy=continue" in command
    assert "KillMode=control-group" in command
    assert "--setenv=CUDA_VISIBLE_DEVICES=0" in command
    assert "--setenv=WNN_GPU_MEMORY_BYTES=4300000000" in command
    assert "--setenv=WNN_VRAM_GB=4.3" in command
    assert "--setenv=WNN_LOADER_WORKERS=0" in command


def test_gpu_observer_timeout_is_unknown_and_does_not_kill_healthy_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, state, group, record = worker_workspace(tmp_path)
    timeout = subprocess.TimeoutExpired("nvidia-smi", 5)
    prepare_worker_observers(
        monkeypatch,
        group,
        gpu_reads=[100_000_000, timeout, timeout, timeout, 100_000_000],
    )
    systemd_calls = 0
    verified_unit = {
        "ControlGroup": group,
        "ActiveState": "active",
        "CPUQuotaPerSecUSec": "1s",
        "MemoryMax": "2000000000",
        "MemorySwapMax": "0",
        "OOMPolicy": "continue",
        "KillMode": "control-group",
    }

    def flaky_systemd(_unit: str) -> dict[str, str]:
        nonlocal systemd_calls
        systemd_calls += 1
        if 2 <= systemd_calls <= 4:
            raise subprocess.TimeoutExpired("systemctl show", 5)
        return verified_unit

    monkeypatch.setattr(resources, "systemd_properties", flaky_systemd)
    monkeypatch.setattr(resources, "memory_peak", lambda _group: 100_000_000)
    polls = 0

    class Child:
        pid = os.getpid()
        returncode: int | None = None

        def poll(self) -> int | None:
            nonlocal polls
            polls += 1
            if polls == 1:
                return None
            self.returncode = 0
            return 0

        def wait(self, timeout: float | None = None) -> int:
            return 0 if self.returncode is None else self.returncode

    monkeypatch.setattr(resources.subprocess, "Popen", lambda *_args, **_kwargs: Child())
    monkeypatch.setattr(
        resources, "stop_child", lambda _child: pytest.fail("healthy child was stopped")
    )
    monkeypatch.setattr(resources, "POLL_SECONDS", 0)
    result = resources.worker(root, state, "a1", 2)
    final = resources.read_json(resources.attempt_path(state, "a1"))
    assert result == 0
    assert final["status"] == "finished"
    assert final["exit_status"] == 0
    assert {
        event["observer"] for event in final["observer_events"] if event["state"] == "unknown"
    } >= {"nvidia_smi_runtime", "attempt_systemd_runtime"}
    assert not (state / "STOP_ADMISSION.json").exists()


def test_observed_vram_breach_stops_only_that_child_and_closes_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, state, group, _ = worker_workspace(tmp_path)
    prepare_worker_observers(
        monkeypatch,
        group,
        gpu_reads=[100_000_000, 5_100_000_000, 100_000_000],
    )
    child_instance: object | None = None

    class Child:
        pid = os.getpid()
        returncode: int | None = None

        def poll(self) -> int | None:
            return self.returncode

        def wait(self, timeout: float | None = None) -> int:
            return -15 if self.returncode is None else self.returncode

    def start_child(*_args: object, **_kwargs: object) -> Child:
        nonlocal child_instance
        child_instance = Child()
        return child_instance

    def stop_only_child(child: Child) -> None:
        child.returncode = -15

    monkeypatch.setattr(resources.subprocess, "Popen", start_child)
    monkeypatch.setattr(resources, "stop_child", stop_only_child)
    monkeypatch.setattr(resources, "memory_peak", lambda _group: 100_000_000)
    assert resources.worker(root, state, "a1", 2) == 1
    final = resources.read_json(resources.attempt_path(state, "a1"))
    assert child_instance is not None
    assert final["status"] == "failed"
    assert final["exit_status"] == -15
    assert "observed VRAM exceeded" in str(final["failure"])
    assert resources.read_json(state / "STOP_ADMISSION.json")["reason"].startswith(
        "resource budget breach in a1"
    )


@pytest.mark.parametrize("exit_status", [7, -9])
def test_abnormal_child_exit_is_durable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exit_status: int
) -> None:
    root, state, group, _ = worker_workspace(tmp_path, kind="cpu")
    prepare_worker_observers(monkeypatch, group)

    class Child:
        pid = os.getpid()
        returncode = exit_status

        def poll(self) -> int:
            return exit_status

        def wait(self, timeout: float | None = None) -> int:
            return exit_status

    monkeypatch.setattr(resources.subprocess, "Popen", lambda *_args, **_kwargs: Child())
    assert resources.worker(root, state, "a1", 2) == 1
    final = resources.read_json(resources.attempt_path(state, "a1"))
    assert final["status"] == "failed"
    assert final["exit_status"] == exit_status


def test_drain_closes_admission_and_waits_without_stopping_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = tmp_path / "state"
    resources.atomic_json(
        resources.attempt_path(state, "active"),
        {**attempt_record("active", cpus=2, ram_bytes=3_000_000_000), "unit": "unit.service"},
    )
    monkeypatch.setattr(resources, "reconcile", lambda _state: [])

    def finish_after_barrier(_seconds: float) -> None:
        record = resources.read_json(resources.attempt_path(state, "active"))
        assert record["status"] == "running"
        record["status"] = "finished"
        resources.atomic_json(resources.attempt_path(state, "active"), record)

    monkeypatch.setattr(resources.time, "sleep", finish_after_barrier)
    drained = resources.drain(state, "explicit user stop", poll_seconds=0)
    assert drained[0]["status"] == "finished"
    assert resources.read_json(state / "STOP_ADMISSION.json")["reason"] == "explicit user stop"


def test_stop_barrier_is_durable_even_with_malformed_attempt_evidence(tmp_path: Path) -> None:
    state = tmp_path / "state"
    resources.atomic_json(
        resources.attempt_path(state, "starting"),
        {**attempt_record("starting", cpus=1, ram_bytes=1_000_000_000), "status": "preflight"},
    )
    broken = state / "attempts" / "broken.json"
    broken.write_text("{invalid", encoding="utf-8")

    barrier = resources.close_admission(state, "explicit stop")

    assert barrier["drain_attempts"] == ["starting"]
    assert barrier["unreadable_attempt_records"] == ["broken.json"]
    assert (state / "STOP_ADMISSION.json").is_file()


def test_gpu_lock_rejects_a_second_simultaneous_worker(tmp_path: Path) -> None:
    state = tmp_path / "state"
    state.mkdir()
    owner = (state / "gpu.lock").open("a")
    import fcntl

    fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
    record: dict[str, object] = {"id": "queued", "status": "waiting"}
    try:
        lock = resources.wait_for_gpu_lock(state, 0, record, state / "queued.json")
    finally:
        owner.close()
    assert lock is None
    assert record["status"] == "not_started"
