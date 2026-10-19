from __future__ import annotations

import sys
from pathlib import Path

import pytest

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))

import feature_resources as resources  # noqa: E402
from panel import GPU_UUIDS  # noqa: E402


def test_foreign_pid_exemption_is_bound_to_gpu_and_process_start(monkeypatch) -> None:
    pid = 4060404
    gpu, start_ticks = resources.PRESERVED_FOREIGN[pid]
    monkeypatch.setattr(resources, "process_start_ticks", lambda observed: start_ticks)
    assert resources.is_preserved_foreign(pid, gpu)
    assert not resources.is_preserved_foreign(pid, GPU_UUIDS[1])

    monkeypatch.setattr(resources, "process_start_ticks", lambda observed: start_ticks + 1)
    assert not resources.is_preserved_foreign(pid, gpu)
    assert not resources.is_preserved_foreign(pid + 1, gpu)


def test_process_start_ticks_parses_stat_with_parentheses_in_command(tmp_path: Path) -> None:
    pid = 123
    expected = 987654
    fields = ["S", *(["0"] * 18), str(expected)]
    path = tmp_path / str(pid)
    path.mkdir()
    (path / "stat").write_text(f"{pid} (worker (child)) {' '.join(fields)}\n")
    assert resources.process_start_ticks(pid, tmp_path) == expected


def test_resource_gate_requires_exact_eight_gpu_uuid_index_map() -> None:
    devices = {uuid: {"index": index} for index, uuid in enumerate(GPU_UUIDS)}
    resources.validate_device_map(devices)
    missing = dict(devices)
    missing.pop(GPU_UUIDS[-1])
    with pytest.raises(ValueError, match="eight-card panel"):
        resources.validate_device_map(missing)
    reordered = dict(devices)
    reordered[GPU_UUIDS[0]] = {"index": 7}
    with pytest.raises(ValueError, match="eight-card panel"):
        resources.validate_device_map(reordered)


def test_cgroup_members_include_nested_processes(tmp_path: Path) -> None:
    base = tmp_path / "user.slice" / "owner.service"
    child = base / "job.scope"
    child.mkdir(parents=True)
    (base / "cgroup.procs").write_text("10\n")
    (child / "cgroup.procs").write_text("20\n30\n")
    unit = {"ActiveState": "active", "ControlGroup": "/user.slice/owner.service"}
    assert resources.cgroup_members(unit, tmp_path) == [10, 20, 30]
