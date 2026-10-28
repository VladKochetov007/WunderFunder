from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.galam_l4_quality8 import fit_processes, host, monitor, supervisor
from scripts.galam_l4_quality8.common import LaunchError, atomic_write_json, sha256_file
from scripts.galam_l4_quality8.contract import load_panel_cells
from wnn.training.sequence_config import load_config
from wnn.training.sequence_provenance import canonical_sha256

REPO = Path(__file__).resolve().parents[1]
PACKET = REPO / "research" / "experiment_drafts" / "galam-l4-quality8-20261028" / "launch"


@pytest.fixture
def approved_packet(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    """Create a temporary approved panel and eight hash-bound mock profiles."""
    packet = tmp_path / "launch"
    shutil.copytree(PACKET / "configs", packet / "configs")
    panel = json.loads((PACKET / "PANEL.json").read_text(encoding="utf-8"))
    panel["status"] = "approved_for_launch"
    panel["batch_status"] = "profiled_and_frozen"
    for cell in panel["cells"]:
        cell["config_file_sha256"] = _sha256(packet / cell["config"])
    panel_path = packet / "PANEL.json"
    panel_path.write_text(json.dumps(panel), encoding="utf-8")
    profiles: dict[str, Path] = {}
    for cell in panel["cells"]:
        config = load_config(packet / cell["config"])
        identity = {
            "config_sha256": canonical_sha256(config),
            "source_sha256": "a" * 64,
        }
        path = packet / "profiles" / f"{cell['id']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "receipt_schema": "wnn.sequence-profile.v1",
                    "status": "profile_complete",
                    "approved": True,
                    "identity": identity,
                }
            ),
            encoding="utf-8",
        )
        profiles[cell["id"]] = path
    return panel_path, profiles


def test_panel_admission_requires_exact_eight_unique_cells(approved_packet: tuple[Path, dict[str, Path]]) -> None:
    panel_path, profiles = approved_packet
    panel, cells = load_panel_cells(panel_path, profiles)
    assert panel["scientific_fits"] == 8
    assert len(cells) == 8
    assert [cell.gpu for cell in cells] == list(range(8))
    with pytest.raises(LaunchError, match="exactly"):
        load_panel_cells(panel_path, dict(list(profiles.items())[:-1]))


def test_panel_admission_rejects_duplicate_gpu(approved_packet: tuple[Path, dict[str, Path]]) -> None:
    panel_path, profiles = approved_packet
    panel = json.loads(panel_path.read_text(encoding="utf-8"))
    panel["cells"][1]["gpu"] = 0
    panel_path.write_text(json.dumps(panel), encoding="utf-8")
    with pytest.raises(LaunchError, match="one fit to each GPU"):
        load_panel_cells(panel_path, profiles)


def test_host_probe_requires_exactly_eight_l4_uuids(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(host.sys, "platform", "linux")
    monkeypatch.setattr(
        host.os, "sched_getaffinity", lambda _pid: set(range(80)), raising=False
    )
    original_read_text = Path.read_text

    def read_text(path: Path, *args: Any, **kwargs: Any) -> str:
        normalized = str(path).replace("\\", "/")
        if normalized.endswith("node0/cpulist"):
            return "0-31"
        if normalized.endswith("node1/cpulist"):
            return "48-79"
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)

    def completed(command: list[str], **_kwargs: Any) -> SimpleNamespace:
        if "--query-gpu=index" in command[1]:
            rows = [
                f"{index}, NVIDIA L4, GPU-{index:08x}-abcd-ef01-2345-{index:012x}, 23034, 22000"
                for index in range(8)
            ]
            return SimpleNamespace(stdout="\n".join(rows))
        if command[1:2] == ["-c"]:
            return SimpleNamespace(
                stdout=json.dumps(
                    [
                        {
                            "name": "NVIDIA L4",
                            "uuid": f"{index:08x}-abcd-ef01-2345-{index:012x}",
                            "physical_uuid": f"GPU-{index:08x}-abcd-ef01-2345-{index:012x}",
                            "capability": [8, 9],
                            "total_memory_bytes": 23_670_685_696,
                        }
                        for index in range(8)
                    ]
                )
            )
        return SimpleNamespace(stdout="\n".join(["570.0"] * 8))

    monkeypatch.setattr(host.subprocess, "run", completed)
    state = host.probe_host()
    assert set(state.gpus) == set(range(8))
    assert state.driver == "570.0"
    assert state.cuda_devices[
        "GPU-00000000-abcd-ef01-2345-000000000000"
    ]["total_memory_bytes"] == 23_670_685_696

    def seven_gpus(command: list[str], **kwargs: Any) -> SimpleNamespace:
        result = completed(command, **kwargs)
        if "--query-gpu=index" in command[1]:
            result.stdout = "\n".join(result.stdout.splitlines()[:7])
        return result

    monkeypatch.setattr(host.subprocess, "run", seven_gpus)
    with pytest.raises(LaunchError, match="eight indexed GPUs"):
        host.probe_host()


def test_host_profile_uses_exact_cuda_memory_not_nvidia_smi_capacity(
    approved_packet: tuple[Path, dict[str, Path]], monkeypatch: pytest.MonkeyPatch
) -> None:
    panel_path, profiles = approved_packet
    _, cells = load_panel_cells(panel_path, profiles)
    monkeypatch.setattr(host.torch, "__version__", host.EXPECTED_TORCH)
    monkeypatch.setattr(host.torch.version, "cuda", host.EXPECTED_CUDA)
    monkeypatch.setattr(host.torch.backends.cudnn, "version", lambda: 8902)
    cuda_bytes = 23_670_685_696
    gpus: dict[int, dict[str, Any]] = {}
    cuda_devices: dict[str, dict[str, Any]] = {}
    for index in range(8):
        uuid = f"GPU-{index:08x}-abcd-ef01-2345-{index:012x}"
        raw_uuid = uuid.removeprefix("GPU-")
        gpus[index] = {
            "name": "NVIDIA L4",
            "uuid": uuid,
            "total_mib": 23034,
            "free_mib": 22000,
        }
        cuda_devices[uuid] = {
            "name": "NVIDIA L4",
            "uuid": raw_uuid,
            "physical_uuid": uuid,
            "capability": [8, 9],
            "total_memory_bytes": cuda_bytes,
        }
    state = host.HostState(
        gpus=gpus,
        cuda_devices=cuda_devices,
        driver="570.0",
        available_cpus=frozenset(range(80)),
        numa_cpus={0: frozenset(range(32)), 1: frozenset(range(48, 80))},
    )
    for cell in cells:
        device = {
            "rank": 0,
            "local_rank": 0,
            "name": "NVIDIA L4",
            "capability": [8, 9],
            "uuid": cuda_devices[gpus[cell.gpu]["uuid"]]["uuid"],
            "total_memory_bytes": cuda_bytes,
        }
        raw_runtime = {
            "python": host.platform.python_version(),
            "platform": host.platform.platform(),
            "torch": host.EXPECTED_TORCH,
            "cuda_runtime": host.EXPECTED_CUDA,
            "cudnn": 8902,
            "nvidia_driver": "570.0",
            "visible_devices": 1,
            "world_size": 1,
            "matmul_precision": "highest",
            "matmul_tf32": False,
            "cudnn_tf32": False,
            "autocast": False,
            "amp_scaler": False,
            "cublas_workspace_config": host.EXPECTED_CUBLAS_WORKSPACE,
            "devices": [device],
        }
        partial = {"precision": host.precision_identity("fp32", "math", environ={})}
        cell.profile["identity"]["runtime"] = host._runtime_precision(
            cell, raw_runtime, partial
        )
    assert abs(cuda_bytes - 23034 * (1 << 20)) > 1 << 20
    assert len(host.validate_host_profiles(cells, state)) == 8


def test_launch_reserves_output_and_spawns_supervisor_detached(
    tmp_path: Path, approved_packet: tuple[Path, dict[str, Path]], monkeypatch: pytest.MonkeyPatch
) -> None:
    panel, profiles = approved_packet
    output = tmp_path / "results"
    seen: dict[str, Any] = {}

    def fake_popen(command: list[str], **kwargs: Any) -> SimpleNamespace:
        seen.update(command=command, kwargs=kwargs)
        return SimpleNamespace(pid=12345)

    monkeypatch.setattr(supervisor.sys, "platform", "linux")
    monkeypatch.setattr(supervisor.subprocess, "Popen", fake_popen)
    result = supervisor.reserve_and_detach(
        python=Path(sys.executable),
        script=REPO / "scripts" / "launch_galam_quality8.py",
        repo_root=REPO,
        panel=panel,
        profiles=[f"{cell}={path}" for cell, path in profiles.items()],
        data_root=tmp_path / "data",
        split_manifest=tmp_path / "split.json",
        output_root=output,
    )
    assert result["supervisor_pid"] == 12345
    assert seen["kwargs"]["start_new_session"] is True
    assert seen["kwargs"]["stdin"] is supervisor.subprocess.DEVNULL
    assert "shell" not in seen["kwargs"]
    receipt = json.loads((output / "launch_receipt.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "reserved"
    with pytest.raises(LaunchError, match="already exists"):
        supervisor.reserve_and_detach(
            python=Path(sys.executable),
            script=REPO / "scripts" / "launch_galam_quality8.py",
            repo_root=REPO,
            panel=panel,
            profiles=[],
            data_root=tmp_path / "data",
            split_manifest=tmp_path / "split.json",
            output_root=output,
        )


def test_monitor_reads_terminal_artifacts_and_accepts_declared_early_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launch_root = tmp_path / "launch"
    runs_root = launch_root / "runs"
    runs_root.mkdir(parents=True)
    cells: dict[str, dict[str, Any]] = {}
    for index in range(8):
        cell_id = f"g{index}-cell"
        output = runs_root / cell_id
        output.mkdir()
        identity = {"cell": index}
        early_stop = index == 0
        completed_epochs = 20
        model_path = output / "model.pt"
        model_path.write_bytes(b"mock model weights")
        status = "early_stopped" if early_stop else "completed"
        (output / "terminal_status.json").write_text(
            json.dumps(
                {
                    "status": status,
                    "completion_kind": "declared_early_stop" if early_stop else "max_epochs",
                    "completed_epochs": completed_epochs,
                    "requested_epochs": 20,
                    "global_updates": 1000,
                    "model_sha256": sha256_file(model_path),
                    "identity": identity,
                }
            ),
            encoding="utf-8",
        )
        (output / "checkpoint_pointers.json").write_text(
            json.dumps(
                {
                    "run_identity_sha256": canonical_sha256(identity),
                    "completed_epochs": completed_epochs,
                    "status": status,
                    "best_raw_wp": 0.6,
                    "best": {"epoch": 3, "valid_wp": 0.6, "sha256": "a" * 64},
                    "last": {
                        "epoch": completed_epochs,
                        "valid_wp": 0.5,
                        "sha256": "b" * 64,
                    },
                    "early_stopping": {
                        "min_epoch": 5,
                        "patience": 4,
                        "min_delta": 0.0005,
                        "bad_count": 4 if early_stop else 0,
                    },
                }
            ),
            encoding="utf-8",
        )
        (output / "monitor_evaluations.jsonl").write_text(
            json.dumps(
                {
                    "schema": "wnn.monitor.v1",
                    "kind": "evaluation",
                    "epoch": completed_epochs,
                    "context": {
                        "valid": {"row_mask": "is_scored"},
                        "train_eval": {"row_mask": "all_after_warmup"},
                    },
                    "metrics": {"valid/wp": 0.5},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        cells[cell_id] = {
            "gpu": index,
            "gpu_uuid": f"GPU-{index}",
            "pid": 1000 + index,
            "start_ticks": 55,
            "process_group": 1000 + index,
            "identity_sha256": canonical_sha256(identity),
            "output": str(output),
            "log": str(launch_root / "logs" / f"{cell_id}.log"),
        }
    receipt = {
        "schema": monitor.RECEIPT_SCHEMA,
        "launch_id": "a" * 32,
        "panel_sha256": "b" * 64,
        "source_sha256": "c" * 64,
        "split_sha256": "d" * 64,
        "cells": cells,
    }
    receipt_path = launch_root / "launch_receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    monkeypatch.setattr(monitor, "owned_process_state", lambda _record: "exited")
    summary = monitor.refresh_summary(receipt_path)
    assert summary["status"] == "completed"
    assert summary["cells"]["g0-cell"]["state"] == "early_stopped"
    assert summary["official_valid_mask"] == "is_scored"
    first = runs_root / "g0-cell" / "monitor_evaluations.jsonl"
    event = json.loads(first.read_text(encoding="utf-8"))
    event["context"]["valid"]["row_mask"] = "all_after_warmup"
    first.write_text(json.dumps(event) + "\n", encoding="utf-8")
    summary = monitor.refresh_summary(receipt_path)
    assert summary["status"] == "failed"
    assert "official is_scored" in summary["cells"]["g0-cell"]["integrity_error"]


def test_monitor_waits_for_trainer_owned_output_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launch_root = tmp_path / "launch"
    (launch_root / "runs").mkdir(parents=True)
    cells = {
        f"g{index}-cell": {
            "gpu": index,
            "gpu_uuid": f"GPU-{index}",
            "pid": 1000 + index,
            "start_ticks": 100 + index,
            "process_group": 1000 + index,
            "output": str(launch_root / "runs" / f"g{index}-cell"),
            "log": str(launch_root / "logs" / f"g{index}-cell.log"),
        }
        for index in range(8)
    }
    receipt_path = launch_root / "launch_receipt.json"
    receipt_path.write_text(
        json.dumps({"schema": monitor.RECEIPT_SCHEMA, "cells": cells}), encoding="utf-8"
    )
    monkeypatch.setattr(monitor, "owned_process_state", lambda _record: "running")
    summary = monitor.refresh_summary(receipt_path)
    assert summary["status"] == "running"
    assert all(cell["state"] == "running" for cell in summary["cells"].values())
    assert not (launch_root / "runs" / "g0-cell").exists()


def test_fit_spawn_is_exact_eight_uuid_pinned_detached_processes(
    tmp_path: Path, approved_packet: tuple[Path, dict[str, Path]], monkeypatch: pytest.MonkeyPatch
) -> None:
    panel_path, profile_paths = approved_packet
    cells = json.loads(panel_path.read_text(encoding="utf-8"))["cells"]
    profiles: dict[str, Path] = {}
    for panel_cell in cells:
        path = profile_paths[panel_cell["id"]]
        profile = json.loads(path.read_text(encoding="utf-8"))
        profile["identity"]["runtime"] = {
            "precision": {"cublas_emulate_single_precision": None}
        }
        path.write_text(json.dumps(profile), encoding="utf-8")
        profiles[panel_cell["id"]] = path
    _, plans = load_panel_cells(panel_path, profiles)
    output_root = tmp_path / "owned-launch"
    output_root.mkdir()
    receipt_path = output_root / "launch_receipt.json"
    admitted_cells: dict[str, dict[str, Any]] = {}
    for cell in plans:
        admitted_cells[cell.cell_id] = {
            "gpu": cell.gpu,
            "gpu_uuid": f"GPU-{cell.gpu}",
            "cpu_affinity": list(cell.cpus),
            "numa_node": cell.numa_node,
            "config": str(cell.config_path),
            "config_file_sha256": cell.config_file_sha256,
            "profile_receipt": str(cell.profile_path),
            "profile_receipt_sha256": cell.profile_sha256,
            "identity_sha256": "e" * 64,
            "identity": {},
        }
    receipt = {
        "schema": monitor.RECEIPT_SCHEMA,
        "launch_id": "f" * 32,
        "output_root": str(output_root),
        "repo_root": str(REPO),
        "python": sys.executable,
        "data_root": str(tmp_path / "data"),
        "split_manifest": str(tmp_path / "split.json"),
        "panel": str(panel_path),
        "profiles": [f"{cell}={path}" for cell, path in profiles.items()],
        "cells": fit_processes.receipt_cells(output_root, {"cells": admitted_cells}),
    }
    atomic_write_json(receipt_path, receipt)
    launches: list[dict[str, Any]] = []
    applied_affinity: list[set[int]] = []

    def fake_popen(command: list[str], **kwargs: Any) -> SimpleNamespace:
        launches.append({"command": command, "kwargs": kwargs})
        return SimpleNamespace(pid=2000 + len(launches))

    monkeypatch.setattr(fit_processes.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        fit_processes.os,
        "sched_setaffinity",
        lambda _pid, cpus: applied_affinity.append(set(cpus)),
        raising=False,
    )
    monkeypatch.setattr(
        fit_processes, "_wait_process_identity", lambda pid: {"start_ticks": pid + 10, "process_group": pid}
    )
    children = fit_processes.start_fits(
        receipt_path,
        receipt,
        {"panel_sha256": sha256_file(panel_path), "cells": admitted_cells},
    )
    assert len(children) == 8
    assert len(launches) == 8
    assert not any((output_root / "runs" / cell.cell_id).exists() for cell in plans)
    for index, launch in enumerate(launches):
        kwargs = launch["kwargs"]
        assert kwargs["start_new_session"] is True
        assert kwargs["stdin"] is fit_processes.subprocess.DEVNULL
        assert "shell" not in kwargs
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == f"GPU-{index}"
        assert kwargs["env"]["OMP_NUM_THREADS"] == "8"
        assert kwargs["env"]["MKL_NUM_THREADS"] == "8"
        assert kwargs["env"]["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
        assert callable(kwargs["preexec_fn"])
        kwargs["preexec_fn"]()
    assert applied_affinity == [set(cell.cpus) for cell in plans]
    saved = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert {item["pid"] for item in saved["cells"].values()} == set(range(2001, 2009))
    assert {item["process_group"] for item in saved["cells"].values()} == set(range(2001, 2009))


def _sha256(path: Path) -> str:
    """Hash small fixture configs for the generated PANEL."""
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()
