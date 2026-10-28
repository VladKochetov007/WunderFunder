from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import run_tl3_finite17 as driver
from scripts import tl3_finite17_contract as contract
from scripts import tl3_finite17_process as processes
from scripts.tl3_chronological_recipes import FINAL_WEIGHTS, PREFIX, W18_COEFFICIENTS, build_nodes
from scripts.tl3_finite17_contract import canonical_sha256, ensure_private_train_index
from wnn.distillation_cache import row_groups_sha256

R3_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "tl3_finite17"


def _plan(tmp_path: Path) -> dict[str, object]:
    nodes = build_nodes(Path("split.json"), tmp_path)
    for node in nodes:
        node["args_sha256"] = canonical_sha256(node["args"])
        node["run_directory"] = str(tmp_path / "runs" / str(node["run"]))
    return {
        "nodes": nodes,
        "split": {"split_manifest": str(tmp_path / "split.json")},
        "source_identity": {"training_source_sha256": "a" * 64},
        "final_runs": [PREFIX + name for name in FINAL_WEIGHTS],
        "w18_coefficients": {
            name: {"family": family, "coefficient": coefficient, "epoch": epoch}
            for name, (family, coefficient, epoch) in W18_COEFFICIENTS.items()
        },
    }


def test_finite_recipe_priority_starts_fin256_s4_before_other_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan(tmp_path)
    node_map = {node["id"]: node for node in plan["nodes"]}
    states = {
        name: {"state": "complete", "phase": "fit_complete", "source_sha256": "a" * 64}
        for name in node_map
    }
    states["fin256_s4"] = {"state": "not_started", "phase": "not_started"}
    status = {"nodes": states, "fits_used": 1, "tasks": {}}
    launched: list[dict[str, object]] = []
    monkeypatch.setattr(
        driver,
        "source_identity",
        lambda _root: {"training_source_sha256": "a" * 64},
    )
    monkeypatch.setattr(driver, "_launch_task", lambda **kwargs: launched.append(kwargs))

    started = driver._start_ready_gpu_task(
        root=tmp_path,
        python=Path("python"),
        state_dir=tmp_path / "state",
        state_path=tmp_path / "state/status.json",
        status=status,
        plan=plan,
        gpu=(0, "GPU-test"),
        cpu_affinity=tuple(range(8)),
        children={},
    )

    assert started
    assert len(launched) == 1
    assert launched[0]["task_id"] == "fit:fin256_s4"
    assert launched[0]["kind"] == "fit"
    assert launched[0]["cpu_affinity"] == tuple(range(8))


def test_child_environment_matches_admitted_cpu_and_loader_limits(tmp_path: Path) -> None:
    environment = driver._child_environment(tmp_path, "GPU-test")

    assert str(tmp_path) in environment["PYTHONPATH"]
    assert environment["WNN_THREADS"] == "4"
    assert environment["WNN_LOADER_WORKERS"] == "2"
    assert environment["WNN_LOADER_RAM_GB"] == "8"
    assert environment["OMP_NUM_THREADS"] == environment["MKL_NUM_THREADS"] == "4"


def test_failed_w18_ancestor_blocks_that_tail_but_preserves_independent_work(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    nodes = {node["id"]: node for node in plan["nodes"]}
    status = {
        "nodes": {
            name: {"state": "not_started", "phase": "not_started"} for name in nodes
        },
        "cache": {"state": "not_started"},
        "final_exports": {run: "not_started" for run in plan["final_runs"]},
        "score": {"state": "not_started"},
        "tasks": {},
    }
    status["nodes"]["fmu128_s3"]["state"] = "failed"

    driver._mark_blocked_stages(status, plan)

    assert status["nodes"]["w18final_distilled_s5"]["state"] == "blocked"
    assert status["cache"]["state"] == "blocked"
    assert status["nodes"]["ffull_s3"]["state"] == "blocked"
    assert driver._has_possible_work(status, plan)


def test_launch_failure_does_not_count_a_spawned_fit_twice(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    nodes = {node["id"]: node for node in plan["nodes"]}
    status = {
        "fits_used": 2,
        "gpu_seconds_completed": 0.0,
        "last_error": None,
        "nodes": {
            name: {"state": "not_started", "phase": "not_started"} for name in nodes
        },
    }
    task = {
        "task_id": "fit:fin256_s4",
        "kind": "fit",
        "node_id": "fin256_s4",
        "state": "launch_intent",
        "gpu_index": 0,
        "started_unix": 1.0,
        "popen_started": True,
    }

    driver._mark_launch_failed(status, task, "unbound child")

    assert status["fits_used"] == 2
    assert status["gpu_seconds_completed"] > 0
    assert status["nodes"]["fin256_s4"]["state"] == "failed"


def test_spawned_fit_group_binding_failure_reaps_child_and_counts_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []

    class FakePopen:
        pid = 4125364

        def __init__(self) -> None:
            self.wait_calls = 0

        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            events.append("terminate")

        def kill(self) -> None:
            events.append("kill")

        def wait(self, timeout: float | None = None) -> int:
            self.wait_calls += 1
            events.append("wait")
            if self.wait_calls == 1:
                raise subprocess.TimeoutExpired("fake-child", timeout)
            return -9

    child = FakePopen()
    monkeypatch.setattr(driver.subprocess, "Popen", lambda *_args, **_kwargs: child)
    monkeypatch.setattr(driver, "_write_status", lambda _path, _status: None)
    monkeypatch.setattr(driver, "_child_environment", lambda _root, _uuid: {})
    monkeypatch.setattr(
        driver,
        "proc_identity",
        lambda pid: {"start_ticks": 77, "process_group": pid + 1, "session": pid + 1},
    )
    status: dict[str, object] = {
        "tasks": {},
        "fits_used": 1,
        "gpu_seconds_completed": 0.0,
        "last_error": None,
        "nodes": {"fin256_s4": {"state": "not_started", "phase": "not_started"}},
    }
    children: dict[str, subprocess.Popen[bytes]] = {}

    driver._launch_task(
        root=tmp_path,
        state_dir=tmp_path / "state",
        state_path=tmp_path / "state/status.json",
        status=status,
        children=children,
        task_id="fit:fin256_s4",
        kind="fit",
        command=["python", "scripts/train_rnn.py", "--name", "tl3c_fin256_s4"],
        gpu=(0, "GPU-test"),
        node_id="fin256_s4",
        cpu_affinity=tuple(range(8)),
        expected_source_sha256="a" * 64,
    )

    task = status["tasks"]["fit:fin256_s4"]
    assert status["fits_used"] == 2
    assert task["state"] == "failed"
    assert task["pid"] == child.pid
    assert events == ["terminate", "wait", "kill", "wait"]
    assert "fit:fin256_s4" not in children
    assert status["nodes"]["fin256_s4"]["state"] == "failed"


def test_train_index_cache_contains_only_verified_row_group_ids(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    groups = np.asarray([9, 2, 15], dtype=np.int64)

    receipt = ensure_private_train_index(tmp_path, groups)

    with np.load(cache / "train_targets.npz", allow_pickle=False) as archive:
        assert archive.files == ["row_group"]
        np.testing.assert_array_equal(archive["row_group"], groups)
    assert receipt["fields"] == ["row_group"]
    assert receipt["targets_or_scoring_mask_read"] is False
    (cache / "valid_targets.npz").write_bytes(b"must not be used")
    with pytest.raises(driver.DriverError, match="valid target arrays"):
        ensure_private_train_index(tmp_path, groups)


def test_gpu_inventory_never_reuses_apps_protected_cards_or_gpu4() -> None:
    devices = {index: f"GPU-{index}" for index in range(6)}
    apps = {uuid: set() for uuid in devices.values()}
    apps["GPU-1"].add(991)

    free = processes.free_gpu_devices(devices, apps, {2}, {3})

    assert free == [(0, "GPU-0"), (5, "GPU-5")]


def test_galam_protection_requires_a_live_pid_with_matching_start_ticks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt_path = tmp_path / "galam.json"
    receipt = {
        "schema": "wnn.galam-l4-quality8-launch.v1",
        "cells": {
            f"g{index}": {
                "gpu": index,
                "gpu_uuid": f"GPU-{index}",
                "pid": 100 + index,
                "start_ticks": str(200 + index),
                "process_group": 100 + index,
                "state": "running",
            }
            for index in range(8)
        },
    }
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    def identity(pid: int) -> dict[str, int | str] | None:
        if pid == 102:
            return {
                "state": "S",
                "parent_pid": 1,
                "process_group": pid,
                "session": pid,
                "start_ticks": 202,
            }
        return None

    monkeypatch.setattr(processes, "proc_identity", identity)
    assert processes.protected_gpu_indices(receipt_path) == {2}


def test_gpu_cpu_placement_uses_the_receipt_crosswalk(tmp_path: Path) -> None:
    receipt_path = tmp_path / "galam.json"
    affinities = {
        0: [0, 1, 2, 3, 4, 5, 6, 7],
        1: [8, 9, 10, 11, 12, 13, 14, 15],
        2: [16, 17, 18, 19, 20, 21, 22, 23],
        3: [24, 25, 26, 27, 28, 29, 30, 31],
        4: [48, 49, 50, 51, 52, 53, 54, 55],
        5: [56, 57, 58, 59, 60, 61, 62, 63],
        6: [64, 65, 66, 67, 68, 69, 70, 71],
        7: [72, 73, 74, 75, 76, 77, 78, 79],
    }
    devices = {gpu: f"GPU-{gpu}" for gpu in affinities}
    receipt_path.write_text(
        json.dumps(
            {
                "schema": "wnn.galam-l4-quality8-launch.v1",
                "cells": {
                    f"g{gpu}": {
                        "gpu": gpu,
                        "gpu_uuid": devices[gpu],
                        "cpu_affinity": cores,
                    }
                    for gpu, cores in affinities.items()
                },
            }
        ),
        encoding="utf-8",
    )

    actual = processes.gpu_cpu_affinities(receipt_path, devices)

    assert actual[4] == tuple(affinities[4])
    assert set(actual) == set(devices)


def test_owned_timeout_kills_only_the_recorded_session_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = {"pid": 123, "process_group": 123, "session": 123, "start_ticks": "456"}
    monkeypatch.setattr(
        processes,
        "proc_identity",
        lambda _pid: {
            "state": "S",
            "parent_pid": 1,
            "process_group": 123,
            "session": 123,
            "start_ticks": 456,
        },
    )
    group_members = iter(([123, 124], [123, 124], []))
    monkeypatch.setattr(processes, "process_group_members", lambda *_args: next(group_members))
    signaled: list[tuple[int, int]] = []

    stopped = processes.terminate_owned_group(
        record,
        grace_seconds=0,
        killpg=lambda pgid, sig: signaled.append((pgid, sig)),
        monotonic=lambda: 0.0,
        sleep=lambda _duration: None,
    )

    assert stopped
    assert [pgid for pgid, _signal in signaled] == [123, 123]


def test_pid_reuse_is_rejected_before_any_signal(monkeypatch: pytest.MonkeyPatch) -> None:
    record = {"pid": 123, "process_group": 123, "session": 123, "start_ticks": "456"}
    monkeypatch.setattr(
        processes,
        "proc_identity",
        lambda _pid: {
            "state": "S",
            "parent_pid": 1,
            "process_group": 123,
            "session": 123,
            "start_ticks": 999,
        },
    )
    signaled: list[int] = []

    with pytest.raises(driver.DriverError, match="unowned process group"):
        processes.terminate_owned_group(record, killpg=lambda pgid, _sig: signaled.append(pgid))
    assert signaled == []


def test_exited_adopted_fit_is_reconciled_through_artifact_verifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan(tmp_path)
    plan["sha256"] = "e" * 64
    started = 1000.0
    adoption = {
        "pid": 4125363,
        "process_group": 4125363,
        "session": 4125363,
        "start_ticks": "2060336421",
        "gpu_index": 0,
        "gpu_uuid": "GPU-test",
        "started_unix": started,
        "source_sha256": "c" * 64,
        "args_sha256": "d" * 64,
        "argv": ["python", "scripts/train_rnn.py"],
    }
    status = driver._new_status(plan, adoption)
    state_path = tmp_path / "state" / "status.json"
    state_path.parent.mkdir()
    verified: list[tuple[str, int, str]] = []
    monkeypatch.setattr(driver, "owned_group_state", lambda _record: "exited")
    monkeypatch.setattr(
        driver,
        "verify_run_output",
        lambda _root, node, _split, source: verified.append(
            (str(node["id"]), int(node["endpoint"]), str(source))
        )
        or {"run": node["run"], "epoch": node["endpoint"]},
    )

    driver._poll_tasks(tmp_path, state_path, status, plan, object(), {}, started + 10.0)

    task = status["tasks"]["fit:fin256_s3"]
    node = status["nodes"]["fin256_s3"]
    assert verified == [("fin256_s3", 3, "c" * 64)]
    assert task["state"] == "complete"
    assert task["return_code"] == 0
    assert node["state"] == "running"
    assert node["phase"] == "export_pending"


def test_actual_r3_adoption_receipts_validate_without_current_hash_recipe(
    tmp_path: Path,
) -> None:
    for name in (
        "adopted-source.identity.json",
        "source-r3.identity.json",
        "fin256_s3.adoption.json",
    ):
        shutil.copyfile(R3_FIXTURE_DIR / name, tmp_path / name)
    adoption = json.loads((tmp_path / "fin256_s3.adoption.json").read_text(encoding="utf-8"))

    source_root, files = contract._verify_legacy_source_documents(adoption, tmp_path)

    assert source_root.endswith("/source-r3")
    assert len(files) == 120
    assert files["scripts/train_rnn.py"] == (
        "4b64186743e7c9535abc544b63139ff5ad6b7a1e5d9bc6f72feb689d451fb5ab"
    )


def test_legacy_source_file_check_hashes_the_bounded_code_closure(tmp_path: Path) -> None:
    source_root = tmp_path / "source-r3"
    source_root.mkdir()
    source_file = source_root / "src/wnn/training/legacy.py"
    source_file.parent.mkdir(parents=True)
    source_file.write_bytes(b"pinned old R3 implementation\n")
    identity = hashlib.sha256(source_file.read_bytes()).hexdigest()

    contract._verify_legacy_source_files(
        str(source_root), {"src/wnn/training/legacy.py": identity}, tmp_path
    )
    source_file.write_bytes(b"modified implementation\n")
    with pytest.raises(driver.DriverError, match="source file bytes changed"):
        contract._verify_legacy_source_files(
            str(source_root), {"src/wnn/training/legacy.py": identity}, tmp_path
        )


def test_w18_cache_verifies_exact_sources_membership_and_file_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train_groups = [3, 8]
    eval_groups = [30]
    split = SimpleNamespace(train_groups=train_groups, valid_groups=(), evaluation_groups=eval_groups)
    monkeypatch.setattr(driver, "load_campaign_split", lambda *_args: split)
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    with (cache_root / "train_targets.npz").open("wb") as handle:
        np.savez(handle, row_group=np.asarray(train_groups, dtype=np.int64))
    output = cache_root / "w18"
    output.mkdir()
    plan = {
        "root": str(tmp_path),
        "w18_cache": str(output),
        "split": {
            "split_manifest": str(tmp_path / "split.json"),
            "split_manifest_sha256": "b" * 64,
        },
        "w18_coefficients": {
            name: {"family": family, "coefficient": coefficient, "epoch": epoch}
            for name, (family, coefficient, epoch) in W18_COEFFICIENTS.items()
        },
    }
    files: dict[str, dict[str, object]] = {}
    widths = {"main_source": 2, "main_normalized": 2, "main_homogeneous": 2, "families": 8}
    for target, width in widths.items():
        for split_name, groups in (("train", train_groups), ("valid", [])):
            name = f"{target}_{split_name}"
            array = np.zeros((len(groups), 20000, width), dtype=np.float32)
            path = output / f"{name}.npy"
            np.save(path, array, allow_pickle=False)
            files[name] = {
                "path": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
                "shape": list(array.shape),
                "dtype": "float32",
                "row_groups": groups,
                "row_groups_sha256": row_groups_sha256(groups),
            }
    expected_sources = [
        {
            "family": value["family"],
            "run": PREFIX + name,
            "epoch": value["epoch"],
            "coefficient": value["coefficient"],
        }
        for name, value in plan["w18_coefficients"].items()
    ]
    metadata = {
        "split_manifest_sha256": "b" * 64,
        "source_coefficients": expected_sources,
        "coefficient_sum": 0.8125,
        "output_profile": "all",
        "fit_groups": {"train": train_groups, "valid": []},
        "evaluation_valid_groups": eval_groups,
        "stored_targets": list(widths),
        "files": files,
    }
    (output / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    receipt = driver.verify_distillation_cache(plan)

    assert receipt["file_count"] == 8
    metadata["evaluation_valid_groups"] = []
    (output / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(driver.DriverError, match="metadata differs"):
        driver.verify_distillation_cache(plan)
