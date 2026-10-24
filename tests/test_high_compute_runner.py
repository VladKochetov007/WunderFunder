"""Receiver-only checks for explicit R4 dispatch and teacher-cache prerequisites."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

PACKET = (
    Path(__file__).resolve().parents[1]
    / "research/experiment_drafts/b300-transformer-distillation-runtime/high_compute_r4"
)


@pytest.fixture
def dispatcher(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(PACKET))
    spec = importlib.util.spec_from_file_location("r4_test_dispatch", PACKET / "dispatch.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cell() -> dict[str, object]:
    return {
        "id": "B_lr3e-4_f0",
        "kind": "teacher",
        "reused": False,
        "run_dir": "runs/B_lr3e-4_f0",
        "config": "cells/B_lr3e-4_f0/config.json",
        "profile_one": "cells/B_lr3e-4_f0/profile_one.json",
        "profile_peak": "cells/B_lr3e-4_f0/profile_peak.json",
        "split_manifest": "/bound/inner-f0.json",
        "teacher_cache": "runs/B_lr1e-4_f0/teacher-cache",
    }


def test_reused_endpoint_cannot_launch(dispatcher: ModuleType, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="reused"):
        dispatcher.execute_cell("fit", tmp_path, {}, {"reused": True}, tmp_path / "checks.json")


def test_profile_one_uses_inner_split_and_one_rank(
    dispatcher: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(dispatcher, "execute", commands.append)
    cell = _cell()
    dispatcher.execute_cell(
        "profile-one", tmp_path, {"data_root": "/data"}, cell, tmp_path / "checks"
    )
    assert len(commands) == 1
    command = commands[0]
    assert "--nproc-per-node=1" in command
    assert command[command.index("--config") + 1].endswith("profile_one.json")
    assert command[command.index("--split-manifest") + 1] == "/bound/inner-f0.json"
    assert command[command.index("--profile-updates") + 1] == "12"


def test_eight_rank_profile_requires_one_rank_receipt(
    dispatcher: ModuleType, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="profile-one"):
        dispatcher.execute_cell(
            "profile-eight", tmp_path, {"data_root": "/data"}, _cell(), tmp_path / "checks"
        )


def test_recorded_failure_cannot_silently_restart(dispatcher: ModuleType, tmp_path: Path) -> None:
    cell = _cell()
    run = tmp_path / str(cell["run_dir"])
    run.mkdir(parents=True)
    (run / "failure.json").write_text("{}")
    with pytest.raises(ValueError, match="frozen failure"):
        dispatcher.execute_cell("fit", tmp_path, {"data_root": "/data"}, cell, tmp_path / "checks")


def test_fit_and_resume_require_admitted_profile_arguments(
    dispatcher: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(dispatcher, "execute", commands.append)
    monkeypatch.setattr(
        dispatcher, "verify_admission_binding", lambda run, config: None, raising=False
    )
    for action in ("fit", "resume"):
        dispatcher.execute_cell(
            action, tmp_path, {"data_root": "/data"}, _cell(), tmp_path / "checks"
        )
    assert all("--nproc-per-node=8" in command for command in commands)
    assert all("--profile-receipt" in command for command in commands)
    assert "--resume" not in commands[0] and "--resume" in commands[1]


def test_kd_parent_guard_runs_before_profile(
    dispatcher: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(dispatcher, "execute", commands.append)

    def reject(*args: object) -> None:
        raise ValueError("missing actual export parity")

    monkeypatch.setattr(dispatcher, "parent_cache_guards", reject)
    cell = _cell() | {"teacher_cache": "caches/mean", "kind": "student"}
    with pytest.raises(ValueError, match="export parity"):
        dispatcher.execute_cell(
            "profile-one",
            tmp_path,
            {"data_root": "/data", "runtime_source_sha256": "x"},
            cell,
            tmp_path / "checks",
        )
    assert commands == []


def test_cache_parity_exports_one_rank_before_comparison(
    dispatcher: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(dispatcher, "execute", commands.append)
    cell = _cell()
    directory = tmp_path / str(cell["run_dir"]) / "teacher-cache"
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text("{}")
    dispatcher.execute_cell(
        "cache-parity", tmp_path, {"data_root": "/data"}, cell, tmp_path / "checks"
    )
    assert len(commands) == 2
    assert "--nproc-per-node=1" in commands[0]
    assert "--membership" in commands[0] and "fit" in commands[0]
    assert commands[1][1].endswith("cache_parity.py")


def test_parent_cache_requires_unchanged_successful_parity(
    dispatcher: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import catalog
    import materialize
    import runner_checks

    parent = tmp_path / "runs/teacher"
    (parent / "fit").mkdir(parents=True)
    (parent / "teacher-cache").mkdir()
    raw_config = catalog.teacher_base(0, 301001)
    config_path = tmp_path / "cells/G_candidate_f0_s301001/config.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps(raw_config))
    normalized_config = catalog.normalized_runtime_config(raw_config)
    config_sha256 = catalog.canonical_sha256(raw_config)
    producer_cell = {
        "id": "G_candidate_f0_s301001",
        "kind": "teacher",
        "fold": 0,
        "seed": 301001,
        "config": "cells/G_candidate_f0_s301001/config.json",
        "config_sha256": config_sha256,
        "run_dir": "runs/teacher",
        "teacher_cache": "runs/teacher/teacher-cache",
    }

    def get_cell(root: Path, stage: str, cell_id: str) -> dict[str, object]:
        assert root == tmp_path and stage == "G" and cell_id == producer_cell["id"]
        return producer_cell

    monkeypatch.setattr(materialize, "get_cell", get_cell)
    identity = {
        "source_sha256": "source",
        "config_sha256": catalog.canonical_sha256(normalized_config),
    }
    (parent / "fit/terminal_status.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "model_sha256": "a" * 64,
                "identity": identity,
            }
        )
    )
    manifest = parent / "teacher-cache/manifest.json"
    manifest_value = {
        "complete": True,
        "split_sha256": "split",
        "producer": {
            "checkpoint_sha256": "a" * 64,
            "config": json.loads(json.dumps(normalized_config)),
            "provenance": {"identity": identity},
        },
    }
    manifest.write_text(json.dumps(manifest_value))
    one_manifest = parent / "teacher-cache-one/manifest.json"
    one_manifest.parent.mkdir()
    one_manifest.write_text(manifest.read_text())
    parity = {
        "schema": "wnn.sequence-cache-parity.v1",
        "passed": True,
        "checkpoint_sha256": "a" * 64,
        "split_sha256": "split",
        "manifest_sha256": {
            "eight": runner_checks.file_hash(manifest),
            "one": runner_checks.file_hash(one_manifest),
        },
    }
    parity_path = parent / "cache-parity.json"
    parity_path.write_text(json.dumps(parity))
    cell = {
        "cache_inputs": ["runs/teacher/teacher-cache"],
        "parent_run_dirs": ["runs/teacher"],
        "cache_producers": [
            {
                "path": "runs/teacher/teacher-cache",
                "producer_run_dir": "runs/teacher",
                "producer_cell_id": producer_cell["id"],
                "producer_config_sha256": config_sha256,
                "fold": 0,
                "seed": 301001,
                "variant": "candidate",
            }
        ],
        "split_sha256": "split",
    }
    runner_checks.parent_cache_guards(tmp_path, cell, "source")
    manifest_value["producer"]["config"]["lr"] = 0.001
    manifest.write_text(json.dumps(manifest_value))
    with pytest.raises(ValueError, match="producer config"):
        runner_checks.parent_cache_guards(tmp_path, cell, "source")
    manifest_value["producer"]["config"] = json.loads(json.dumps(normalized_config))
    manifest.write_text(json.dumps(manifest_value))
    cell["cache_producers"][0]["producer_config_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="frozen G cell"):
        runner_checks.parent_cache_guards(tmp_path, cell, "source")
    cell["cache_producers"][0]["producer_config_sha256"] = config_sha256
    parity["manifest_sha256"]["eight"] = runner_checks.file_hash(manifest)
    parity_path.write_text(json.dumps(parity))
    one_manifest.write_text(one_manifest.read_text() + "\n")
    with pytest.raises(ValueError, match="identities"):
        runner_checks.parent_cache_guards(tmp_path, cell, "source")
    one_manifest.write_text(manifest.read_text())
    parity_path.write_text(json.dumps(parity | {"passed": False}))
    with pytest.raises(ValueError, match="identities"):
        runner_checks.parent_cache_guards(tmp_path, cell, "source")
    parity_path.write_text(json.dumps(parity))
    manifest.write_text(manifest.read_text() + "\n")
    with pytest.raises(ValueError, match="identities"):
        runner_checks.parent_cache_guards(tmp_path, cell, "source")


def test_mean_cache_rejects_a_different_clean_parent_composition(
    dispatcher: ModuleType, tmp_path: Path
) -> None:
    import runner_checks

    inputs = [f"cache-{index}" for index in range(3)]
    parents = []
    for name in inputs:
        directory = tmp_path / name
        directory.mkdir()
        manifest = directory / "manifest.json"
        manifest.write_text(json.dumps({"name": name}))
        parents.append({"manifest_sha256": runner_checks.file_hash(manifest)})
    merged = tmp_path / "mean"
    merged.mkdir()
    manifest = {
        "complete": True,
        "split_sha256": "split",
        "producer": {"kind": "mean_of_three_fp32", "parents": parents},
    }
    path = merged / "manifest.json"
    path.write_text(json.dumps(manifest))
    cell = {"cache_inputs": inputs, "teacher_cache": "mean", "split_sha256": "split"}
    runner_checks.merged_cache_guard(tmp_path, cell)
    manifest["producer"]["parents"] = list(reversed(parents))
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="composition"):
        runner_checks.merged_cache_guard(tmp_path, cell)


def test_receiver_check_subprocess_environment_disables_cublas_emulation(
    dispatcher: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os

    import runner_checks

    name = "CUBLAS_EMULATE_SINGLE_PRECISION"
    monkeypatch.setenv(name, "1")
    environment = runner_checks.receiver_check_environment()
    assert environment[name] == "0"
    assert os.environ[name] == "1"

    received: dict[str, object] = {}
    monkeypatch.setattr(
        runner_checks.subprocess,
        "run",
        lambda command, *, cwd, check, env: received.update(
            command=command, cwd=cwd, check=check, env=env
        ),
    )
    runner_checks.execute(["python", "-V"], env=environment)
    assert received["env"][name] == "0"
    assert received["env"] is not environment


def test_fit_admission_binding_requires_unchanged_config_and_profile(
    dispatcher: ModuleType, tmp_path: Path
) -> None:
    import runner_checks

    config_path = tmp_path / "student.json"
    config_path.write_text((PACKET / "templates/student_gru512.json").read_text())
    raw_config = runner_checks.read(config_path)
    normalized = runner_checks.normalized_runtime_config(raw_config)
    run = tmp_path / "runs/student"
    run.mkdir(parents=True)
    normalized_path = run / "admission-normalized-config.json"
    normalized_path.write_text(json.dumps(normalized, indent=2) + "\n")
    approved_path = run / "approved_profile.json"
    approved_path.write_text(
        json.dumps(
            {
                "approved": True,
                "identity": {"config_sha256": runner_checks.canonical_sha256(normalized)},
            }
        )
    )
    binding_path = run / "r4_admission_binding.json"
    binding_path.write_text(
        json.dumps(
            {
                "schema": "wnn.high-compute-r4-admission-binding.v1",
                "raw_config_path": str(config_path.resolve()),
                "raw_config_sha256": runner_checks.file_hash(config_path),
                "normalized_config_path": str(normalized_path.resolve()),
                "normalized_config_sha256": runner_checks.file_hash(normalized_path),
                "approved_profile_path": str(approved_path.resolve()),
                "approved_profile_sha256": runner_checks.file_hash(approved_path),
            }
        )
    )

    runner_checks.verify_admission_binding(run, config_path)
    approved_path.write_text(approved_path.read_text() + "\n")
    with pytest.raises(ValueError, match="evidence changed"):
        runner_checks.verify_admission_binding(run, config_path)
