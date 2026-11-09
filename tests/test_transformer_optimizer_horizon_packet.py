from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

_PACKET = (
    Path(__file__).parents[1] / "research/experiment_drafts/transformer-optimizer-horizon-20261108"
)


def _modules():
    sys.path.insert(0, str(_PACKET / "receiving"))
    return importlib.import_module("build_profile_plan"), importlib.import_module("admission")


def test_exact_six_configs_and_cache_free_profile_plan() -> None:
    builder, _ = _modules()
    CELLS, SPLIT_SHA256, build_plan = builder.CELLS, builder.SPLIT_SHA256, builder.build_plan
    plan = build_plan()
    assert plan["new_scientific_fit_cap"] == 6
    assert plan["profile_cap"] == 6
    assert len(plan["cells"]) == len(CELLS) == 6
    assert plan["split_sha256"] == SPLIT_SHA256
    for item in plan["cells"]:
        command = item["profile_command"]
        assert "--nproc-per-node=2" in command
        assert command[-2:] == ["--profile-updates", "50"]
        assert not {
            "--teacher-cache",
            "--reference-cache",
            "--monitor-reference-root",
        }.intersection(command)
        config = json.loads((_PACKET / "configs" / item["config"]).read_text())
        assert config["objective_mode"] == "standalone_labels_wmse_v2"
        assert config["batch_size"] == 128
        assert config["required_world_size"] == 2
        assert config["precision_mode"] == "bf16_bf16x9"
        assert config["optimizer_update_diagnostics_every"] == 50
        assert config["gradient_horizon_rows"] == config["chunk_rows"]
        assert config["kd_weight"] == config["hint_weight"] == 0
        assert config["lr_head_factor"] == config["lr_body_factor"] == 1.0


def test_pair_schedule_has_no_concurrent_gpu_overlap() -> None:
    plan = _modules()[0].build_plan()
    cells = {item["cell"]: item for item in plan["cells"]}
    assert [len(wave) for wave in plan["profile_wave_order"]] == [4, 2]
    for wave in plan["profile_wave_order"]:
        pairs = [cells[name]["pair"] for name in wave]
        assert len(pairs) == len(set(pairs))
    flattened = [name for wave in plan["profile_wave_order"] for name in wave]
    assert set(flattened) == set(cells)
    assert len(flattened) == 6


def test_adamw_recipe_and_context_factors_are_explicit() -> None:
    plan = _modules()[0].build_plan()
    adamw = next(item for item in plan["cells"] if item["arm"] == "C1")
    config = json.loads((_PACKET / "configs" / adamw["config"]).read_text())
    assert config["optimizer"] == "adamw"
    assert config["lr_role_factors"] == {"adamw": 1}
    assert config["lr_head_factor"] == config["lr_body_factor"] == 1
    assert config["chunk_rows"] == config["gradient_horizon_rows"] == 256
    assert config["accumulate_chunks"] == 4


def test_root_authority_is_finite_and_has_collection_reserve() -> None:
    authority = _modules()[1].authority
    value = {
        "approved_by": "ROOT",
        "fit_cap": 6,
        "profile_cap": 6,
        "first_profile_epoch": 1794168764.8507051,
        "fit_deadline_epoch": 1794168764.8507051 + 7 * 3600,
        "deadline_epoch": 1794168764.8507051 + 8 * 3600,
        "collection_reserve_seconds": 3600,
        "source_commit": "5d73bc2b10ae20e607e0d4b4533da85e76e27bca",
    }
    authority(value)
    with pytest.raises(ValueError):
        authority(value | {"deadline_epoch": 1794168764.8507051 + 9 * 3600})
    with pytest.raises(ValueError, match="authority"):
        authority(value | {"first_profile_epoch": 1000})


def test_software_gate_binds_root_runtime_cpu_and_all_six_profiles(
    tmp_path, monkeypatch
) -> None:
    admission = _modules()[1]
    runtime = tmp_path / "runtime"
    (runtime / "software").mkdir(parents=True)
    (runtime / "approvals").mkdir()
    cells = []
    proofs = {}
    for index in range(6):
        name = f"cell-{index}"
        profile = tmp_path / "profiles" / name / "profile_receipt.json"
        profile.parent.mkdir(parents=True)
        profile.write_text(json.dumps({"status": "profile_complete"}), encoding="utf-8")
        approval = runtime / "approvals" / f"{name}.json"
        approval.write_text(json.dumps({"approved": True}), encoding="utf-8")
        cells.append({"cell": name, "profile_receipt_path": str(profile)})
        proofs[name] = {
            "profile_receipt_sha256": admission.digest(profile),
            "approval_sha256": admission.digest(approval),
        }
    plan = {
        "source_commit": "5d73bc2b10ae20e607e0d4b4533da85e76e27bca",
        "source_manifest_sha256": "a" * 64,
        "expected_training_source_sha256": "b" * 64,
        "cells": cells,
    }
    qualification = {
        "schema": "wnn.b300-panel-software-qualification.v1",
        "approved_by": "ROOT",
        "status": "passed",
        "source_commit": plan["source_commit"],
        "source_manifest_sha256": plan["source_manifest_sha256"],
        "split_sha256": admission.SPLIT_SHA,
        "runtime": {
            "python_version": "3.12.14",
            "torch_version": "2.14.0+cu130",
            "cuda_version": "13.0",
            "nccl_status": "observed",
            "world_size": 2,
            "device_model": "NVIDIA_B300",
        },
        "cpu_tests": {"status": "passed", "receipt_sha256": "c" * 64},
        "native_profiles": proofs,
    }
    path = runtime / "software" / "COMPLETE.json"
    path.write_text(json.dumps(qualification), encoding="utf-8")
    calls = []
    monkeypatch.setattr(
        admission,
        "approved",
        lambda root, cell, source: calls.append((root, cell["cell"], source)) or True,
    )
    assert admission.software(runtime, plan)
    assert len(calls) == 6
    assert not hasattr(admission, "retirement")
    with pytest.raises(ValueError, match="exactly the six frozen profile cells"):
        admission.software(
            runtime,
            plan | {"cells": cells[:5]},
        )
    qualification["cpu_tests"]["receipt_sha256"] = "not-a-digest"
    path.write_text(json.dumps(qualification), encoding="utf-8")
    with pytest.raises(ValueError, match="runtime/CPU qualification"):
        admission.software(runtime, plan)


def test_primary_score_rejects_checkpoint_hash_mismatch() -> None:
    compare = importlib.import_module("compare")
    terminal = {"model_sha256": "a" * 64}
    producer = {"checkpoint_sha256": "a" * 64}
    compare.validate_checkpoint_binding(terminal, producer, "a" * 64)
    with pytest.raises(ValueError, match="not bound"):
        compare.validate_checkpoint_binding(terminal, producer, "b" * 64)


def test_phase_commands_keep_standalone_fit_and_evaluation_cache_free(
    tmp_path, monkeypatch
) -> None:
    _modules()
    plan = {
        "source_root": str(tmp_path / "source-r2"),
        "runtime": str(tmp_path),
        "python": "/env/bin/python",
        "data_root": "/data/project",
        "split_manifest": "/data/chronological_primary.json",
        "cells": [
            {
                "cell": "C0",
                "config_path": "/runtime/configs/C0.json",
                "approved_profile_receipt_path": "/runtime/approvals/C0.json",
            }
        ],
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    calls = []
    phase = importlib.import_module("phase")
    monkeypatch.setattr(phase.subprocess, "run", lambda command, **_: calls.append(command))
    monkeypatch.setattr(
        sys, "argv", ["phase.py", "--plan", str(plan_path), "--action", "fit", "--cell", "C0"]
    )
    phase.main()
    monkeypatch.setattr(
        sys, "argv", ["phase.py", "--plan", str(plan_path), "--action", "evaluate", "--cell", "C0"]
    )
    phase.main()
    assert len(calls) == 3
    fit = calls[0]
    evaluation = calls[1]
    assert fit[1] == str(tmp_path / "source-r2/scripts/observe_sequence.py")
    assert fit[fit.index("--output") + 1] == str(tmp_path / "runs/C0/fit")
    assert fit[fit.index("--interval") + 1] == "10"
    assert fit[fit.index("--max-seconds") + 1] == "25200"
    trainer = fit[fit.index("--") + 1 :]
    assert trainer[trainer.index("--profile-receipt") + 1] == "/runtime/approvals/C0.json"
    assert trainer[1] == "-m"
    for command in (fit, evaluation):
        assert not {
            "--reference-cache",
            "--teacher-cache",
            "--monitor-reference-root",
            "--blend-alpha",
        }.intersection(command)
    assert evaluation[evaluation.index("--membership") + 1] == "evaluation"
