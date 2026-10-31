from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from wnn import multistage_search_successor_prepare as successor_prepare
from wnn.multistage_search import canonical_json_sha256, file_sha256
from wnn.multistage_search_adopt_fit_validate import validate_terminal_owner
from wnn.multistage_search_stage3_pair import finalize_pair
from wnn.multistage_search_successor_parent import qualified_parent
from wnn.training.sequence_config import load_config


def test_stage2_parent_requires_exact_direct_fit_identity(tmp_path: Path) -> None:
    """Accept a fixed-final label-only parent and reject mismatched mask ancestry."""
    trial_dir = tmp_path / "parent"
    fit_dir = trial_dir / "fit"
    fit_dir.mkdir(parents=True)
    raw_config = {
        "epochs": 4,
        "kd_weight": 0.0,
        "hint_weight": 0.0,
        "precision_mode": "fp32",
        "lr": 0.001,
        "model": {"architecture": "gru", "hint_dim": 0},
    }
    config_path = trial_dir / "config.json"
    config_path.write_text(json.dumps(raw_config), encoding="utf-8")
    config = load_config(config_path)
    data = {
        "teacher": None,
        "initializer": None,
        "training_mask": "all_after_warmup",
        "evaluation_role": "control",
        "evaluation_source": "train",
    }
    source_files = {"src/entry.py": "a" * 64}
    identity = {
        "config_sha256": canonical_json_sha256(config),
        "data_sha256": canonical_json_sha256(data),
        "data": data,
        "source_files": source_files,
        "source_sha256": canonical_json_sha256(source_files),
    }
    checkpoint = fit_dir / "model.pt"
    torch.save(
        {
            "checkpoint_schema": "wnn.sequence-fit.v1",
            "config": config,
            "data": data,
            "provenance": {"identity": identity},
            "completed_epochs": 4,
            "epoch": 3,
        },
        checkpoint,
    )
    result_path = trial_dir / "trial_result.json"
    result_path.write_text(
        json.dumps(
            {
                "status": "COMPLETE",
                "stage": 1,
                "family": "gru",
                "fit_output": str(fit_dir),
                "fit_model_sha256": file_sha256(checkpoint),
                "config_sha256": file_sha256(config_path),
                "objective": {"evaluation": {"scope": "inner_train_proxy"}},
            }
        ),
        encoding="utf-8",
    )

    parent, loaded_checkpoint, loaded_config = qualified_parent(result_path, allowed={1})
    assert parent["status"] == "COMPLETE"
    assert loaded_checkpoint == checkpoint
    assert loaded_config == config

    data["training_mask"] = "is_scored"
    identity["data"] = data
    identity["data_sha256"] = canonical_json_sha256(data)
    torch.save(
        {
            "checkpoint_schema": "wnn.sequence-fit.v1",
            "config": config,
            "data": data,
            "provenance": {"identity": identity},
            "completed_epochs": 4,
            "epoch": 3,
        },
        checkpoint,
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["fit_model_sha256"] = file_sha256(checkpoint)
    result_path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="unqualified fit/config endpoint"):
        qualified_parent(result_path, allowed={1})


def test_stage3_pair_report_records_only_the_matched_inner_proxy(tmp_path: Path) -> None:
    """Pair report requires matched configs, fit/control scope and emits real plot artifacts."""
    run_dir = tmp_path / "run"
    pair_id = "pair-000001-000002"
    pair_dir = run_dir / "stage3" / "pairs" / pair_id
    pair_dir.mkdir(parents=True)
    variants = [
        {"trial_number": 1, "pair_role": "label_only_control"},
        {"trial_number": 2, "pair_role": "kd_student"},
    ]
    (pair_dir / "pair.json").write_text(
        json.dumps(
            {
                "schema": "wnn.multistage-stage3-pair.v1",
                "pair_id": pair_id,
                "teacher_pool_id": "pool-a",
                "teacher_cache_manifest_sha256": "a" * 64,
                "variants": variants,
            }
        ),
        encoding="utf-8",
    )
    config = {
        "epochs": 2,
        "seed": 1301,
        "precision_mode": "fp32",
        "model": {"architecture": "gru", "hint_dim": 0},
    }
    values = ((1, "label_only_control", 0.0, 0.4), (2, "kd_student", 0.25, 0.42))
    for number, role, kd_weight, score in values:
        trial_dir = run_dir / "stage3" / "trials" / f"{number:06d}"
        trial_dir.mkdir(parents=True)
        (trial_dir / "config.json").write_text(
            json.dumps(config | {"kd_weight": kd_weight}), encoding="utf-8"
        )
        result = {
            "status": "COMPLETE",
            "stage": 3,
            "pair_id": pair_id,
            "pair_role": role,
            "trial_number": number,
            "fit_model_sha256": f"{number:x}" * 64,
            "config_sha256": f"{number + 1:x}" * 64,
            "objective": {
                "split_sha256": "b" * 64,
                "primary_split_sha256": "c" * 64,
                "objective": {"value": score},
                "evaluation": {
                    "scope": "inner_train_proxy",
                    "source": "train",
                    "is_scored_used": False,
                    "row_mask": "all_after_warmup AND row_index>=99",
                },
            },
        }
        (trial_dir / "trial_result.json").write_text(json.dumps(result), encoding="utf-8")

    result = finalize_pair(run_dir, {"pair_id": pair_id})
    assert result["status"] == "complete"
    pair_result = json.loads(Path(str(result["result_path"])).read_text(encoding="utf-8"))
    assert pair_result["delta_wp_kd_minus_control"] == pytest.approx(0.02)
    assert pair_result["primary_valid_used"] is False
    assert {Path(item["path"]).suffix for item in result["plots"]} == {".png", ".svg"}
    assert finalize_pair(run_dir, {"pair_id": pair_id})["status"] == "already_complete"


def test_completed_fit_adoption_preserves_unknown_outer_exit_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Accept complete native artifacts only with an explicit unknown wrapper exit code."""
    import hashlib

    from wnn import multistage_search_runtime

    monkeypatch.setattr(multistage_search_runtime, "process_identity", lambda _pid: None)
    attempt_id = "preexisting-stage1-gru-b256-r5"
    output_dir = tmp_path / "fit"
    output_dir.mkdir()
    command = ["python", "scripts/train_sequence.py"]
    owner = {
        "schema": "wnn.multistage-external-worker.v1",
        "attempt_id": attempt_id,
        "stage": 1,
        "mode": "fit",
        "pid": 93142,
        "birth": "birth-token",
        "pgid": 93142,
        "command": command,
        "command_sha256": hashlib.sha256(
            json.dumps(command, separators=(",", ":")).encode()
        ).hexdigest(),
        "output_dir": str(output_dir),
        "started_at_epoch": 100.0,
        "deadline_epoch": 7300.0,
    }
    owner_path = tmp_path / "owner.json"
    owner_path.write_text(json.dumps(owner), encoding="utf-8")
    deadline = {"reserve_start_epoch": 8000.0}
    attempt = {"attempt_id": attempt_id, "stage": 1, "output_dir": str(output_dir)}
    terminal = {
        "schema": "wnn.multistage-fit-terminal.v1",
        "status": "artifacts_complete_exit_unknown",
        "attempt_id": attempt_id,
        "exit_code": None,
        "exit_observation": "outer_wrapper_returncode_not_persisted",
        "failure_logs_empty": True,
        "owner_receipt_sha256": file_sha256(owner_path),
        "finished_at_epoch": 200.0,
        "output_dir": str(output_dir),
    }

    validate_terminal_owner(attempt_id, attempt, owner, terminal, owner_path, output_dir, deadline)

    terminal["exit_code"] = 0
    with pytest.raises(ValueError, match="terminal fit receipt"):
        validate_terminal_owner(
            attempt_id, attempt, owner, terminal, owner_path, output_dir, deadline
        )


def test_stage2_anchor_multiplier_is_enqueued_as_a_tpe_trial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A planned low-LR anchor uses normal Optuna suggestion and config materialization."""

    class Trial:
        number = 4

        def __init__(self, params: dict[str, float]) -> None:
            self.params = params
            self.attrs: dict[str, object] = {}

        def suggest_categorical(self, name: str, choices: list[float]) -> float:
            value = self.params.get(name, choices[0])
            assert value in choices
            return value

        def set_user_attr(self, name: str, value: object) -> None:
            self.attrs[name] = value

    class Study:
        def __init__(self) -> None:
            self.enqueued: list[dict[str, float]] = []
            self.trial = Trial({})

        def get_trials(self, *, deepcopy: bool) -> list[object]:
            assert deepcopy is False
            return []

        def enqueue_trial(self, params: dict[str, float], **_kwargs: object) -> None:
            self.enqueued.append(params)
            self.trial.params = params

        def ask(self) -> Trial:
            return self.trial

    study = Study()
    parent = {
        "family": "gru",
        "objective": {
            "split_sha256": "a" * 64,
            "primary_split_sha256": "b" * 64,
        },
    }
    config = {
        "lr": 0.001,
        "epochs": 4,
        "model": {"architecture": "gru", "hidden": 256, "layers": 2},
    }
    admission = {
        "data": {
            "auxiliary_manifest": {"sha256": "a" * 64},
            "primary_split_sha256": "b" * 64,
        },
        "stage2": {"max_trials": 4},
    }
    captured: dict[str, object] = {}
    checkpoint_path = tmp_path / "parent.pt"
    checkpoint_path.write_bytes(b"fixed-final-parent")
    monkeypatch.setattr(successor_prepare, "_campaign_inputs", lambda _run: ({}, admission))
    monkeypatch.setattr(
        successor_prepare,
        "qualified_parent",
        lambda _path, *, allowed: (parent, checkpoint_path, config),
    )
    monkeypatch.setattr(successor_prepare, "parent_identity_sha", lambda _path: "c" * 64)
    monkeypatch.setattr(successor_prepare, "open_study", lambda *_args, **_kwargs: study)

    def capture_trial(_run, _manifest, _admission, _trial, definition):
        captured.update(definition)
        return {"trial_dir": str(tmp_path / "trial")}

    monkeypatch.setattr(successor_prepare, "_write_trial", capture_trial)
    parent_result_path = tmp_path / "parent-result.json"
    parent_result_path.write_text("{}", encoding="utf-8")
    result = successor_prepare.prepare_stage2_trial(tmp_path, parent_result_path, lr_multiplier=0.1)

    assert result["trial_dir"] == str(tmp_path / "trial")
    assert study.enqueued == [{"lr_multiplier": 0.1}]
    assert captured["params"]["lr"] == pytest.approx(0.0001)
    assert captured["config"]["epochs"] == 2
