from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import tl3_checkpoint_eval as checkpoint_eval
from scripts import tl3_dashboard_bridge as bridge


def _proc_stat(pid: int, parent_pid: int, start_ticks: int) -> str:
    fields = ["S", str(parent_pid), "1", *(["0"] * 16), str(start_ticks)]
    return f"{pid} (python) {' '.join(fields)}\n"


def test_parse_legacy_log_keeps_full_training_objective_separate() -> None:
    records = bridge.parse_legacy_log(
        [
            "ep 0 batch 1/4 loss 0.25 lr 0.001 seq/s 8.0",
            'GRAD_STATS {"epoch":0,"steps":4,"total_q05_q50_q95":[1,2,3]}',
            'VALID {"epoch":0,"weighted_pearson":0.3,"global_t0":0.2,"global_t1":0.1}',
            "SELECTED_FIXED 0",
        ]
    )

    assert records["progress"][0] == {
        "epoch": 0,
        "batch": 1,
        "total": 4,
        "loss": 0.25,
        "lr": 0.001,
        "rate": 8.0,
    }
    assert records["valid"][0]["weighted_pearson"] == 0.3
    assert records["selected_fixed"] == 0
    event = bridge._event(
        "train_step",
        1,
        0.25,
        {"train/objective_batch": 0.25},
        {"objective_definition": "combined training scalar; not validation WMSE"},
    )
    assert "train_eval/wmse" not in event["metrics"]


def test_wmse_uses_clipped_targets_absolute_weights_and_explicit_mask() -> None:
    totals = [0.0, 0.0]
    checkpoint_eval._accumulate_wmse(
        np.array([[[3.0, -1.0], [2.0, 0.0]]], dtype=np.float32),
        np.array([[[2.0, 0.0], [1.0, 7.0]]], dtype=np.float32),
        np.array([[True, False]]),
        totals,
    )

    assert totals == [1.0, 3.0]


def test_trainer_must_be_descendant_and_start_ticks_must_match(tmp_path: Path) -> None:
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    (proc_root / "41").mkdir()
    (proc_root / "42").mkdir()
    (proc_root / "41" / "stat").write_text(_proc_stat(41, 1, 900), encoding="utf-8")
    (proc_root / "42" / "stat").write_text(_proc_stat(42, 41, 901), encoding="utf-8")
    expected = ["python", "-u", "/frozen/scripts/train_rnn.py", "--name", "tl3c"]
    (proc_root / "42" / "cmdline").write_bytes("\0".join(expected).encode() + b"\0")
    binding = bridge.RunBinding(
        run="tl3c",
        data_root=tmp_path,
        source_root=tmp_path,
        output=tmp_path,
        log=tmp_path / "log",
        split_manifest=tmp_path / "split.json",
        gpu_uuid="GPU-1234",
        parent_pid=41,
        parent_start_ticks=900,
        command=("timeout", "21600", *expected),
        receipt_sha256="a" * 64,
        adoption_sha256="d" * 64,
        source_identity_file_sha256="b" * 64,
        actual_training_source_sha256="e" * 64,
        training_source_identity_record_sha256="f" * 64,
        source_identity_file=tmp_path / "source.identity.json",
        training_source_identity_file=tmp_path / "adopted-source.identity.json",
        split_sha256="c" * 64,
    )

    assert bridge.find_owned_trainer(binding, proc_root) == (
        42,
        {"state": "S", "parent_pid": 41, "process_group": 1, "start_ticks": 901},
    )
    binding = bridge.RunBinding(**{**binding.__dict__, "parent_start_ticks": 899})
    with pytest.raises(bridge.BridgeError, match="PID was reused"):
        bridge.find_owned_trainer(binding, proc_root)


def test_orphaned_exact_trainer_prevents_false_terminal_status(tmp_path: Path) -> None:
    proc_root = tmp_path / "proc"
    (proc_root / "42").mkdir(parents=True)
    (proc_root / "42" / "stat").write_text(_proc_stat(42, 1, 901), encoding="utf-8")
    command = ("timeout", "21600", "python", "/frozen/scripts/train_rnn.py", "--name", "tl3c")
    (proc_root / "42" / "cmdline").write_bytes("\0".join(command[2:]).encode() + b"\0")
    binding = bridge.RunBinding(
        run="tl3c",
        data_root=tmp_path,
        source_root=tmp_path,
        output=tmp_path,
        log=tmp_path / "log",
        split_manifest=tmp_path / "split.json",
        gpu_uuid="GPU-1234",
        parent_pid=41,
        parent_start_ticks=900,
        command=command,
        receipt_sha256="a" * 64,
        adoption_sha256="d" * 64,
        source_identity_file_sha256="b" * 64,
        actual_training_source_sha256="e" * 64,
        training_source_identity_record_sha256="f" * 64,
        source_identity_file=tmp_path / "source.identity.json",
        training_source_identity_file=tmp_path / "adopted-source.identity.json",
        split_sha256="c" * 64,
    )

    assert not bridge.is_timeout_parent_live(binding, proc_root)
    assert bridge.has_exact_trainer_process(binding, proc_root)


def test_verified_resources_require_exact_pid_and_gpu_uuid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proc_root = tmp_path / "proc"
    process_dir = proc_root / "77"
    process_dir.mkdir(parents=True)
    identity = {"state": "S", "parent_pid": 1, "process_group": 1, "start_ticks": 123}
    (process_dir / "stat").write_text(_proc_stat(77, 1, 123), encoding="utf-8")
    (process_dir / "status").write_text("Name:\tpython\nVmRSS:\t2048 kB\n", encoding="utf-8")
    answers = iter(
        [
            "GPU-1234, 4096, 8192, 80, 125.5\n",
            "77, GPU-1234, 512\n",
        ]
    )
    monkeypatch.setattr(bridge, "_run_nvidia_smi", lambda _args: next(answers))

    metrics = bridge.query_verified_resources(77, identity, "GPU-1234", proc_root)

    assert metrics == {
        "gpu/process_memory_mib": 512.0,
        "gpu/device_memory_used_mib": 4096.0,
        "gpu/device_memory_free_mib": 8192.0,
        "gpu/utilization_pct": 80.0,
        "gpu/power_watts": 125.5,
        "cpu/process_rss_mib": 2.0,
    }


def test_train_and_valid_evaluation_masks_are_distinct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    from wnn import gmetric
    from wnn.training import evaluation

    captured: list[np.ndarray] = []

    class CapturingAccumulator:
        def add(self, _target: np.ndarray, _prediction: np.ndarray, mask: np.ndarray) -> None:
            captured.append(mask.copy())

        def result(self) -> dict[str, float]:
            return {"wp": 0.1, "t0": 0.2, "t1": 0.3}

    monkeypatch.setattr(gmetric, "GlobalAccumulator", CapturingAccumulator)
    monkeypatch.setattr(
        evaluation,
        "predict",
        lambda _model, features, _chunk, _device, *, environment: np.zeros(
            (features.shape[0], features.shape[1], 2), dtype=np.float32
        ),
    )
    targets = torch.ones((1, 3, 2), dtype=torch.float32)
    features = torch.zeros((1, 3, 112), dtype=torch.float32)
    train_scored = torch.zeros((1, 3), dtype=torch.bool)
    valid_scored = torch.tensor([[True, False, True]])

    class Loader(list):
        dataset = [object()]

    environment = SimpleNamespace(warmup=1)
    checkpoint_eval._evaluate_cohort(
        object(),
        Loader([(features, targets, train_scored)]),
        None,
        environment,
        row_mask="all_after_warmup",
        group_order_sha256="1" * 64,
    )
    checkpoint_eval._evaluate_cohort(
        object(),
        Loader([(features, targets, valid_scored)]),
        None,
        environment,
        row_mask="is_scored",
        group_order_sha256="2" * 64,
    )

    assert captured[0].tolist() == [[False, True, True]]
    assert captured[1].tolist() == [[False, False, True]]


def _make_bridge_fixture(tmp_path: Path) -> tuple[Path, Path, str, str]:
    data_root = tmp_path / "data"
    source_root = tmp_path / "source"
    (source_root / "scripts").mkdir(parents=True)
    (source_root / "src" / "wnn" / "training").mkdir(parents=True)
    (source_root / "src" / "wnn").mkdir(exist_ok=True)
    (source_root / "scripts" / "train_rnn.py").write_text(
        "# frozen source marker\n", encoding="utf-8"
    )
    (source_root / "src" / "wnn" / "training" / "runner.py").write_text(
        "# trainer source marker\n", encoding="utf-8"
    )
    (source_root / "src" / "wnn" / "nn.py").write_text(
        "# extra full-source file\n", encoding="utf-8"
    )
    (source_root / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    (data_root / "artifacts" / "runs" / "tl3c_fin256_s3").mkdir(parents=True)
    (data_root / "logs").mkdir()
    run_dir = data_root / "artifacts" / "runs" / "tl3c_fin256_s3"
    checkpoint = run_dir / "model_ep0.pt"
    checkpoint.write_bytes(b"checkpoint-fixture")
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    train_groups = list(range(1, 33))
    valid_groups = [101, 102]
    manifest = {
        "schema": "wnn.chronological-primary.v1",
        "fit": {"train": train_groups, "valid": []},
        "evaluation": {"valid": valid_groups},
    }
    manifest_path = tmp_path / "chronological_primary.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    split_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    source_git = "a" * 40
    trainer_paths = (
        "scripts/train_rnn.py",
        "pyproject.toml",
        "src/wnn/training/runner.py",
    )
    trainer_files = {
        relative: hashlib.sha256((source_root / relative).read_bytes()).hexdigest()
        for relative in trainer_paths
    }
    trainer_sha = bridge._canonical_sha256(trainer_files)
    training_source_identity = {
        "schema": "wnn.training-source.v1",
        "sha256": trainer_sha,
        "files": trainer_files,
        "missing_dependency_files": ["uv.lock"],
        "runtime_dependencies": {"attrs": "26.1.0"},
    }
    source_identity_file = source_root.parent / "source.identity.json"
    full_source_files = {
        name: digest for name, digest in trainer_files.items() if name.endswith(".py")
    }
    full_source_files["src/wnn/nn.py"] = hashlib.sha256(
        (source_root / "src" / "wnn" / "nn.py").read_bytes()
    ).hexdigest()
    source_identity_file.write_text(
        json.dumps({"git": source_git, "files": full_source_files}), encoding="utf-8"
    )
    source_identity_file_sha = hashlib.sha256(source_identity_file.read_bytes()).hexdigest()
    args = {
        "name": "tl3c_fin256_s3",
        "train_source": "train",
        "eval_source": "dev",
        "full_valid": True,
        "train_seqs": 32,
        "no_eval_during_fit": False,
        "checkpoint_policy": "fixed",
        "checkpoint_epoch": 0,
        "epochs": 1,
        "precision": "fp32",
        "init": None,
        "resume_state": None,
        "propensity": None,
        "distill": None,
    }
    provenance = {
        "trainer_implementation": training_source_identity,
        "split_manifest_sha256": split_sha,
        "fit_row_groups": {"train": train_groups, "valid": []},
        "evaluation_row_groups": valid_groups,
        "args": args,
        "checkpoint_sha256": {"model_ep0.pt": checkpoint_sha},
    }
    (run_dir / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    (data_root / "logs" / "tl3c_fin256_s3.log").write_text(
        "ep 0 batch 0/1 loss 0.75 lr 0.001 seq/s 1.25\n"
        'GRAD_STATS {"epoch":0,"steps":1,"clip":1.0,"clip_rate":0.0}\n'
        'VALID {"epoch":0,"weighted_pearson":0.25,"global_t0":0.1,"global_t1":0.2,'
        '"update_diagnostics":{"cumulative":{"optimizer_attempts":1}},'
        '"fit_summary":{"objective_mean":0.75}}\n'
        "SELECTED_FIXED 0\n",
        encoding="utf-8",
    )
    command = [
        "timeout",
        "21600",
        "python",
        str(source_root / "scripts" / "train_rnn.py"),
        "--name",
        "tl3c_fin256_s3",
        "--split-manifest",
        str(manifest_path),
    ]
    script_index = command.index(str(source_root / "scripts" / "train_rnn.py"))
    adoption = {
        "schema": "wnn.tl3-fit-adoption.v1",
        "node_id": "fin256_s3",
        "run": "tl3c_fin256_s3",
        "source_sha256": trainer_sha,
        "source_identity_file_sha256": source_identity_file_sha,
        "source_root": str(source_root.resolve()),
        "argv": command,
        "training_args": command[script_index + 1 :],
        "pid": 999999991,
        "start_ticks": "900",
    }
    (tmp_path / "fin256_s3.adoption.json").write_text(json.dumps(adoption), encoding="utf-8")
    (tmp_path / "adopted-source.identity.json").write_text(
        json.dumps(training_source_identity), encoding="utf-8"
    )
    receipt = {
        "run": "tl3c_fin256_s3",
        "command": command,
        "environment": {
            "CUDA_VISIBLE_DEVICES": "GPU-1234",
            "WNN_PROJECT_ROOT": str(data_root),
        },
        "pid": 999999991,
        "start_ticks": "900",
        "source_identity_sha256": source_identity_file_sha,
        "source_git": source_git,
        "split_sha256": split_sha,
        "limits": {"wall_seconds": 21600},
        "started_utc": "2026-10-28T16:50:48Z",
    }
    receipt_path = tmp_path / "fin256_s3.launch.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    return receipt_path, manifest_path, checkpoint_sha, split_sha


def test_bridge_import_is_idempotent_and_labels_both_masks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from wnn.dashboard.schema import parse_event

    receipt_path, _manifest_path, checkpoint_sha, _split_sha = _make_bridge_fixture(tmp_path)
    runs_root = tmp_path / "dashboard-runs"
    runs_root.mkdir()
    monkeypatch.setattr(bridge, "is_timeout_parent_live", lambda *_args: False)
    monkeypatch.setattr(bridge, "has_exact_trainer_process", lambda *_args: False)
    pending = bridge.bridge_once(receipt_path, runs_root, evaluate_missing=False)
    assert pending["status"] == "fit_complete_pending_evaluation"
    view = runs_root / "tl3c_fin256_s3"
    initial_events = [
        json.loads(line) for line in (view / "monitor_events.jsonl").read_text().splitlines()
    ]
    valid_log_events = [
        event
        for event in initial_events
        if event["kind"] == "diagnostics" and "valid/score" in event["metrics"]
    ]
    assert len(valid_log_events) == 1
    assert valid_log_events[0]["metrics"]["valid/score"] == 0.25
    assert valid_log_events[0]["metrics"]["valid/score/t0"] == 0.1
    assert valid_log_events[0]["metrics"]["valid/score/t1"] == 0.2

    binding, _receipt = bridge.load_binding(receipt_path)
    run_identity = bridge._run_identity(binding)
    train_groups = list(range(1, 33))
    valid_groups = [101, 102]
    evaluation = {
        "schema": "wnn.tl3-frozen-checkpoint-evaluation.v1",
        "run_identity_sha256": run_identity,
        "source_identity_file_sha256": binding.source_identity_file_sha256,
        "actual_training_source_sha256": binding.actual_training_source_sha256,
        "completed_epoch": 1,
        "source_epoch_index": 0,
        "checkpoint_sha256": checkpoint_sha,
        "train_eval": {
            "row_mask": "all_after_warmup",
            "cohort_count": 32,
            "cohort_group_order_sha256": bridge._canonical_sha256(train_groups),
            "metrics": {"wp": 0.1, "t0": 0.2, "t1": 0.3, "wmse": 0.4},
        },
        "valid": {
            "row_mask": "is_scored",
            "cohort_count": 2,
            "cohort_group_order_sha256": bridge._canonical_sha256(valid_groups),
            "metrics": {"wp": 0.25, "t0": 0.1, "t1": 0.2, "wmse": 0.5},
        },
    }
    monkeypatch.setattr(bridge, "_run_evaluator", lambda *_args: evaluation)
    completed = bridge.bridge_once(receipt_path, runs_root)
    repeated = bridge.bridge_once(receipt_path, runs_root)
    assert completed["status"] == "completed"
    assert repeated["status"] == "completed"

    events = [json.loads(line) for line in (view / "monitor_events.jsonl").read_text().splitlines()]
    evaluations = [event for event in events if event["kind"] == "evaluation"]
    assert len(evaluations) == 1
    event = evaluations[0]
    assert event["metrics"]["valid/wp"] == 0.25
    assert event["metrics"]["valid/wmse"] == 0.5
    assert event["metrics"]["train_eval/wmse"] == 0.4
    assert event["metrics"]["train/objective_window_mean"] == 0.75
    assert event["metrics"]["train_eval/loss"] == 0.4
    assert event["metrics"]["valid/loss"] == 0.5
    assert event["metrics"]["train_eval/score"] == 0.1
    assert event["metrics"]["valid/score"] == 0.25
    assert event["metrics"]["train_eval/score/t0"] == 0.2
    assert event["metrics"]["valid/score/t0"] == 0.1
    assert event["context"]["train_eval"]["row_mask"] == "all_after_warmup"
    assert event["context"]["valid"]["row_mask"] == "is_scored"
    assert "observed_at" not in event["context"]
    training_events = [record for record in events if record["kind"] == "train_step"]
    assert training_events[0]["metrics"]["train/loss"] == 0.75
    assert parse_event(json.dumps(event).encode()) is not None
    assert sum(1 for record in (view / "monitor_evaluations.jsonl").read_text().splitlines()) == 1
    assert (view / "bridge_receipt.json").is_file()


def test_epoch_zero_full_valid_is_monitorable_before_fixed_epoch_three(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt_path, _manifest_path, checkpoint_sha, _split_sha = _make_bridge_fixture(tmp_path)
    binding, _receipt = bridge.load_binding(receipt_path)
    provenance_path = binding.output / "provenance.json"
    provenance = bridge._read_json(provenance_path)
    provenance["args"]["epochs"] = 4
    provenance["args"]["checkpoint_epoch"] = 3
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    runs_root = tmp_path / "dashboard-runs"
    runs_root.mkdir()
    monkeypatch.setattr(bridge, "is_timeout_parent_live", lambda *_args: False)
    monkeypatch.setattr(bridge, "has_exact_trainer_process", lambda *_args: False)
    run_identity = bridge._run_identity(binding)
    evaluation = {
        "schema": "wnn.tl3-frozen-checkpoint-evaluation.v1",
        "run_identity_sha256": run_identity,
        "source_identity_file_sha256": binding.source_identity_file_sha256,
        "actual_training_source_sha256": binding.actual_training_source_sha256,
        "completed_epoch": 1,
        "source_epoch_index": 0,
        "checkpoint_sha256": checkpoint_sha,
        "train_eval": {
            "row_mask": "all_after_warmup",
            "cohort_count": 32,
            "cohort_group_order_sha256": bridge._canonical_sha256(list(range(1, 33))),
            "metrics": {"wp": 0.1, "t0": 0.2, "t1": 0.3, "wmse": 0.4},
        },
        "valid": {
            "row_mask": "is_scored",
            "cohort_count": 2,
            "cohort_group_order_sha256": bridge._canonical_sha256([101, 102]),
            "metrics": {"wp": 0.25, "t0": 0.1, "t1": 0.2, "wmse": 0.5},
        },
    }
    monkeypatch.setattr(bridge, "_run_evaluator", lambda *_args: evaluation)

    result = bridge.bridge_once(receipt_path, runs_root)

    assert result["status"] == "running"
    events = [
        json.loads(line)
        for line in (runs_root / "tl3c_fin256_s3" / "monitor_events.jsonl").read_text().splitlines()
    ]
    evaluation_events = [event for event in events if event["kind"] == "evaluation"]
    assert len(evaluation_events) == 1
    assert evaluation_events[0]["context"]["completed_epoch"] == 1
    assert evaluation_events[0]["metrics"]["valid/wp"] == 0.25


def test_bridge_rejects_worker_mask_or_logged_valid_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt_path, _manifest_path, checkpoint_sha, _split_sha = _make_bridge_fixture(tmp_path)
    runs_root = tmp_path / "dashboard-runs"
    runs_root.mkdir()
    binding, _receipt = bridge.load_binding(receipt_path)
    evaluation = {
        "schema": "wnn.tl3-frozen-checkpoint-evaluation.v1",
        "run_identity_sha256": bridge._run_identity(binding),
        "source_identity_file_sha256": binding.source_identity_file_sha256,
        "actual_training_source_sha256": binding.actual_training_source_sha256,
        "completed_epoch": 1,
        "source_epoch_index": 0,
        "checkpoint_sha256": checkpoint_sha,
        "train_eval": {
            "row_mask": "is_scored",
            "cohort_count": 32,
            "cohort_group_order_sha256": "x",
            "metrics": {"wp": 0.1, "t0": 0.2, "t1": 0.3, "wmse": 0.4},
        },
        "valid": {
            "row_mask": "is_scored",
            "cohort_count": 2,
            "cohort_group_order_sha256": bridge._canonical_sha256([101, 102]),
            "metrics": {"wp": 0.25, "t0": 0.1, "t1": 0.2, "wmse": 0.5},
        },
    }
    monkeypatch.setattr(bridge, "_run_evaluator", lambda *_args: evaluation)

    with pytest.raises(bridge.BridgeError, match="cohort order or explicit row-mask"):
        bridge.bridge_once(receipt_path, runs_root)


def test_provenance_requires_full_train_membership_without_duplicates(tmp_path: Path) -> None:
    receipt_path, _manifest_path, _checkpoint_sha, _split_sha = _make_bridge_fixture(tmp_path)
    binding, _receipt = bridge.load_binding(receipt_path)
    provenance = bridge._read_json(binding.output / "provenance.json")
    bridge.validate_run_provenance(binding, provenance)
    provenance["fit_row_groups"]["train"][-1] = provenance["fit_row_groups"]["train"][0]

    with pytest.raises(bridge.BridgeError, match="train fit membership"):
        bridge.validate_run_provenance(binding, provenance)


def test_launch_and_trainer_source_hash_scopes_are_independent(tmp_path: Path) -> None:
    receipt_path, _manifest_path, _checkpoint_sha, _split_sha = _make_bridge_fixture(tmp_path)
    binding, receipt = bridge.load_binding(receipt_path)
    provenance = bridge._read_json(binding.output / "provenance.json")

    assert binding.source_identity_file_sha256 == receipt["source_identity_sha256"]
    assert binding.actual_training_source_sha256 == provenance["trainer_implementation"]["sha256"]
    assert binding.actual_training_source_sha256 != binding.source_identity_file_sha256
    assert (
        bridge._canonical_sha256(provenance["trainer_implementation"])
        == binding.training_source_identity_record_sha256
    )


def test_evaluator_cli_receives_both_source_hash_scopes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt_path, _manifest_path, _checkpoint_sha, _split_sha = _make_bridge_fixture(tmp_path)
    binding, _receipt = bridge.load_binding(receipt_path)
    invocation: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        invocation["command"] = command
        invocation["kwargs"] = kwargs
        return SimpleNamespace(stdout="{}", stderr="")

    monkeypatch.setattr(bridge.subprocess, "run", fake_run)
    assert bridge._run_evaluator(binding, "a" * 64, 0, 16, 30) == {}

    command = invocation["command"]
    assert isinstance(command, list)
    parsed = checkpoint_eval._parser().parse_args(command[2:])
    assert parsed.source_identity_file_sha256 == binding.source_identity_file_sha256
    assert parsed.training_source_sha256 == binding.actual_training_source_sha256


def test_manifest_managed_vfold_requires_the_single_full_valid_fold(tmp_path: Path) -> None:
    receipt_path, _manifest_path, _checkpoint_sha, _split_sha = _make_bridge_fixture(tmp_path)
    binding, _receipt = bridge.load_binding(receipt_path)
    provenance = bridge._read_json(binding.output / "provenance.json")
    provenance["args"].update({"eval_source": "vfold", "valid_fold": 0, "n_valid_folds": 1})

    bridge.validate_run_provenance(binding, provenance)

    # These frozen ancestry declarations are resolved by the lineage driver and
    # teacher binder; the bridge still enforces the same exact fit/eval groups.
    provenance["args"].update(
        {"init": "frozen_initializer.pt", "distill": "fit_only_teacher", "raw_only": True}
    )
    bridge.validate_run_provenance(binding, provenance)

    provenance["args"]["n_valid_folds"] = 2
    with pytest.raises(bridge.BridgeError, match="single full official-valid fold"):
        bridge.validate_run_provenance(binding, provenance)


def test_torch_uuid_spelling_is_normalized_only_for_physical_identity() -> None:
    assert checkpoint_eval._physical_gpu_uuid("222adc44-ecd3-a645-cf7e-ddda9d495f7a") == (
        "GPU-222adc44-ecd3-a645-cf7e-ddda9d495f7a"
    )
    assert checkpoint_eval._physical_gpu_uuid("GPU-222adc44-ecd3-a645-cf7e-ddda9d495f7a") == (
        "GPU-222adc44-ecd3-a645-cf7e-ddda9d495f7a"
    )
