"""Consumer-bound checks for the actual R5 producer and FP16 command policy."""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(os.environ.get("WNN_REPO_ROOT", Path(__file__).resolve().parents[7]))
sys.path.insert(0, str(HERE))

import native_gpu_fp16_contract as contract  # noqa: E402
import native_gpu_fp16_worker as worker  # noqa: E402

R5_OUTPUT = (
    REPO_ROOT
    / "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/"
    / "cloud_delivery/gpu_loss_pair_r5_export/terminal_collect_r1/attempt_001/outputs/out"
)
R4_WORKER = (
    REPO_ROOT
    / "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/"
    / "cloud_delivery/gpu_loss_pair_r4/code/scripts/loss_pair_gpu_worker.py"
)


def _actual_r4_worker():
    spec = importlib.util.spec_from_file_location("actual_r4_worker", R4_WORKER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_actual_r5_contract_and_every_shared_role_are_consumable():
    review = (
        REPO_ROOT
        / "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/"
        / "ROOT_GPU_R5_TERMINAL_REVIEW_R1.json"
    )
    value = contract.verify_r5_control_tree(R5_OUTPUT, review_path=review)
    assert value["cache"]["fit_train_count"] == 350
    assert value["fit"]["updates_per_arm"] == 220
    assert set(value["artifacts"]) == set(contract.REQUIRED_SHARED_ARTIFACTS)


def test_actual_r5_artifact_consumer_checks_checkpoints_traces_and_prediction_rows():
    result = contract.verify_r5_full_output_artifacts(R5_OUTPUT)
    assert result["training_contract_sha256"] == contract.R5_CONTRACT_SHA256
    assert result["arms"] == ["wmse", "huber_d1"]
    assert result["verified_arm_artifact_count"] >= 25
    for precision_state in result["model_states"].values():
        assert precision_state["model_parameter_dtype"] == "torch.float32"


def test_native_command_changes_only_autocast_precision_and_predeclared_scale():
    base = _actual_r4_worker()
    original = base._common_train_argv(
        "loss_pair_r3_wmse",
        "wmse",
        Path("/cache/manifest.json"),
        Path("/out/validation.json"),
        "a" * 64,
    )
    derived = contract.native_train_argv(original)
    assert derived[derived.index("--precision") + 1] == "fp16"
    assert derived[derived.index("--amp-init-scale") + 1] == "1024.0"
    assert [item for item in derived if item not in {"fp16", "1024.0", "--amp-init-scale"}] == [
        item for item in original if item != "fp32"
    ]
    assert derived.count("--fused-adam") == 1
    assert derived.count("--grad-stats") == 1


def test_command_policy_fails_closed_on_non_r4_or_duplicate_precision():
    with pytest.raises(contract.ContractError, match="frozen FP32 D01"):
        contract.native_train_argv(
            ["train_rnn.py", "--precision", "bf16", "--fused-adam", "--grad-stats"]
        )
    with pytest.raises(contract.ContractError, match="exactly one"):
        contract.native_train_argv(["train_rnn.py", "--precision", "fp32", "--precision", "fp32"])


def test_two_loss_worker_runs_one_shared_profile_gate_before_both_fits():
    base = _actual_r4_worker()
    worker_tree = ast.parse(R4_WORKER.read_text(encoding="utf-8"))
    run_worker = next(
        node
        for node in worker_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_worker"
    )
    gate_calls = [
        node
        for node in ast.walk(run_worker)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_run_device_shape_update_gate"
    ]
    fit_specs_assignments = [
        node
        for node in run_worker.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "fit_specs" for target in node.targets
        )
    ]

    assert base.run_worker.__name__ == "run_worker"
    assert len(gate_calls) == 1
    assert len(fit_specs_assignments) == 1
    assert gate_calls[0].lineno < fit_specs_assignments[0].lineno

    native_source = worker.__file__
    native_tree = ast.parse(Path(native_source).read_text(encoding="utf-8"))
    patch_worker = next(
        node
        for node in native_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_patch_worker"
    )
    gate_replacements = [
        node
        for node in ast.walk(patch_worker)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute) and target.attr == "_run_device_shape_update_gate"
            for target in node.targets
        )
    ]
    assert len(gate_replacements) == 1

    gate_log = Path("out/logs/device_shape_update_gate.log")
    lifecycle_log = Path("out/logs/native_fp16_lifecycle_gate.log")
    assert gate_log != lifecycle_log


def test_skipped_amp_updates_are_never_counted_as_throughput(tmp_path: Path):
    path = tmp_path / "update_trace.jsonl"
    rows = [
        {
            "attempt_index": index,
            "scaler_skipped": index == 1,
            "amp_scale_before": 1024.0,
            "amp_scale_after": 512.0 if index == 1 else 1024.0,
            "unclipped_global_grad_norm": 1.0,
            "optimizer_groups": [{"lr": 0.001, "beta1": 0.95}],
        }
        for index in range(3)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    summary = contract.summarize_update_trace(path, expected_attempts=3)
    assert summary["optimizer_attempts"] == 3
    assert summary["optimizer_completed_updates"] == 2
    assert summary["amp_skipped_updates"] == 1
    assert summary["throughput_update_count"] == 2
    assert summary["all_attempts_completed"] is False


def test_update_trace_requires_skip_and_finite_scaler_fields(tmp_path: Path):
    path = tmp_path / "update_trace.jsonl"
    path.write_text(
        json.dumps({"attempt_index": 0, "amp_scale_before": float("nan")}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(contract.ContractError, match="skip flag"):
        contract.summarize_update_trace(path, expected_attempts=1)


def _headroom_sample() -> dict:
    return {
        "working_disk_free_bytes": 20_000_000_000,
        "memory": {
            "mem_available_bytes": 10_000_000_000,
            "cgroup_current_bytes": 8_000_000_000,
            "cgroup_max_bytes": 16_000_000_000,
        },
        "gpus": [
            {
                "physical_index": index,
                "memory_total_mib": 15360,
                "memory_used_mib": 5000,
            }
            for index in (0, 1)
        ],
    }


def test_fit_headroom_uses_live_two_gpu_host_cgroup_and_disk_samples():
    evidence = contract.validate_fit_headroom(_headroom_sample(), {0, 1})
    assert evidence["state"] == "passed"
    assert evidence["cgroup_headroom_bytes"] == 8_000_000_000
    assert set(evidence["device_free_bytes"]) == {"0", "1"}


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda sample: sample["memory"].pop("mem_available_bytes"), "MemAvailable"),
        (lambda sample: sample["memory"].update(cgroup_max_bytes=9_000_000_000), "cgroup headroom"),
        (lambda sample: sample.update(working_disk_free_bytes=1), "disk free"),
        (lambda sample: sample.update(gpus=sample["gpus"][:1]), "inventory differs"),
        (lambda sample: sample["gpus"][0].update(memory_used_mib=15300), "less than 2 GB"),
    ],
)
def test_fit_headroom_defers_on_missing_or_exceeded_measurements(mutate, message):
    sample = _headroom_sample()
    mutate(sample)
    with pytest.raises(contract.ContractError, match=message):
        contract.validate_fit_headroom(sample, {0, 1})


def test_explicit_unbounded_cgroup_uses_the_host_reserve():
    sample = _headroom_sample()
    sample["memory"].pop("cgroup_max_bytes")
    sample["memory"]["cgroup_max_unbounded"] = True
    evidence = contract.validate_fit_headroom(sample, {0, 1})
    assert evidence["cgroup_headroom_bytes"] is None


def test_unknown_cgroup_limit_does_not_count_as_unbounded():
    sample = _headroom_sample()
    sample["memory"].pop("cgroup_max_bytes")
    with pytest.raises(contract.ContractError, match="missing or was not verified unbounded"):
        contract.validate_fit_headroom(sample, {0, 1})


def test_pending_root_admission_blocks_before_profile_or_fit(tmp_path: Path):
    admission_path = tmp_path / "root_admission.json"
    controls = tmp_path / "controls"
    controls.mkdir()
    admission_path.write_text(
        json.dumps(
            {
                "schema": "wnn.native-gpu-fp16.root-admission.v1",
                "state": "pending",
                "source_manifest_sha256": "a" * 64,
                "r5_training_contract_sha256": contract.R5_CONTRACT_SHA256,
                "r5_root_review_sha256": contract.R5_REVIEW_SHA256,
                "compute_precision": "fp16",
                "amp_initial_scale": contract.AMP_INITIAL_SCALE,
                "push_allowed": False,
                "profile_allowed": False,
                "conditional_fit_allowed": False,
                "training_contract_sha256": None,
            }
        ),
        encoding="utf-8",
    )
    bootstrap = {
        "native_root_admission_path": str(admission_path),
        "native_controls_root": str(controls),
        "native_source_manifest_sha256": "a" * 64,
    }
    with pytest.raises(contract.ContractError, match="root admission"):
        worker._validate_admission(bootstrap)


def test_checkpoint_consumer_requires_fp32_master_and_adamw_accumulators(tmp_path: Path):
    import torch

    model = {"weight": torch.ones((2, 2), dtype=torch.float32)}
    optimizer_state = {
        "state": {
            0: {
                "exp_avg": torch.ones((2, 2), dtype=torch.float32),
                "exp_avg_sq": torch.ones((2, 2), dtype=torch.float32),
                "step": torch.tensor(1.0),
            }
        }
    }
    checkpoint = tmp_path / "model.pt"
    state = tmp_path / "training_state.pt"
    torch.save({"model": model}, checkpoint)
    torch.save({"model": model, "optimizer": optimizer_state}, state)
    evidence = contract.verify_fp32_master_checkpoint(checkpoint, state)
    assert evidence["model_parameter_dtype"] == "torch.float32"
    assert evidence["optimizer_accumulator_dtype"] == "torch.float32"

    torch.save(
        {
            "model": model,
            "optimizer": {
                "state": {
                    0: {
                        "exp_avg": torch.ones((2, 2), dtype=torch.float16),
                        "exp_avg_sq": torch.ones((2, 2), dtype=torch.float16),
                    }
                }
            },
        },
        state,
    )
    with pytest.raises(contract.ContractError, match="accumulators are not FP32"):
        contract.verify_fp32_master_checkpoint(checkpoint, state)
