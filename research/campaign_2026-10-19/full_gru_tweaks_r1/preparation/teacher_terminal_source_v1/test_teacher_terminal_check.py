"""Synthetic tests for strict teacher terminal evidence validation."""

from __future__ import annotations

import numpy as np
import pytest
import teacher_terminal_check as check
import torch
from propensity_terminal_support import run_loader_self_test


def _counters() -> tuple[list[dict[str, object]], dict[str, object], dict[str, object]]:
    diagnostics = []
    history = []
    for epoch in range(3):
        epoch_counts = {
            "optimizer_attempts": check.PER_EPOCH,
            "optimizer_completed_updates": check.PER_EPOCH,
            "completed_batches": check.BATCHES,
            "amp_skipped_updates": 0,
        }
        cumulative = {
            "optimizer_attempts": check.PER_EPOCH * (epoch + 1),
            "optimizer_completed_updates": check.PER_EPOCH * (epoch + 1),
            "completed_batches": check.BATCHES * (epoch + 1),
            "amp_skipped_updates": 0,
        }
        record = {
            "epoch_index": epoch,
            "epoch": epoch_counts,
            "cumulative": cumulative,
            "grad_stats": {"steps": check.PER_EPOCH, "nonfinite_steps": 0},
        }
        diagnostics.append(record)
        history.append(
            {
                "epoch": epoch,
                "evaluation": "withheld",
                "update_diagnostics": {
                    key: value for key, value in record.items() if key != "epoch_index"
                },
            }
        )
    total = diagnostics[-1]["cumulative"]
    progress = {
        "history": history,
        "update_diagnostics": diagnostics,
        "step_count": check.UPDATES,
        "cumulative_updates": total,
    }
    provenance = {"update_diagnostics": diagnostics, "cumulative_updates": total}
    return history, progress, provenance


def _optimizer_fixture() -> dict[str, object]:
    groups = []
    states = {}
    next_id = 0
    for group_index, shapes in enumerate(check.OPTIMIZER_GROUP_PARAMETER_SHAPES):
        muon = group_index > 0
        groups.append(
            {
                "params": list(range(next_id, next_id + len(shapes))),
                "muon": muon,
                "blocks": (3 if group_index == 1 else 1),
                "muon_fp32_ortho": muon,
                "weight_decay": 0.01,
                "eps": 1e-8,
                "ns_steps": 5,
                "rms": 0.2,
                "betas": (0.9, 0.999),
                "lr": 1e-5,
            }
        )
        for parameter_id, shape in zip(range(next_id, next_id + len(shapes)), shapes, strict=True):
            if group_index == 0:
                states[parameter_id] = {
                    "step": check.UPDATES,
                    "exp_avg": torch.zeros(shape, dtype=torch.float32),
                    "exp_avg_sq": torch.zeros(shape, dtype=torch.float32),
                }
            else:
                states[parameter_id] = {"momentum_buffer": torch.zeros(shape, dtype=torch.float32)}
        next_id += len(shapes)
    return {"param_groups": groups, "state": states}


def _model_fixture() -> dict[str, torch.Tensor]:
    model = {
        name: torch.zeros(shape, dtype=torch.float32) for name, shape in check.MODEL_SHAPES.items()
    }
    model["std"].fill_(1.0)
    return model


def test_native_exit_requires_exact_integer_evidence_and_admitted_invocation() -> None:
    launch = {"unit": check.UNIT, "invocation_id": check.INVOCATION}
    evidence = {
        **launch,
        "run_name": check.RUN,
        "main_pid": 0,
        "active_state": "active",
        "sub_state": "exited",
        "exec_main_code": 1,
        "exec_main_status": 0,
    }
    check._check_exit(evidence, launch)
    with pytest.raises(ValueError, match="native unit state"):
        check._check_exit({**evidence, "sub_state": "running"}, launch)
    with pytest.raises(ValueError, match="exec_main_status"):
        check._check_exit({**evidence, "exec_main_status": 1}, launch)
    with pytest.raises(ValueError, match="integer"):
        check._check_exit({**evidence, "main_pid": "0"}, launch)
    with pytest.raises(ValueError, match="integer"):
        check._check_exit({**evidence, "exec_main_status": 0.0}, launch)


def test_checkpoint_arguments_bind_the_entire_parser_namespace() -> None:
    parsed = dict(check.CRITICAL_ARGS)
    parsed["an_unlisted_parser_default"] = "pinned"
    check._check_args(parsed, parsed)
    changed = dict(parsed)
    changed["an_unlisted_parser_default"] = "drifted"
    with pytest.raises(ValueError, match="checkpoint arguments vs launched CLI"):
        check._check_args(parsed, changed)


def test_campaign_split_runtime_defaults_are_applied_only_from_pinned_split() -> None:
    parsed = {"split_manifest": "/bound/splits/fold0.json", "eval_source": "dev"}
    runtime = check._effective_runtime_args(parsed, {"fold": 0, "n_folds": 4}, check.SPLIT_SHA)
    assert parsed["eval_source"] == "dev"
    assert runtime["eval_source"] == "vfold"
    assert runtime["valid_fold"] == 0
    assert runtime["n_valid_folds"] == 4
    assert runtime["split_manifest_sha256"] == check.SPLIT_SHA
    with pytest.raises(ValueError, match="campaign split fold"):
        check._effective_runtime_args(parsed, {"fold": 1, "n_folds": 4}, check.SPLIT_SHA)


def test_checkpoint_digest_map_is_complete_and_matches_bounded_prehashes() -> None:
    files = {
        name: (f"sha-{name}", {"bytes": 1})
        for name in ("model.pt", "model_ep0.pt", "model_ep1.pt", "model_ep2.pt")
    }
    provenance = {"checkpoint_sha256": {name: item[0] for name, item in files.items()}}
    check._check_checkpoint_hashes(provenance, files)
    provenance["checkpoint_sha256"]["model_ep1.pt"] = "different"
    with pytest.raises(ValueError, match="checkpoint hash model_ep1.pt"):
        check._check_checkpoint_hashes(provenance, files)
    with pytest.raises(ValueError, match="checkpoint digest map"):
        check._check_checkpoint_hashes({}, files)


def test_fit_evaluation_components_use_global_ids_across_sources() -> None:
    components = {"train": {1: 8, 2: 9}, "valid": {3: 17, 4: 18}}
    check._check_components(components, [1, 2], [3], [4])
    leaking = {"train": {1: 7}, "valid": {2: 7, 3: 8}}
    with pytest.raises(ValueError, match="components overlap"):
        check._check_components(leaking, [1], [], [2])


def test_epoch_counters_reconcile_three_fit_only_epochs() -> None:
    history, progress, provenance = _counters()
    check._check_counters(history, progress, provenance)
    history[1]["update_diagnostics"]["epoch"]["amp_skipped_updates"] = 1
    with pytest.raises(ValueError, match="epoch counters"):
        check._check_counters(history, progress, provenance)


def test_exact_model_layout_and_fingerprint_are_strict() -> None:
    model = _model_fixture()
    assert check._check_model_layout(model) == (16, 2_159_639)
    fingerprint = check._model_fingerprint(model)
    assert fingerprint == check._model_fingerprint(model)
    del model["gru.bias_ih_l0"]
    with pytest.raises(ValueError, match="inventory"):
        check._check_model_layout(model)
    with pytest.raises(ValueError, match="dtype"):
        check._model_fingerprint({"weight": torch.ones(2, dtype=torch.float64)})
    with pytest.raises(ValueError, match="non-finite"):
        check._model_fingerprint({"weight": torch.tensor([float("nan")], dtype=torch.float32)})


def test_optimizer_requires_exact_group_shapes_and_finite_fp32_state() -> None:
    optimizer = _optimizer_fixture()
    checked = check._check_optimizer(optimizer)
    assert checked == {
        "groups": 3,
        "adamw_parameters": 9,
        "recurrent_muon_parameters": 4,
        "interior_muon_parameters": 1,
        "floating_tensors": 23,
    }
    optimizer["param_groups"][1]["muon_fp32_ortho"] = False
    with pytest.raises(ValueError, match="muon_fp32_ortho"):
        check._check_optimizer(optimizer)
    optimizer = _optimizer_fixture()
    optimizer["state"][0]["exp_avg"] = torch.zeros((1,), dtype=torch.float32)
    with pytest.raises(ValueError, match="expected shape"):
        check._check_optimizer(optimizer)


def test_runtime_must_be_single_thread_strict_fp32() -> None:
    check._check_runtime({"threads": 1, "matmul_tf32": False, "cudnn_tf32": False})
    with pytest.raises(ValueError, match="matmul_tf32"):
        check._check_runtime({"threads": 1, "matmul_tf32": True, "cudnn_tf32": False})
    with pytest.raises(ValueError, match="thread count"):
        check._check_runtime({"threads": 2, "matmul_tf32": False, "cudnn_tf32": False})


def test_restricted_training_state_loader_accepts_numpy_rng_and_blocks_unknown_global() -> None:
    allowed = run_loader_self_test(torch, np)
    assert any("reconstruct" in name for name in allowed)
    assert any("dtype" in name.lower() for name in allowed)


def test_synthetic_producer_state_roundtrips_through_bounded_restricted_consumer(tmp_path) -> None:
    state = {
        "format": "wnn.epoch-training-state",
        "version": 1,
        "boundary": "epoch",
        "next_epoch": 3,
        "model": _model_fixture(),
        "optimizer": _optimizer_fixture(),
        "ema": None,
        "swa": None,
        "rng": {"numpy": np.random.RandomState(23).get_state()},
    }
    path = tmp_path / "synthetic-training-state.pt"
    torch.save(state, path)

    digest, before = check._hash_file(path, 64_000_000)
    restored, loaded_digest, after = check._load_torch_file(
        path, torch, allowlist=check._numpy_safe_globals(np)
    )
    assert (loaded_digest, after) == (digest, before)
    assert check._check_model_layout(restored["model"]) == (16, check.PARAMETER_COUNT)
    assert check._check_optimizer(restored["optimizer"])["floating_tensors"] == 23
