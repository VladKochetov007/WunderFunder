"""Small synthetic invariants for the bounded parent terminal checker."""

from __future__ import annotations

import parent_fit_terminal_check_v1 as check
import pytest
import torch


def _adamw_state() -> dict[str, object]:
    ids = list(range(14))
    states = {
        identifier: {
            "step": torch.tensor(float(check.UPDATES)),
            "exp_avg": torch.zeros(shape, dtype=torch.float32),
            "exp_avg_sq": torch.ones(shape, dtype=torch.float32),
        }
        for identifier, shape in zip(ids, check.PARAMETER_SHAPES, strict=True)
    }
    return {
        "state": states,
        "param_groups": [
            {
                "params": ids,
                "betas": (0.9, 0.999),
                "weight_decay": 0.01,
                "eps": 1e-8,
                "lr": 0.0,
                "amsgrad": False,
                "maximize": False,
                "capturable": False,
                "differentiable": False,
                "fused": False,
            }
        ],
    }


def test_adamw_terminal_state_checks_all_moments_and_steps() -> None:
    optimizer = _adamw_state()
    assert check._check_optimizer(optimizer, 0.0)["step"] == check.UPDATES
    optimizer["state"][0]["exp_avg_sq"][0, 0] = -1
    with pytest.raises(ValueError, match="second moment"):
        check._check_optimizer(optimizer, 0.0)


def test_adamw_rate_must_match_saved_onecycle_rate() -> None:
    optimizer = _adamw_state()
    optimizer["param_groups"][0]["lr"] = 1e-4
    with pytest.raises(ValueError, match="terminal learning rate"):
        check._check_optimizer(optimizer, 4e-8)


def test_auxiliary_state_is_exact_source_derived_no_ema_shape() -> None:
    dynamics = {
        "switch": None,
        "caps": {},
        "eta": 0.0,
        "clip": 1.0,
        "no_clip": False,
        "seed": 1001 + 104729,
        "is_switched": False,
        "events": [],
        "generators": {},
    }
    state = {
        "norm_ema": None,
        "dynamics": dynamics,
        "pool_memory": None,
        "readout_moments": None,
        "input_representation": None,
    }
    check._check_auxiliary(state)
    state["dynamics"] = {**dynamics, "events": [{"switch": "muon"}]}
    with pytest.raises(ValueError, match="dynamics"):
        check._check_auxiliary(state)


def test_model_fingerprint_requires_full_finite_fp32_layout() -> None:
    model = {
        name: torch.zeros(shape, dtype=torch.float32) for name, shape in check.MODEL_SHAPES.items()
    }
    model["std"].fill_(0.001)
    fingerprint, count = check._model_fingerprint(model)
    assert len(fingerprint) == 64
    assert count == 16
    model["head.2.bias"][0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        check._model_fingerprint(model)


def test_epoch_counters_reconcile_cumulative_updates_and_no_ema() -> None:
    diagnostics = []
    history = []
    for epoch in range(4):
        item = {
            "epoch_index": epoch,
            "epoch": {
                "optimizer_attempts": check.EPOCH_UPDATES,
                "optimizer_completed_updates": check.EPOCH_UPDATES,
                "completed_batches": check.BATCHES,
                "amp_skipped_updates": 0,
                "ema_updates": 0,
            },
            "cumulative": {
                "optimizer_attempts": (epoch + 1) * check.EPOCH_UPDATES,
                "optimizer_completed_updates": (epoch + 1) * check.EPOCH_UPDATES,
                "completed_batches": (epoch + 1) * check.BATCHES,
                "amp_skipped_updates": 0,
                "ema_updates": 0,
            },
            "grad_stats": {"steps": check.EPOCH_UPDATES, "nonfinite_steps": 0},
        }
        diagnostics.append(item)
        history.append(
            {
                "epoch": epoch,
                "evaluation": "withheld",
                "update_diagnostics": {
                    key: value for key, value in item.items() if key != "epoch_index"
                },
            }
        )
    totals = {
        "optimizer_attempts": check.UPDATES,
        "optimizer_completed_updates": check.UPDATES,
        "completed_batches": check.BATCHES * 4,
        "amp_skipped_updates": 0,
        "ema_updates": 0,
    }
    state = {
        "progress": {
            "step_count": check.UPDATES,
            "best": -1.0,
            "history": history,
            "cumulative_updates": totals,
            "update_diagnostics": diagnostics,
            "fitting_seconds": 10.0,
        }
    }
    provenance = {"update_diagnostics": diagnostics, "cumulative_updates": totals}
    check._check_counters(history, state, provenance)
    diagnostics[2]["epoch"]["ema_updates"] = 1
    history[2]["update_diagnostics"]["epoch"]["ema_updates"] = 1
    with pytest.raises(ValueError, match="EMA"):
        check._check_counters(history, state, provenance)


def test_onecycle_state_is_bound_to_parent_update_horizon() -> None:
    schedule = {
        "total_steps": check.UPDATES,
        "_schedule_phases": [
            {
                "end_step": 1491.0,
                "start_lr": "initial_lr",
                "end_lr": "max_lr",
                "start_momentum": "max_momentum",
                "end_momentum": "base_momentum",
            },
            {
                "end_step": 29839,
                "start_lr": "max_lr",
                "end_lr": "min_lr",
                "start_momentum": "base_momentum",
                "end_momentum": "max_momentum",
            },
        ],
        "_anneal_func_type": "cos",
        "cycle_momentum": True,
        "use_beta1": True,
        "base_lrs": [4e-06],
        "last_epoch": check.UPDATES,
        "_step_count": check.UPDATES + 1,
        "_is_initial": False,
        "_get_lr_called_within_step": False,
        "_last_lr": [4e-08],
    }
    assert check._check_scheduler(schedule) == 4e-08
    schedule["total_steps"] = check.UPDATES - 1
    with pytest.raises(ValueError, match="total steps"):
        check._check_scheduler(schedule)


def test_native_terminal_types_reject_bool_substitution() -> None:
    evidence = {
        "schema": "fg19.parent-fit-r2-native-terminal-evidence.v1",
        "unit": check.UNIT,
        "invocation_id": check.INVOCATION,
        "run_name": check.RUN,
        "active_state": "active",
        "sub_state": "exited",
        "main_pid": 0,
        "exec_main_code": 1,
        "exec_main_status": 0,
        "raw_bytes": 32,
        "raw_sha256": "a" * 64,
    }
    check._check_terminal(evidence)
    with pytest.raises(ValueError, match="main_pid"):
        check._check_terminal({**evidence, "main_pid": False})


def test_cache_provenance_rejects_second_array_hash_or_scale_drift(tmp_path) -> None:
    provenance = {"distillation_cache": {**check.CACHE_PROVENANCE, "root": str(tmp_path)}}
    check._check_cache_provenance(provenance, tmp_path)
    changed = {
        **check.CACHE_PROVENANCE,
        "root": str(tmp_path),
        "verified_files": {
            "main_homogeneous_train": check.TRAIN_CACHE_SHA,
            "main_homogeneous_valid": "0" * 64,
        },
    }
    with pytest.raises(ValueError, match="cache source"):
        check._check_cache_provenance({"distillation_cache": changed}, tmp_path)


def test_restricted_torch_loader_honors_parent_payload_cap(tmp_path) -> None:
    path = tmp_path / "synthetic.pt"
    torch.save({"x": torch.tensor([1.0])}, path)
    with pytest.raises(ValueError, match="bounded parent checkpoint"):
        check._load_checkpoint(path, max_bytes=1)
