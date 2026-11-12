from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from wnn.training.sequence_config import load_config
from wnn.training.sequence_distributed import DistributedContext
from wnn.training.sequence_fit_utils import make_optimizer, total_updates
from wnn.training.sequence_models import build_sequence_model
from wnn.training.sequence_optimizer import build_sequence_muon_optimizer
from wnn.training.sequence_runtime_context import runtime_context_contract
from wnn.training.sequence_update import finish_update
from wnn.training.sequence_update_telemetry import (
    capture_optimizer_update,
    finalize_optimizer_update,
    optimizer_gradient_statistics,
)

_ROOT = Path(__file__).parents[1]
_PRESET = (
    _ROOT
    / "research/experiment_drafts/training-repair-20261108/configs/standalone_labels_wmse_v2.json"
)
_LABELS_V1 = (
    _ROOT / "research/experiment_drafts/ensemble-objective-fix-20261108/configs/labels_s101101.json"
)


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Linear(3, 4)
        self.head = nn.Linear(4, 2)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.head(torch.tanh(self.body(values)))


def _adamw_config(**changes: object) -> dict[str, object]:
    return {
        "optimizer": "adamw",
        "lr": 0.02,
        "weight_decay": 0.01,
        "adamw_beta1": 0.9,
        "adamw_beta2": 0.99,
        "adamw_eps": 1e-8,
        "warmup_fraction": 0.25,
        "min_lr_ratio": 0.1,
        "schedule": "warmup_cosine_v2",
        **changes,
    }


def _assert_nested_tensors_equal(left: object, right: object) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert isinstance(right, dict) and left.keys() == right.keys()
        for key in left:
            _assert_nested_tensors_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert isinstance(right, type(left)) and len(left) == len(right)
        for first, second in zip(left, right, strict=True):
            _assert_nested_tensors_equal(first, second)
    else:
        assert left == right


def test_runtime_context_distinguishes_attention_chunk_and_optimizer_window() -> None:
    common = {"model": {"architecture": "transformer", "context_rows": 1024}}
    for chunk, accumulation in ((256, 4), (512, 2), (1024, 1)):
        contract = runtime_context_contract(
            common
            | {
                "chunk_rows": chunk,
                "accumulate_chunks": accumulation,
                "gradient_horizon_rows": chunk,
            }
        )
        assert contract["sequence_rows"] == 20_000
        assert contract["model_context"]["attention_window_rows_per_layer"] == 1024
        assert contract["model_context"]["max_cached_rows_per_layer"] == 1023
        assert contract["model_context"]["forward_state_kind"] == (
            "per-layer Transformer key/value cache"
        )
        assert contract["tbptt"]["detach_chunk_rows"] == chunk
        assert contract["tbptt"]["gradient_horizon_rows"] == chunk
        assert contract["tbptt"]["nominal_optimizer_window_rows"] == 1024
        assert contract["tbptt"]["optimizer_windows_per_sequence"] == 20
        assert contract["tbptt"]["final_optimizer_window_rows"] == 544
        assert contract["tbptt"]["state_carried_across_optimizer_updates"] is True
        assert "stale approximation" in contract["tbptt"]["carried_state_semantics"]
    gru = runtime_context_contract(
        {
            "model": {"architecture": "gru", "context_rows": 1024},
            "chunk_rows": 256,
            "accumulate_chunks": 4,
        }
    )
    assert gru["model_context"]["forward_state_kind"] == "continuous GRU recurrent hidden state"
    assert gru["model_context"]["attention_window_rows_per_layer"] is None
    assert gru["model_context"]["max_cached_rows_per_layer"] is None
    with pytest.raises(ValueError, match="only Transformer or GRU"):
        runtime_context_contract(
            {
                "model": {"architecture": "other"},
                "chunk_rows": 256,
                "accumulate_chunks": 4,
            }
        )


def test_runtime_horizon_mismatch_and_frozen_v1_controls_reject(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("CUBLAS_EMULATE_SINGLE_PRECISION", "1")
    raw = json.loads(_PRESET.read_text(encoding="utf-8"))
    raw["gradient_horizon_rows"] = 512
    candidate = tmp_path / "mismatched-horizon.json"
    candidate.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="must equal chunk_rows"):
        load_config(candidate)

    legacy = json.loads(_LABELS_V1.read_text(encoding="utf-8"))
    legacy["lr_head_factor"] = 1.2
    candidate.write_text(json.dumps(legacy), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen four-fit"):
        load_config(candidate)


def test_new_mode_accepts_opt_in_controls_and_rejects_invalid_sampling(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("CUBLAS_EMULATE_SINGLE_PRECISION", "1")
    raw = json.loads(_PRESET.read_text(encoding="utf-8"))
    raw.update(
        gradient_horizon_rows=256,
        optimizer_update_diagnostics_every=50,
        lr_role_factors={"muon_qkv": 0.5},
        lr_head_factor=1.5,
        lr_body_factor=0.75,
    )
    candidate = tmp_path / "opted-in-controls.json"
    candidate.write_text(json.dumps(raw), encoding="utf-8")
    config = load_config(candidate)
    assert config["optimizer_update_diagnostics_every"] == 50
    assert config["lr_head_factor"] == 1.5
    raw["optimizer_update_diagnostics_every"] = 0
    candidate.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="diagnostics_every must be a positive integer"):
        load_config(candidate)


def test_profile_variants_preserve_update_count_across_chunk_and_world_choices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUBLAS_EMULATE_SINGLE_PRECISION", "1")
    profile_dir = _PRESET.parent / "profiles"
    variants = [
        load_config(_PRESET),
        *(
            load_config(profile_dir / filename)
            for filename in (
                "profile_only_chunk512_accumulate2.json",
                "profile_only_chunk1024_accumulate1.json",
                "profile_only_world1_batch256.json",
            )
        ),
    ]
    assert all(item["optimizer_update_diagnostics_every"] == 50 for item in variants[1:])
    assert all(item["gradient_horizon_rows"] == item["chunk_rows"] for item in variants[1:])
    assert all(item["lr_head_factor"] == item["lr_body_factor"] == 1.0 for item in variants)
    assert all(set(item["lr_role_factors"].values()) == {1.0} for item in variants)
    counts = [
        total_updates(item, sequence_count=10_325, world_size=world)
        for item, world in zip(variants, (2, 2, 2, 1), strict=True)
    ]
    assert counts == [4 * 820] * 4


def test_muon_lr_factors_split_actual_roles_and_survive_scheduler_resume() -> None:
    model_config = {
        "architecture": "transformer",
        "hidden": 8,
        "layers": 2,
        "heads": 2,
        "ff_multiplier": 2,
        "context_rows": 8,
        "dropout": 0.0,
        "checkpoint_blocks": False,
        "hint_dim": 0,
        "hint_projection": "learned",
        "attention_backend": "math",
        "compile_blocks": False,
        "precision_mode": "fp32",
    }
    config = {
        "optimizer": "muon",
        "lr": 0.002,
        "weight_decay": 0.01,
        "muon_momentum": 0.95,
        "muon_ns_steps": 1,
        "muon_ns_eps": 1e-7,
        "adamw_beta1": 0.9,
        "adamw_beta2": 0.95,
        "adamw_eps": 1e-8,
        "schedule": "warmup_cosine_v2",
        "warmup_fraction": 0.25,
        "min_lr_ratio": 0.1,
        "lr_role_factors": {"muon_qkv": 0.5},
        "lr_head_factor": 2.0,
        "lr_body_factor": 0.5,
        "model": model_config,
    }
    model = build_sequence_model(model_config)
    optimizer = build_sequence_muon_optimizer(model, config)
    by_name = {group["group_name"]: group for group in optimizer.param_groups}
    assert by_name["muon_qkv_body"]["lr"] == pytest.approx(0.0005)
    assert by_name["muon_hidden_body"]["lr"] == pytest.approx(0.001)
    assert by_name["adamw_matrix_head"]["lr"] == pytest.approx(0.004)
    assert by_name["adamw_no_decay_head"]["lr"] == pytest.approx(0.004)
    assert by_name["adamw_matrix_head"]["role"] == "adamw_matrix"

    optimizer, scheduler = make_optimizer(model, config, total_updates=4)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    scheduler.step()
    saved_optimizer = optimizer.state_dict()
    saved_scheduler = scheduler.state_dict()
    resumed_model = build_sequence_model(model_config)
    resumed_optimizer, resumed_scheduler = make_optimizer(resumed_model, config, total_updates=4)
    resumed_optimizer.load_state_dict(saved_optimizer)
    resumed_scheduler.load_state_dict(saved_scheduler)
    assert [group["group_name"] for group in resumed_optimizer.param_groups] == [
        group["group_name"] for group in optimizer.param_groups
    ]
    resumed_by_name = {group["group_name"]: group for group in resumed_optimizer.param_groups}
    current_by_name = {group["group_name"]: group for group in optimizer.param_groups}
    assert current_by_name["muon_qkv_body"]["lr"] / current_by_name["muon_hidden_body"][
        "lr"
    ] == pytest.approx(0.5)
    assert resumed_by_name["muon_qkv_body"]["lr"] / resumed_by_name["muon_hidden_body"][
        "lr"
    ] == pytest.approx(0.5)
    for resumed, original in zip(
        resumed_optimizer.param_groups, optimizer.param_groups, strict=True
    ):
        assert resumed["lr"] == pytest.approx(original["lr"])
    assert optimizer.param_groups[0]["lr"] > 0
    _assert_nested_tensors_equal(optimizer.state_dict(), resumed_optimizer.state_dict())
    assert resumed_scheduler.state_dict() == saved_scheduler

    bad = config | {"lr_role_factors": {"unused_role": 1.5}}
    with pytest.raises(ValueError, match="unknown or unused roles"):
        build_sequence_muon_optimizer(build_sequence_model(model_config), bad)


def test_adamw_head_body_split_and_unknown_role_rejection() -> None:
    model = _TinyModel()
    optimizer, _ = make_optimizer(
        model,
        _adamw_config(lr_head_factor=2.0, lr_body_factor=0.5),
        total_updates=4,
    )
    by_name = {group["group_name"]: group for group in optimizer.param_groups}
    assert set(by_name) == {"adamw_head", "adamw_body"}
    assert by_name["adamw_head"]["lr"] == pytest.approx(0.04)
    assert by_name["adamw_body"]["lr"] == pytest.approx(0.01)
    with pytest.raises(ValueError, match="unknown or unused roles"):
        make_optimizer(_TinyModel(), _adamw_config(lr_role_factors={"adamw_no_decay": 1.2}), 4)


def test_sampled_update_diagnostics_preserve_actual_step_and_scheduler_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Event:
        def __init__(self, **_: object) -> None:
            pass

        def record(self) -> None:
            pass

        def elapsed_time(self, _: object) -> float:
            return 1.0

    monkeypatch.setattr(torch.cuda, "Event", _Event)
    torch.manual_seed(20261108)
    initial = _TinyModel()
    baseline_model = copy.deepcopy(initial)
    sampled_model = copy.deepcopy(initial)
    config = _adamw_config(lr_head_factor=1.5, lr_body_factor=0.5)
    baseline_optimizer, baseline_scheduler = make_optimizer(baseline_model, config, 4)
    sampled_optimizer, sampled_scheduler = make_optimizer(sampled_model, config, 4)
    for optimizer, scheduler in (
        (baseline_optimizer, baseline_scheduler),
        (sampled_optimizer, sampled_scheduler),
    ):
        for _ in range(2):
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    parameter.grad = torch.full_like(parameter, 0.25)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

    input_values = torch.tensor([[5.0, -3.0, 2.0], [-4.0, 6.0, 1.0]])
    baseline_loss = baseline_model(input_values).square().sum()
    sampled_loss = sampled_model(input_values).square().sum()
    baseline_loss.backward()
    sampled_loss.backward()
    saved_gradients = [parameter.grad.clone() for parameter in sampled_model.parameters()]
    baseline_gradients = [parameter.grad.clone() for parameter in baseline_model.parameters()]
    assert all(
        torch.equal(left, right)
        for left, right in zip(baseline_gradients, saved_gradients, strict=True)
    )
    applied_rates = {
        group["group_name"]: float(group["lr"]) for group in sampled_optimizer.param_groups
    }
    before_rng = torch.get_rng_state().clone()
    first = capture_optimizer_update(sampled_optimizer, 1, 50)
    skipped = capture_optimizer_update(sampled_optimizer, 2, 50)
    fiftieth = capture_optimizer_update(sampled_optimizer, 50, 50)
    torch.testing.assert_close(torch.get_rng_state(), before_rng, rtol=0, atol=0)
    assert first is not None and skipped is None and fiftieth is not None
    assert all(
        torch.equal(parameter.grad, saved)
        for parameter, saved in zip(sampled_model.parameters(), saved_gradients, strict=True)
    )

    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)
    baseline_rng = torch.get_rng_state().clone()
    baseline_result = finish_update(
        baseline_model, baseline_optimizer, baseline_scheduler, context, 0.05
    )
    torch.testing.assert_close(torch.get_rng_state(), baseline_rng, rtol=0, atol=0)
    sampled_rng = torch.get_rng_state().clone()
    probe_result = finish_update(
        sampled_model,
        sampled_optimizer,
        sampled_scheduler,
        context,
        0.05,
        telemetry_snapshot=fiftieth,
    )
    torch.testing.assert_close(torch.get_rng_state(), sampled_rng, rtol=0, atol=0)
    baseline_norm, _ = baseline_result
    probe_norm, _, diagnostics = probe_result
    torch.testing.assert_close(baseline_norm, probe_norm, rtol=0, atol=0)
    assert diagnostics["sampled_update_step"] == 50
    assert diagnostics["sampling_interval"] == 50
    assert diagnostics["applied_learning_rates"] == pytest.approx(applied_rates)
    assert applied_rates["adamw_body"] == pytest.approx(0.02 * 0.5 * 0.55)
    assert applied_rates["adamw_head"] == pytest.approx(0.02 * 1.5 * 0.55)
    assert diagnostics["preclip_gradient_norm"] > diagnostics["postclip_gradient_norm"]
    assert diagnostics["preclip_gradient_norm"] == pytest.approx(float(probe_norm))
    assert diagnostics["preclip_gradient_scalar_count"] == sum(
        gradient.numel() for gradient in saved_gradients
    )
    assert set(diagnostics["groups"]) == {"adamw_body", "adamw_head"}
    for name, group in diagnostics["groups"].items():
        assert group["applied_lr"] == pytest.approx(applied_rates[name])
        assert group["parameter_scalar_count"] > 0
        assert group["pre_weight_norm"] > 0
        assert group["delta_weight_norm"] > 0
        assert group["zero_weight_denominator_count"] == 0
        assert group["weight_norm_denominator_floor"] == 1e-12
        assert group["exp_avg_parameter_tensor_count"] > 0
        assert group["preclip_gradient_parameter_tensor_count"] > 0
        assert group["postclip_gradient_parameter_tensor_count"] > 0
    for left, right in zip(baseline_model.parameters(), sampled_model.parameters(), strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    torch.testing.assert_close(
        baseline_model(input_values), sampled_model(input_values), rtol=0, atol=0
    )
    for left, right in zip(baseline_model.parameters(), sampled_model.parameters(), strict=True):
        assert left.grad is None and right.grad is None
    _assert_nested_tensors_equal(baseline_optimizer.state_dict(), sampled_optimizer.state_dict())
    assert baseline_scheduler.state_dict() == sampled_scheduler.state_dict()
    assert all(rate > 0 for rate in applied_rates.values())
    assert any(
        float(group["lr"]) != applied_rates[group["group_name"]]
        for group in sampled_optimizer.param_groups
    )


def test_sampled_gradient_stats_are_read_only() -> None:
    model = _TinyModel()
    optimizer, _ = make_optimizer(
        model,
        _adamw_config(lr_head_factor=1.5, lr_body_factor=0.5),
        total_updates=4,
    )
    model(torch.ones(2, 3)).square().sum().backward()
    grads = [parameter.grad.clone() for parameter in model.parameters()]
    rng = torch.get_rng_state().clone()
    metrics = optimizer_gradient_statistics(optimizer)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    assert all(
        torch.equal(parameter.grad, saved)
        for parameter, saved in zip(model.parameters(), grads, strict=True)
    )
    assert set(metrics) == {"adamw_body", "adamw_head"}


def test_zero_weight_group_reports_applied_denominator_floor() -> None:
    model = _TinyModel()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    optimizer, _ = make_optimizer(
        model,
        _adamw_config(lr_head_factor=1.5, lr_body_factor=0.5),
        total_updates=4,
    )
    for parameter in model.parameters():
        parameter.grad = torch.zeros_like(parameter)
    snapshot = capture_optimizer_update(optimizer, 1, 50)
    assert snapshot is not None
    preclip = optimizer_gradient_statistics(optimizer)
    applied_rates = {group["group_name"]: float(group["lr"]) for group in optimizer.param_groups}
    optimizer.step()
    postclip = optimizer_gradient_statistics(optimizer)
    diagnostics = finalize_optimizer_update(
        snapshot, optimizer, applied_rates, preclip, postclip, preclip_gradient_norm=0.0
    )
    for group in diagnostics["groups"].values():
        assert group["weight_norm_denominator_floor"] == 1e-12
        assert group["zero_weight_denominator_count"] == 1
        assert group["delta_weight_over_pre_weight_norm"] == 0.0
