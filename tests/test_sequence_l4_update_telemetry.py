from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from wnn.training.sequence_config import load_config
from wnn.training.sequence_distributed import DistributedContext
from wnn.training.sequence_fit_utils import make_optimizer
from wnn.training.sequence_models import build_sequence_model
from wnn.training.sequence_monitor_diagnostics_events import record_diagnostics_event
from wnn.training.sequence_monitor_events import MonitorEventWriter
from wnn.training.sequence_optimizer_telemetry import optimizer_learning_rates
from wnn.training.sequence_update import finish_update

L4_PRESET = Path(
    "research/experiment_drafts/training-repair-20261108/l4-standalone-labels-wmse-v2.json"
)


class TinyHeadBody(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Linear(3, 4)
        self.head = nn.Linear(4, 2)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.head(torch.tanh(self.body(values)))


def _config(**overrides: object) -> dict[str, object]:
    return {
        "optimizer": "adamw",
        "lr": 0.01,
        "weight_decay": 0.0,
        "adamw_beta1": 0.9,
        "adamw_beta2": 0.999,
        "adamw_eps": 1e-8,
        "schedule": "legacy_warmup_cosine",
        "warmup_fraction": 0.25,
        "min_lr_ratio": 0.0,
        "lr_role_factors": {"adamw": 1.0},
        "lr_head_factor": 1.0,
        "lr_body_factor": 1.0,
        **overrides,
    }


def _copy_nested(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _copy_nested(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_nested(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_nested(item) for item in value)
    return copy.deepcopy(value)


def _assert_nested_equal(left: object, right: object) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert isinstance(right, dict) and left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert isinstance(right, type(left)) and len(left) == len(right)
        for item_left, item_right in zip(left, right, strict=True):
            _assert_nested_equal(item_left, item_right)
    else:
        assert left == right


def _cpu_context() -> DistributedContext:
    return DistributedContext(0, 0, 1, torch.device("cpu"), False)


def _loss(model: TinyHeadBody, values: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (model(values) - target).square().mean()


def _assert_l4_update_payload(
    model: TinyHeadBody,
    optimizer: torch.optim.Optimizer,
    before: dict[str, torch.Tensor],
    gradients: dict[str, torch.Tensor],
    payload: dict[str, object],
    clip: float,
) -> None:
    params = dict(model.named_parameters())
    groups = payload["groups"]
    assert isinstance(groups, dict)
    aggregate_pre_squared = 0.0
    aggregate_post_squared = 0.0
    for optimizer_group in optimizer.param_groups:
        name = optimizer_group["group_name"]
        param_names = optimizer_group["param_names"]
        metric = groups[name]
        before_values = torch.cat([before[item].double().flatten() for item in param_names])
        after_values = torch.cat([params[item].detach().double().flatten() for item in param_names])
        gradient_values = torch.cat([gradients[item].double().flatten() for item in param_names])
        before_norm = float(torch.linalg.vector_norm(before_values))
        delta_norm = float(torch.linalg.vector_norm(after_values - before_values))
        preclip_norm = float(torch.linalg.vector_norm(gradient_values))
        state_values = torch.cat(
            [optimizer.state[params[item]]["exp_avg"].double().flatten() for item in param_names]
        )
        expected_exp_avg_norm = float(torch.linalg.vector_norm(state_values))
        aggregate_pre_squared += preclip_norm**2
        assert metric["pre_weight_norm"] == pytest.approx(before_norm)
        assert metric["pre_weight_rms"] == pytest.approx(
            before_norm / (sum(params[item].numel() for item in param_names) ** 0.5)
        )
        assert metric["delta_weight_norm"] == pytest.approx(delta_norm)
        assert metric["delta_weight_rms"] == pytest.approx(
            delta_norm / (sum(params[item].numel() for item in param_names) ** 0.5)
        )
        assert metric["delta_weight_over_pre_weight_norm"] == pytest.approx(
            delta_norm / max(before_norm, 1e-12)
        )
        assert metric["preclip_gradient_norm"] == pytest.approx(preclip_norm)
        assert metric["postclip_gradient_norm"] <= preclip_norm
        assert metric["parameter_tensor_count"] == len(param_names)
        assert metric["exp_avg_parameter_tensor_count"] == len(param_names)
        assert metric["exp_avg_scalar_count"] == sum(params[item].numel() for item in param_names)
        assert metric["exp_avg_norm"] == pytest.approx(expected_exp_avg_norm)
        assert metric["momentum_parameter_tensor_count"] == 0
        assert metric["momentum_norm"] == 0
        assert metric["weight_norm_denominator_floor"] == 1e-12
        assert metric["zero_weight_denominator_count"] == 0
        aggregate_post_squared += float(metric["postclip_gradient_norm"]) ** 2
    assert payload["preclip_gradient_norm"] == pytest.approx(aggregate_pre_squared**0.5)
    assert payload["preclip_gradient_norm"] > clip
    assert aggregate_post_squared**0.5 <= clip + 1e-6


def test_optimizer_probe_preserves_rng_gradients_parameters_and_resume_state() -> None:
    torch.manual_seed(1701)
    original = TinyHeadBody()
    measured = copy.deepcopy(original)
    control = copy.deepcopy(original)
    config = _config()
    measured_optimizer, measured_scheduler = make_optimizer(measured, config, 4)
    control_optimizer, control_scheduler = make_optimizer(control, config, 4)
    values = torch.tensor([[0.2, -0.1, 0.5], [-0.3, 0.4, 0.7]], dtype=torch.float32)
    target = torch.tensor([[0.6, -0.2], [0.1, 0.9]], dtype=torch.float32)

    diagnostics: dict[str, object] | None = None
    for step, interval in ((1, 50), (2, 2)):
        _loss(measured, values, target).backward()
        _loss(control, values, target).backward()
        measured_gradients = {
            name: parameter.grad.detach().clone()
            for name, parameter in measured.named_parameters()
            if parameter.grad is not None
        }
        control_gradients = {
            name: parameter.grad.detach().clone()
            for name, parameter in control.named_parameters()
            if parameter.grad is not None
        }
        _assert_nested_equal(measured_gradients, control_gradients)
        measured_before = {
            name: parameter.detach().clone() for name, parameter in measured.named_parameters()
        }
        applied_rates = optimizer_learning_rates(measured_optimizer)
        rng_before = torch.get_rng_state().clone()
        _, _, sampled = finish_update(
            measured,
            measured_optimizer,
            measured_scheduler,
            _cpu_context(),
            0.001,
            update_step=step,
            diagnostics_interval=interval,
        )
        rng_after = torch.get_rng_state().clone()
        _, _, absent = finish_update(
            control,
            control_optimizer,
            control_scheduler,
            _cpu_context(),
            0.001,
            update_step=step,
            diagnostics_interval=0,
        )
        assert torch.equal(rng_before, rng_after)
        assert absent is None and sampled is not None
        assert sampled["sampled_update_step"] == step
        _assert_l4_update_payload(
            measured,
            measured_optimizer,
            measured_before,
            measured_gradients,
            sampled,
            0.001,
        )
        for name, applied_lr in applied_rates.items():
            assert sampled["applied_learning_rates"][name] == applied_lr
        if step == 1:
            diagnostics = sampled
        else:
            assert applied_rates["adamw_body"] > 0
            assert applied_rates != optimizer_learning_rates(measured_optimizer)
    assert diagnostics is not None and diagnostics["sampled_update_step"] == 1
    _assert_nested_equal(measured.state_dict(), control.state_dict())
    _assert_nested_equal(measured_optimizer.state_dict(), control_optimizer.state_dict())
    _assert_nested_equal(measured_scheduler.state_dict(), control_scheduler.state_dict())
    assert all(parameter.grad is None for parameter in measured.parameters())
    assert all(parameter.grad is None for parameter in control.parameters())

    resumed = copy.deepcopy(measured)
    resumed_optimizer, resumed_scheduler = make_optimizer(resumed, config, 4)
    resumed_optimizer.load_state_dict(_copy_nested(measured_optimizer.state_dict()))
    resumed_scheduler.load_state_dict(_copy_nested(measured_scheduler.state_dict()))
    _assert_nested_equal(resumed_optimizer.state_dict(), measured_optimizer.state_dict())
    _assert_nested_equal(resumed_scheduler.state_dict(), measured_scheduler.state_dict())


def test_head_body_lr_factors_are_neutral_by_default_and_scale_actual_lrs() -> None:
    model = TinyHeadBody()
    base_config = _config_without_controls(weight_decay=0.02)
    legacy_optimizer, _ = make_optimizer(model, base_config, 4)
    neutral_optimizer, _ = make_optimizer(model, _config(), 4)
    legacy_lr = legacy_optimizer.param_groups[0]["lr"]
    neutral_rates = {
        group["parameter_partition"]: group["lr"] for group in neutral_optimizer.param_groups
    }

    assert neutral_rates == {"body": legacy_lr, "head": legacy_lr}

    scaled_optimizer, scaled_scheduler = make_optimizer(
        model,
        _config(
            weight_decay=0.02,
            lr_role_factors={"adamw": 2.0},
            lr_head_factor=0.25,
            lr_body_factor=1.5,
        ),
        4,
    )
    scaled_rates = {
        group["parameter_partition"]: group["lr"] for group in scaled_optimizer.param_groups
    }
    assert scaled_rates["head"] == pytest.approx(legacy_lr * 2.0 * 0.25)
    assert scaled_rates["body"] == pytest.approx(legacy_lr * 2.0 * 1.5)
    assert scaled_rates["head"] / scaled_rates["body"] == pytest.approx(1.0 / 6.0)
    assert all(group["weight_decay"] == 0.02 for group in scaled_optimizer.param_groups)
    assert all(group["betas"] == (0.9, 0.999) for group in scaled_optimizer.param_groups)
    owned = [name for group in scaled_optimizer.param_groups for name in group["param_names"]]
    assert sorted(owned) == sorted(name for name, _ in model.named_parameters())

    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    scaled_optimizer.step()
    scaled_scheduler.step()
    continued_rates = {
        group["parameter_partition"]: group["lr"] for group in scaled_optimizer.param_groups
    }
    assert continued_rates["head"] / continued_rates["body"] == pytest.approx(1.0 / 6.0)

    resumed_model = copy.deepcopy(model)
    resumed_optimizer, resumed_scheduler = make_optimizer(
        resumed_model,
        _config(
            weight_decay=0.02,
            lr_role_factors={"adamw": 2.0},
            lr_head_factor=0.25,
            lr_body_factor=1.5,
        ),
        4,
    )
    resumed_optimizer.load_state_dict(_copy_nested(scaled_optimizer.state_dict()))
    resumed_scheduler.load_state_dict(_copy_nested(scaled_scheduler.state_dict()))
    resumed_rates = {
        group["parameter_partition"]: group["lr"] for group in resumed_optimizer.param_groups
    }
    assert resumed_rates == continued_rates


def _config_without_controls(**overrides: object) -> dict[str, object]:
    config = _config(**overrides)
    for key in ("lr_role_factors", "lr_head_factor", "lr_body_factor"):
        config.pop(key)
    return config


def test_lr_controls_reject_unknown_roles() -> None:
    with pytest.raises(ValueError, match="unknown or unused roles"):
        make_optimizer(TinyHeadBody(), _config(lr_role_factors={"made_up_role": 1.0}), 4)


def test_future_l4_preset_builds_neutral_named_gru_groups() -> None:
    config = load_config(L4_PRESET)
    model = build_sequence_model(config["model"])
    optimizer, _ = make_optimizer(model, config, 3)

    groups = {group["parameter_partition"]: group for group in optimizer.param_groups}
    assert set(groups) == {"body", "head"}
    assert all(name.startswith("head.") for name in groups["head"]["param_names"])
    assert all(not name.startswith("head.") for name in groups["body"]["param_names"])
    assert groups["body"]["lr"] == pytest.approx(0.0003)
    assert groups["head"]["lr"] == pytest.approx(0.0003)
    assert all(group["weight_decay"] == 0.01 for group in groups.values())
    assert all(group["betas"] == (0.9, 0.999) for group in groups.values())
    assert config["chunk_rows"] == config["gradient_horizon_rows"] == 1000
    assert config["batch_size"] == 32
    assert config["accumulate_chunks"] == 1
    assert config["epochs"] == 6


def test_optimizer_probe_merges_into_one_diagnostics_event(tmp_path: Path) -> None:
    writer = MonitorEventWriter(
        tmp_path / "monitor_events.jsonl", resume=False, extended_limits=True
    )
    payload = {
        "sampling_interval": 50,
        "sampled_update_step": 1,
        "preclip_gradient_norm": 2.0,
        "postclip_gradient_norm": 1.0,
        "applied_learning_rates": {"adamw_head": 0.01},
        "groups": {
            "adamw_head": {
                "role": "adamw",
                "parameter_partition": "head",
                "pre_weight_norm": 4.0,
                "postclip_gradient_norm": 0.7,
            }
        },
    }

    record_diagnostics_event(
        writer,
        1,
        0,
        0.25,
        20,
        {"activation/output_rms": 0.5},
        payload,
    )
    event = json.loads((tmp_path / "monitor_events.jsonl").read_text(encoding="utf-8"))

    assert event["kind"] == "diagnostics"
    assert event["metrics"]["activation/output_rms"] == 0.5
    assert event["metrics"]["optimizer_update/group/adamw_head/pre_weight_norm"] == 4.0
    assert event["context"]["optimizer_update_groups"]["adamw_head"] == {
        "role": "adamw",
        "parameter_partition": "head",
    }
