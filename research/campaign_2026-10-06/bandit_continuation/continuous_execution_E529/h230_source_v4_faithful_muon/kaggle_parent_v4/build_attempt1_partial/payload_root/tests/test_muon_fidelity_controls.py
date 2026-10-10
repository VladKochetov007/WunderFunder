from __future__ import annotations

import argparse

import pytest
import torch
from torch import nn

from wnn.nn import RNNModel
from wnn.optim import (
    MuonAdamW,
    muon_param_groups,
    orthogonalize,
    resolve_orthogonalization_dtype,
)
from wnn.training_dynamics import (
    DEFAULT_MUON_ORTHOGONALIZATION_POLICY,
    add_arguments,
    build_optimizer,
    build_scheduler,
    validate_muon_arguments,
)


class TinyRecurrentModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inp = nn.Linear(3, 4)
        self.gru = nn.GRU(4, 4, batch_first=True)
        self.head = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))


def _args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "optim": "muon",
        "optim_switch": "",
        "head_lr_mult": 1.0,
        "muon_lr_mult": 3.0,
        "muon_momentum": 0.95,
        "muon_gatewise": "none",
        "muon_ortho_dtype_policy": DEFAULT_MUON_ORTHOGONALIZATION_POLICY,
        "muon_scope": "hidden",
        "zoneout": 0.0,
        "momentum": 0.0,
        "fused_adam": False,
        "lr": 1e-3,
        "beta2": 0.999,
        "wd": 0.01,
        "sched": "onecycle",
        "warmup_frac": 0.05,
        "wsd_decay": 0.2,
        "wsd_final": 0.01,
        "restarts": 1,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _group_for_parameter(
    optimizer: torch.optim.Optimizer, parameter: nn.Parameter
) -> dict[str, object]:
    return next(
        group
        for group in optimizer.param_groups
        if any(candidate is parameter for candidate in group["params"])
    )


def test_muon_gatewise_geometry_selects_whole_or_gate_blocks() -> None:
    whole = muon_param_groups(TinyRecurrentModel(), scope="hidden", gatewise="none")
    all_gates = muon_param_groups(TinyRecurrentModel(), scope="hidden", gatewise="all")
    hidden_only = muon_param_groups(TinyRecurrentModel(), scope="hidden", gatewise="hh")

    assert [group["blocks"] for group in whole if group.get("muon")] == [1, 1]
    assert [group["blocks"] for group in all_gates if group.get("muon")] == [3, 1]
    recurrent = [
        group for group in hidden_only if group.get("muon") and group["params"][0].shape[0] == 12
    ]
    assert [group["blocks"] for group in recurrent] == [1, 3]


def test_muon_lr_multiplier_and_onecycle_scale_every_muon_group_only() -> None:
    model = TinyRecurrentModel()
    args = _args()
    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer, args, total=100)

    recurrent = _group_for_parameter(optimizer, model.gru.weight_hh_l0)
    hidden = _group_for_parameter(optimizer, model.head[0].weight)
    input_group = _group_for_parameter(optimizer, model.inp.weight)
    assert recurrent["muon"] is True
    assert hidden["muon"] is True
    assert recurrent["lr"] / input_group["lr"] == pytest.approx(3.0)
    assert hidden["lr"] / input_group["lr"] == pytest.approx(3.0)
    assert recurrent["betas"][0] == pytest.approx(0.95)
    # OneCycleLR intentionally cycles beta1/momentum on every optimizer group,
    # matching the reference schedule's coupled momentum phase.
    assert input_group["betas"][0] == pytest.approx(recurrent["betas"][0])
    assert {group["weight_decay"] for group in optimizer.param_groups} == {args.wd}
    assert [group["max_lr"] for group in optimizer.param_groups] == pytest.approx(
        [args.lr * group["lr_scale"] for group in optimizer.param_groups]
    )

    for _ in range(8):
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        recurrent_lr = _group_for_parameter(optimizer, model.gru.weight_hh_l0)["lr"]
        input_lr = _group_for_parameter(optimizer, model.inp.weight)["lr"]
        assert recurrent_lr / input_lr == pytest.approx(3.0)


def test_muon_multiplier_keeps_same_decay_coefficient_and_scales_decay_step() -> None:
    model = TinyRecurrentModel()
    args = _args(wd=0.1)
    optimizer = build_optimizer(model, args)
    muon_parameter = model.gru.weight_hh_l0
    adam_parameter = model.inp.weight
    muon_parameter.data.fill_(1.0)
    adam_parameter.data.fill_(1.0)
    for parameter in model.parameters():
        parameter.grad = torch.zeros_like(parameter)

    optimizer.step()

    muon_shrink = 1.0 - muon_parameter[0, 0].item()
    adam_shrink = 1.0 - adam_parameter[0, 0].item()
    assert _group_for_parameter(optimizer, muon_parameter)["weight_decay"] == pytest.approx(0.1)
    assert _group_for_parameter(optimizer, adam_parameter)["weight_decay"] == pytest.approx(0.1)
    assert muon_shrink / adam_shrink == pytest.approx(3.0, rel=1e-3)


def test_native_orthogonalization_policy_falls_back_to_fp32_and_fails_closed_when_forced() -> None:
    policy = DEFAULT_MUON_ORTHOGONALIZATION_POLICY
    assert (
        resolve_orthogonalization_dtype(
            policy,
            is_cuda=False,
            cuda_bf16_supported=False,
            compute_capability=None,
        )
        is torch.float32
    )
    assert (
        resolve_orthogonalization_dtype(
            policy,
            is_cuda=True,
            cuda_bf16_supported=False,
            compute_capability=(8, 0),
        )
        is torch.float32
    )
    assert (
        resolve_orthogonalization_dtype(
            policy,
            is_cuda=True,
            cuda_bf16_supported=True,
            compute_capability=(8, 0),
        )
        is torch.bfloat16
    )
    with pytest.raises(RuntimeError, match="requires supported CUDA"):
        resolve_orthogonalization_dtype(
            "bf16",
            is_cuda=True,
            cuda_bf16_supported=True,
            compute_capability=(7, 5),
        )


def test_native_bf16_policy_requires_supported_cuda_and_capability_eight() -> None:
    policy = DEFAULT_MUON_ORTHOGONALIZATION_POLICY
    assert (
        resolve_orthogonalization_dtype(
            policy,
            is_cuda=True,
            cuda_bf16_supported=True,
            compute_capability=(7, 5),
        )
        is torch.float32
    )
    assert (
        resolve_orthogonalization_dtype(
            policy,
            is_cuda=True,
            cuda_bf16_supported=False,
            compute_capability=(8, 0),
        )
        is torch.float32
    )


def test_orthogonalize_resolves_default_policy_without_preselected_dtype() -> None:
    matrix = torch.arange(48, dtype=torch.float32).reshape(3, 4, 4) / 48

    result = orthogonalize(matrix)

    assert result.shape == matrix.shape
    assert result.dtype == matrix.dtype
    assert torch.isfinite(result).all()


def test_ortho_policy_uses_the_explicit_runtime_cli_name() -> None:
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args(["--muon-ortho-dtype-policy", "native_bf16_if_supported_else_fp32_v1"])
    assert args.muon_ortho_dtype_policy == DEFAULT_MUON_ORTHOGONALIZATION_POLICY


def test_muon_specific_flags_validate_before_data_loading() -> None:
    with pytest.raises(ValueError, match="positive and finite"):
        validate_muon_arguments(_args(muon_lr_mult=float("inf")))
    with pytest.raises(ValueError, match="require --optim muon"):
        validate_muon_arguments(_args(optim="adamw", muon_lr_mult=3.0))
    with pytest.raises(ValueError, match="cannot be combined with --optim-switch"):
        validate_muon_arguments(_args(optim_switch="0.5:adamw"))


def test_muon_full_matrix_step_runs_in_fp32_cpu_policy() -> None:
    model = TinyRecurrentModel()
    args = _args(muon_ortho_dtype_policy="fp32")
    optimizer = build_optimizer(model, args)
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 0.01)

    optimizer.step()

    assert isinstance(optimizer, MuonAdamW)
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())


def test_strong_parent_profile_model_and_whole_matrix_groups() -> None:
    model = RNNModel(n_in=211, hidden=256, layers=2, dropout=0.0)
    args = _args()
    optimizer = build_optimizer(model, args)
    parameters = dict(model.named_parameters())

    assert parameters["gru.weight_ih_l0"].shape == (768, 256)
    assert parameters["gru.weight_hh_l0"].shape == (768, 256)
    assert parameters["gru.weight_ih_l1"].shape == (768, 256)
    assert parameters["gru.weight_hh_l1"].shape == (768, 256)
    assert parameters["head.0.weight"].shape == (256, 512)
    muon_groups = [group for group in optimizer.param_groups if group.get("muon")]
    assert len(muon_groups) == 2
    assert all(group["blocks"] == 1 for group in muon_groups)
    assert all(group["lr_scale"] == pytest.approx(3.0) for group in muon_groups)
    for name in (
        "gru.weight_ih_l0",
        "gru.weight_hh_l0",
        "gru.weight_ih_l1",
        "gru.weight_hh_l1",
        "head.0.weight",
    ):
        group = _group_for_parameter(optimizer, parameters[name])
        assert group["muon"] is True
        assert group["blocks"] == 1
