"""Prepared CPU-level coverage for sequence Muon groups and schedule semantics."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from torch import nn

from wnn.optim import MuonAdamW, orthogonalize
from wnn.training.sequence_config import load_config
from wnn.training.sequence_fit_utils import (
    make_optimizer,
    optimizer_states_are_finite_fp32,
    sequence_lr_multiplier,
)
from wnn.training.sequence_models import build_sequence_model
from wnn.training.sequence_optimizer import build_sequence_muon_optimizer


def _transformer_config() -> dict[str, object]:
    return {
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
    }


def _muon_config() -> dict[str, object]:
    return {
        "lr": 3e-4,
        "weight_decay": 0.01,
        "muon_momentum": 0.95,
        "muon_ns_steps": 5,
        "muon_ns_eps": 1e-7,
        "adamw_beta1": 0.9,
        "adamw_beta2": 0.95,
        "adamw_eps": 1e-8,
        "model": _transformer_config(),
    }


def _strict_optimizer(matrix: nn.Parameter, vector: nn.Parameter) -> MuonAdamW:
    groups: list[dict[str, object]] = [
        {
            "params": [matrix],
            "param_names": ["blocks.0.qkv.weight"],
            "role": "muon_qkv",
            "group_name": "muon_qkv",
            "muon": True,
            "blocks": 1,
            "betas": (0.95, 0.95),
            "weight_decay": 0.01,
            "strict_fp32": True,
        },
        {
            "params": [vector],
            "param_names": ["output_head.bias"],
            "role": "adamw_no_decay",
            "group_name": "adamw_no_decay",
            "muon": False,
            "weight_decay": 0.01,
            "betas": (0.9, 0.95),
            "strict_fp32": True,
        },
    ]
    return MuonAdamW(
        groups,
        lr=0.002,
        betas=(0.9, 0.95),
        eps=1e-8,
        ns_steps=2,
        ns_eps=1e-7,
        strict_fp32=True,
    )


def _reference_orthogonalize(gradient: torch.Tensor, steps: int) -> torch.Tensor:
    """Compute the declared Newton-Schulz polynomial independently for update checks."""
    a, b, c = 3.4445, -4.7750, 2.0315
    value = gradient / (gradient.norm() + 1e-7)
    for _ in range(steps):
        gram = value @ value.mT
        value = a * value + (b * gram + c * gram @ gram) @ value
    return value


def test_sequence_muon_groups_split_qkv_and_own_every_trainable_parameter() -> None:
    model = build_sequence_model(_transformer_config())
    optimizer = build_sequence_muon_optimizer(model, _muon_config())
    groups = optimizer.param_groups
    by_role = {str(group["role"]): group for group in groups}

    assert {"muon_qkv", "muon_hidden"}.issubset(by_role)
    qkv = by_role["muon_qkv"]
    assert qkv["blocks"] == 3
    assert qkv["param_names"] == ["blocks.0.qkv.weight", "blocks.1.qkv.weight"]
    assert all(parameter.shape[0] == 3 * parameter.shape[1] for parameter in qkv["params"])
    assert by_role["muon_hidden"]["param_names"] == [
        "blocks.0.attention_output.weight",
        "blocks.0.feedforward_in.weight",
        "blocks.0.feedforward_out.weight",
        "blocks.1.attention_output.weight",
        "blocks.1.feedforward_in.weight",
        "blocks.1.feedforward_out.weight",
    ]

    owners = [parameter for group in groups for parameter in group["params"]]
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert len(owners) == len(trainable)
    assert {id(parameter) for parameter in owners} == {id(parameter) for parameter in trainable}
    assert all(isinstance(group["param_names"], list) for group in groups)
    assert all(group["optimizer_group_schema"] == "wnn.sequence-muon-group.v1" for group in groups)
    assert all(group["group_name"] == group["role"] for group in groups)
    assert by_role["adamw_matrix"]["param_names"] == [
        "input_projection.weight",
        "output_head.weight",
    ]
    assert by_role["adamw_matrix"]["weight_decay"] == pytest.approx(0.01)
    assert by_role["adamw_no_decay"]["weight_decay"] == 0.0
    assert by_role["adamw_matrix"]["betas"] == (0.9, 0.95)
    assert qkv["betas"][0] == pytest.approx(0.95)


def test_strict_newton_schulz_keeps_every_matmul_fp32(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: list[tuple[torch.dtype, torch.dtype]] = []
    original_matmul = torch.Tensor.__matmul__

    def record_matmul(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        observed.append((left.dtype, right.dtype))
        return original_matmul(left, right)

    monkeypatch.setattr(torch.Tensor, "__matmul__", record_matmul)
    output = orthogonalize(torch.eye(4, dtype=torch.float32), steps=3, strict_fp32=True)

    assert output.dtype == torch.float32
    assert len(observed) == 9
    assert all(left == right == torch.float32 for left, right in observed)
    with pytest.raises(TypeError, match="float32 input"):
        orthogonalize(torch.eye(4, dtype=torch.bfloat16), strict_fp32=True)


def test_muon_and_adamw_updates_match_independent_reference() -> None:
    matrix = nn.Parameter(torch.tensor([[0.2, -0.1], [0.3, 0.4]], dtype=torch.float32))
    vector = nn.Parameter(torch.tensor([0.3, -0.2], dtype=torch.float32))
    matrix_gradient = torch.tensor([[0.5, -0.4], [0.1, 0.2]], dtype=torch.float32)
    vector_gradient = torch.tensor([0.6, -0.3], dtype=torch.float32)
    matrix.grad = matrix_gradient.clone()
    vector.grad = vector_gradient.clone()
    optimizer = _strict_optimizer(matrix, vector)
    matrix_start, vector_start = matrix.detach().clone(), vector.detach().clone()

    beta_muon, beta1, beta2 = 0.95, 0.9, 0.95
    momentum_buffer = (1.0 - beta_muon) * matrix_gradient
    nesterov = (1.0 - beta_muon) * matrix_gradient + beta_muon * momentum_buffer
    orthogonal = _reference_orthogonalize(nesterov, steps=2)
    expected_matrix = matrix_start * (1.0 - 0.002 * 0.01)
    expected_matrix -= orthogonal * (0.002 * 0.2 * math.sqrt(2.0))
    first_moment = (1.0 - beta1) * vector_gradient
    second_moment = (1.0 - beta2) * vector_gradient.square()
    denominator = second_moment.sqrt() / math.sqrt(1.0 - beta2) + 1e-8
    expected_vector = vector_start * (1.0 - 0.002 * 0.01)
    expected_vector.addcdiv_(first_moment, denominator, value=-0.002 / (1.0 - beta1))

    optimizer.step()

    torch.testing.assert_close(matrix, expected_matrix, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(vector, expected_vector, rtol=1e-6, atol=1e-7)
    assert optimizer_states_are_finite_fp32(optimizer)


def test_strict_optimizer_rejects_reduced_precision_state_without_gradient() -> None:
    matrix = nn.Parameter(torch.eye(2, dtype=torch.float32))
    vector = nn.Parameter(torch.ones(2, dtype=torch.float32))
    optimizer = _strict_optimizer(matrix, vector)
    optimizer.state[matrix]["momentum_buffer"] = torch.zeros_like(matrix, dtype=torch.bfloat16)

    with pytest.raises(TypeError, match="float32 optimizer state"):
        optimizer.step()


def test_muon_loads_legacy_groups_without_strict_fp32_options() -> None:
    initial = torch.tensor([[0.2, -0.1], [0.3, 0.4]], dtype=torch.float32)
    reference_parameter = nn.Parameter(initial.clone())
    reference = MuonAdamW(
        [
            {
                "params": [reference_parameter],
                "muon": True,
                "blocks": 1,
                "betas": (0.95, 0.95),
            }
        ],
        lr=0.002,
        weight_decay=0.01,
        ns_steps=2,
    )
    reference_parameter.grad = torch.tensor([[0.5, -0.4], [0.1, 0.2]], dtype=torch.float32)
    reference.step()

    # Keep an immutable checkpoint snapshot; optimizer.state_dict() returns
    # references to mutable state tensors that optimizer.step() updates in place.
    saved_state = deepcopy(reference.state_dict())
    for group in saved_state["param_groups"]:
        group.pop("strict_fp32", None)
        group.pop("ns_eps", None)

    resumed_parameter = nn.Parameter(reference_parameter.detach().clone())
    resumed = MuonAdamW(
        [
            {
                "params": [resumed_parameter],
                "muon": True,
                "blocks": 1,
                "betas": (0.95, 0.95),
            }
        ],
        lr=0.002,
        weight_decay=0.01,
        ns_steps=2,
    )
    resumed.load_state_dict(saved_state)

    next_gradient = torch.tensor([[-0.2, 0.3], [0.4, -0.1]], dtype=torch.float32)
    reference_parameter.grad = next_gradient.clone()
    resumed_parameter.grad = next_gradient.clone()
    reference.step()
    resumed.step()

    torch.testing.assert_close(resumed_parameter, reference_parameter)
    torch.testing.assert_close(
        resumed.state[resumed_parameter]["momentum_buffer"],
        reference.state[reference_parameter]["momentum_buffer"],
    )


def test_default_sequence_optimizer_keeps_legacy_adamw_and_schedule() -> None:
    model = nn.Linear(2, 2)
    optimizer, _ = make_optimizer(
        model,
        {"lr": 1e-4, "weight_decay": 0.01, "warmup_fraction": 0.05},
        total_updates=100,
    )

    assert type(optimizer) is torch.optim.AdamW
    assert optimizer.defaults["betas"] == (0.9, 0.999)
    assert optimizer.defaults["eps"] == 1e-8


def test_config_keeps_legacy_beta2_and_uses_muon_beta2_default(tmp_path: Path) -> None:
    recipe = tmp_path / "sequence.json"
    recipe.write_text("{}")
    assert load_config(recipe)["adamw_beta2"] == 0.999

    recipe.write_text(json.dumps({"optimizer": "muon"}))
    assert load_config(recipe)["adamw_beta2"] == 0.95


def test_warmup_cosine_v2_hits_exact_warmup_midpoint_and_floor() -> None:
    multiplier = sequence_lr_multiplier("warmup_cosine_v2", 4, 0.05, 0.1)
    legacy = sequence_lr_multiplier("legacy_warmup_cosine", 4, 0.25, 0.0)

    assert [multiplier(step) for step in range(4)] == pytest.approx([1.0, 1.0, 0.55, 0.1])
    assert [legacy(step) for step in range(4)] == pytest.approx([1.0, 0.75, 0.25, 0.0])
    with pytest.raises(ValueError, match="at least two"):
        sequence_lr_multiplier("warmup_cosine_v2", 1, 0.05, 0.1)
    with pytest.raises(ValueError, match="post-warmup"):
        sequence_lr_multiplier("warmup_cosine_v2", 2, 0.05, 0.1)


def test_muon_optimizer_and_schedule_state_round_trip() -> None:
    matrix = nn.Parameter(torch.eye(2, dtype=torch.float32))
    vector = nn.Parameter(torch.ones(2, dtype=torch.float32))
    first_optimizer = _strict_optimizer(matrix, vector)
    multiplier = sequence_lr_multiplier("warmup_cosine_v2", 4, 0.05, 0.1)
    first_scheduler = torch.optim.lr_scheduler.LambdaLR(first_optimizer, multiplier)
    matrix.grad = torch.ones_like(matrix)
    vector.grad = torch.ones_like(vector)
    first_optimizer.step()
    first_scheduler.step()
    optimizer_state = first_optimizer.state_dict()
    scheduler_state = first_scheduler.state_dict()

    next_matrix = nn.Parameter(matrix.detach().clone())
    next_vector = nn.Parameter(vector.detach().clone())
    next_optimizer = _strict_optimizer(next_matrix, next_vector)
    next_scheduler = torch.optim.lr_scheduler.LambdaLR(next_optimizer, multiplier)
    next_optimizer.load_state_dict(optimizer_state)
    next_scheduler.load_state_dict(scheduler_state)

    assert [group["role"] for group in next_optimizer.param_groups] == [
        "muon_qkv",
        "adamw_no_decay",
    ]
    assert [group["param_names"] for group in next_optimizer.param_groups] == [
        ["blocks.0.qkv.weight"],
        ["output_head.bias"],
    ]
    assert [group["group_name"] for group in next_optimizer.param_groups] == [
        "muon_qkv",
        "adamw_no_decay",
    ]
    assert next_scheduler.state_dict() == scheduler_state
    assert optimizer_states_are_finite_fp32(next_optimizer)


def test_config_rejects_muon_for_gru_and_bad_coefficients(tmp_path: Path) -> None:
    recipe = tmp_path / "sequence.json"
    recipe.write_text(json.dumps({"optimizer": "muon", "model": {"architecture": "gru"}}))
    with pytest.raises(ValueError, match="Transformer models only"):
        load_config(recipe)

    recipe.write_text(json.dumps({"muon_momentum": 1.0}))
    with pytest.raises(ValueError, match="muon_momentum"):
        load_config(recipe)
