from __future__ import annotations

import argparse
import math

import pytest
import torch

from wnn.nn import RNNModel
from wnn.optim import MuonAdamW, muon_param_groups, orthogonalize
from wnn.training.cli import build_parser
from wnn.training.muon_precision import muon_precision_error, muon_resume_policy_error
from wnn.training_dynamics import build_optimizer, build_scheduler, switch_optimizer


def _args(*argv: str) -> argparse.Namespace:
    return build_parser().parse_args(["--name", "muon-test", *argv])


def test_cli_defaults_preserve_legacy_optimizer_policy() -> None:
    args = _args()
    assert args.optim == "adamw"
    assert args.muon_scope == "rnn"
    assert args.muon_ortho_dtype == "auto"
    assert not args.strict_muon_fp32
    assert args.precision == "fp16"


def test_strict_muon_recipe_requires_explicit_fp32_and_active_muon() -> None:
    valid = _args(
        "--optim",
        "muon",
        "--precision",
        "fp32",
        "--muon-ortho-dtype",
        "fp32",
        "--strict-muon-fp32",
    )
    assert muon_precision_error(valid) is None

    for argv in (
        ("--optim", "muon", "--strict-muon-fp32"),
        ("--optim", "muon", "--precision", "fp32", "--strict-muon-fp32"),
        ("--precision", "fp32", "--strict-muon-fp32"),
        ("--optim", "adamw", "--muon-ortho-dtype", "fp32"),
    ):
        args = _args(*argv)
        assert muon_precision_error(args) is not None
        with pytest.raises(ValueError):
            build_optimizer(RNNModel(7, 8, 1, n_out=2), args)


def test_explicit_all_scope_selects_only_interior_matrices() -> None:
    model = RNNModel(7, 8, 1, n_out=2, head_layers=2)
    model.extra = torch.nn.Linear(7, 5)
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    groups = muon_param_groups(model, "all")
    assigned = [parameter for group in groups for parameter in group["params"]]
    assert len(assigned) == len({id(parameter) for parameter in assigned})
    assert {id(parameter) for parameter in assigned} == {
        id(parameter) for parameter in model.parameters()
    }

    gate = next(group for group in groups if group.get("blocks") == 3)
    assert {names[id(parameter)] for parameter in gate["params"]} == {
        name for name, parameter in model.named_parameters() if name.startswith("gru.weight_")
    }
    other = next(group for group in groups if group.get("blocks") == 1 and group.get("muon"))
    assert {names[id(parameter)] for parameter in other["params"]} == {
        "head.0.weight",
        "head.extra.weight",
        "extra.weight",
    }
    adam = {names[id(parameter)] for parameter in groups[0]["params"]}
    assert {"inp.weight", "head.2.weight"} <= adam
    adam_parameters = dict(model.named_parameters())
    assert {name for name, parameter in adam_parameters.items() if parameter.ndim == 1} <= adam
    assert {name for name in adam if name not in {"inp.weight", "head.2.weight"}} == {
        name for name, parameter in adam_parameters.items() if parameter.ndim == 1
    }


def test_fp32_orthogonalization_is_explicit_and_default_is_unchanged_on_cpu() -> None:
    gradient = torch.randn(3, 8, 8, generator=torch.Generator().manual_seed(21))
    torch.testing.assert_close(
        orthogonalize(gradient), orthogonalize(gradient, calculation_dtype="fp32")
    )
    with pytest.raises(ValueError, match="requires FP32 gradients"):
        orthogonalize(gradient.double(), calculation_dtype="fp32")

    parameter = torch.nn.Parameter(torch.randn(8, 8, generator=torch.Generator().manual_seed(22)))
    parameter.grad = gradient[0].clone()
    optimizer = MuonAdamW(
        [{"params": [parameter], "muon": True, "blocks": 1, "muon_fp32_ortho": True}],
        lr=0.01,
        weight_decay=0.0,
    )
    before = parameter.detach().clone()
    optimizer.step()
    expected = before - 0.01 * 0.2 * math.sqrt(8) * orthogonalize(
        0.19 * gradient[0], calculation_dtype="fp32"
    )
    torch.testing.assert_close(parameter, expected, rtol=1e-5, atol=2e-7)
    assert optimizer.state_dict()["param_groups"][0]["muon_fp32_ortho"] is True


def test_optimizer_switch_preserves_fp32_policy_and_scope() -> None:
    args = _args(
        "--optim",
        "adamw",
        "--optim-switch",
        "0.5:muon",
        "--muon-scope",
        "all",
        "--precision",
        "fp32",
        "--muon-ortho-dtype",
        "fp32",
        "--strict-muon-fp32",
    )
    model = RNNModel(7, 8, 1, n_out=2)
    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer, args, 10)
    switched, _ = switch_optimizer(optimizer, scheduler, "muon")
    assert isinstance(switched, MuonAdamW)
    assert all(
        group.get("muon_fp32_ortho") is True for group in switched.param_groups if group["muon"]
    )


def test_strict_resume_rejects_checkpoint_that_downgrades_orthogonalization() -> None:
    strict_args = _args(
        "--optim",
        "muon",
        "--precision",
        "fp32",
        "--muon-ortho-dtype",
        "fp32",
        "--strict-muon-fp32",
    )
    model = RNNModel(7, 8, 1, n_out=2)
    strict_optimizer = build_optimizer(model, strict_args)
    legacy_optimizer = build_optimizer(RNNModel(7, 8, 1, n_out=2), _args("--optim", "muon"))

    assert (
        muon_resume_policy_error(
            strict_args, strict_optimizer.param_groups, strict_optimizer.state_dict()
        )
        is None
    )
    assert (
        muon_resume_policy_error(
            strict_args, strict_optimizer.param_groups, legacy_optimizer.state_dict()
        )
        is not None
    )
    assert (
        muon_resume_policy_error(
            _args("--optim", "muon"), legacy_optimizer.param_groups, legacy_optimizer.state_dict()
        )
        is None
    )
