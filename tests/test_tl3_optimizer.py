"""CPU parity checks for the recovered TL3 Muon/AdamW update semantics."""

from __future__ import annotations

import copy
from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tests.training_dynamics_helpers import options
from wnn.training.cli_optimizer import add_optimizer_arguments
from wnn.training.source_identity import training_source_identity
from wnn.training.tl3_optimizer import _CompatMuon, _orthogonalize
from wnn.training_dynamics import build_optimizer, build_scheduler


class _TinyModel(nn.Module):
    """Expose the legacy input, hidden, output, vector and readout parameter roles."""

    def __init__(self) -> None:
        super().__init__()
        self.inp = nn.Linear(3, 4)
        self.gru = nn.GRU(4, 4, batch_first=True)
        self.head = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
        self.mean = nn.Parameter(torch.zeros(2))
        self.std = nn.Parameter(torch.ones(2))


def _set_grads(model: nn.Module, step: int) -> None:
    """Assign deterministic FP32 gradients without drawing global random state."""
    for index, (_, parameter) in enumerate(model.named_parameters()):
        values = torch.arange(parameter.numel(), dtype=torch.float32).reshape_as(parameter)
        parameter.grad = (values + index * 3 + step) * 0.0007 - 0.03


def _named(model: nn.Module) -> dict[str, nn.Parameter]:
    return dict(model.named_parameters())


def test_tl3_optimizer_flags_are_explicit() -> None:
    """Expose the recovered optimizer and independent Muon learning-rate scale."""
    parser = ArgumentParser()
    add_optimizer_arguments(parser)
    args = parser.parse_args(["--optim", "tl3-muon", "--muon-lr-mult", "6"])
    assert args.optim == "tl3-muon"
    assert args.muon_lr_mult == 6


def test_training_identity_binds_tl3_dispatch_and_optimizer_sources() -> None:
    """Bind both the router and semantic modules into the run source identity."""
    root = Path(__file__).resolve().parents[1]
    environment = SimpleNamespace(source_root=root, entrypoint=root / "scripts/train_rnn.py")
    files = training_source_identity(environment)["files"]
    required = {
        "src/wnn/training_dynamics.py",
        "src/wnn/training/cli_optimizer.py",
        "src/wnn/training/tl3_optimizer.py",
        "src/wnn/training/tl3_scheduler.py",
    }
    assert required <= files.keys()
    assert all(len(files[name]) == 64 for name in required)


def _assert_models_equal(left: nn.Module, right: nn.Module) -> None:
    for name, parameter in _named(left).items():
        torch.testing.assert_close(parameter, _named(right)[name], rtol=0, atol=0)


def _assert_pair_states_equal(
    left_optimizers: tuple[torch.optim.Optimizer, torch.optim.Optimizer],
    right_optimizer: torch.optim.Optimizer,
    left_model: nn.Module,
    right_model: nn.Module,
) -> None:
    for name, parameter in _named(left_model).items():
        left_state = next(
            optimizer.state[parameter]
            for optimizer in left_optimizers
            if parameter in optimizer.state
        )
        right_state = right_optimizer.state[_named(right_model)[name]]
        assert left_state.keys() == right_state.keys()
        for key, value in left_state.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, right_state[key], rtol=0, atol=0)
            else:
                assert value == right_state[key]


def _assert_single_states_equal(
    left: torch.optim.Optimizer,
    right: torch.optim.Optimizer,
    left_model: nn.Module,
    right_model: nn.Module,
) -> None:
    """Compare all named parameter state across a save-and-restore pair."""
    for name, parameter in _named(left_model).items():
        left_state = left.state.get(parameter, {})
        right_state = right.state.get(_named(right_model)[name], {})
        assert left_state.keys() == right_state.keys()
        for key, value in left_state.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, right_state[key], rtol=0, atol=0)
            else:
                assert value == right_state[key]


def _native_pair(
    model: nn.Module, *, lr: float, multiplier: float, beta2: float, weight_decay: float
) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
    """Build the two reference optimizers with Torch's native Muon contract."""
    named = _named(model)
    muon_names = [
        name
        for name, parameter in named.items()
        if parameter.ndim == 2
        and not name.startswith("inp.")
        and not name.startswith("head.2.")
        and name not in {"mean", "std"}
    ]
    adam_names = [name for name in named if name not in muon_names]
    muon = torch.optim.Muon(
        [named[name] for name in muon_names],
        lr=lr * multiplier,
        weight_decay=weight_decay,
        momentum=0.95,
        nesterov=True,
        ns_coefficients=(3.4445, -4.775, 2.0315),
        eps=1e-7,
        ns_steps=5,
        adjust_lr_fn="match_rms_adamw",
    )
    adamw = torch.optim.AdamW(
        [named[name] for name in adam_names],
        lr=lr,
        betas=(0.9, beta2),
        weight_decay=weight_decay,
        fused=False,
    )
    return muon, adamw


@pytest.mark.parametrize("multiplier", [3.0, 6.0])
def test_native_muon_adamw_parity_schedule_and_state(multiplier: float) -> None:
    """Match native two-optimizer updates and OneCycles, including mid-run restore."""
    if not hasattr(torch.optim, "Muon"):
        pytest.skip("native Torch Muon is unavailable in this PyTorch version")
    torch.manual_seed(41)
    reference = _TinyModel()
    candidate = copy.deepcopy(reference)
    resumed = copy.deepcopy(reference)
    lr, beta2, weight_decay, total = 0.002, 0.95, 0.01, 8
    reference_muon, reference_adamw = _native_pair(
        reference, lr=lr, multiplier=multiplier, beta2=beta2, weight_decay=weight_decay
    )
    muon_schedule = torch.optim.lr_scheduler.OneCycleLR(
        reference_muon,
        max_lr=lr * multiplier,
        total_steps=total,
        pct_start=0.25,
        final_div_factor=100,
    )
    adamw_schedule = torch.optim.lr_scheduler.OneCycleLR(
        reference_adamw,
        max_lr=lr,
        total_steps=total,
        pct_start=0.25,
        final_div_factor=100,
    )
    args = options(
        optim="tl3-muon",
        lr=lr,
        beta2=beta2,
        wd=weight_decay,
        muon_lr_mult=multiplier,
        fused_adam=False,
        precision="bf16",
        warmup_frac=0.25,
    )
    optimizer = build_optimizer(candidate, args)
    scheduler = build_scheduler(optimizer, args, total)
    resumed_optimizer = build_optimizer(resumed, args)
    resumed_scheduler = build_scheduler(resumed_optimizer, args, total)
    assert scheduler.get_last_lr() == muon_schedule.get_last_lr() + adamw_schedule.get_last_lr()
    for step in range(total):
        _set_grads(reference, step)
        _set_grads(candidate, step)
        if step >= 4:
            _set_grads(resumed, step)
        reference_muon.step()
        reference_adamw.step()
        optimizer.step()
        if step >= 4:
            resumed_optimizer.step()
        muon_schedule.step()
        adamw_schedule.step()
        scheduler.step()
        if step >= 4:
            resumed_scheduler.step()
        if step == 3:
            resumed.load_state_dict(candidate.state_dict())
            resumed_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
            resumed_scheduler.load_state_dict(copy.deepcopy(scheduler.state_dict()))
        _assert_models_equal(reference, candidate)
        _assert_pair_states_equal(
            (reference_muon, reference_adamw), optimizer, reference, candidate
        )
        assert scheduler.get_last_lr() == pytest.approx(
            muon_schedule.get_last_lr() + adamw_schedule.get_last_lr(), rel=0, abs=0
        )
        if step >= 4:
            _assert_models_equal(candidate, resumed)
            _assert_single_states_equal(optimizer, resumed_optimizer, candidate, resumed)
            assert resumed_scheduler.get_last_lr() == scheduler.get_last_lr()
    candidate_names = optimizer.param_groups[0]["param_names"]
    assert "head.0.weight" in candidate_names
    assert "inp.weight" not in candidate_names and "head.2.weight" not in candidate_names
    assert "momentum_buffer" in optimizer.state[_named(candidate)["gru.weight_ih_l0"]]
    assert "exp_avg_sq" in optimizer.state[_named(candidate)["inp.weight"]]


def test_torch_28_fallback_keeps_two_scheduler_and_optimizer_interfaces(monkeypatch) -> None:
    """Exercise the compatibility Muon path used when torch.optim.Muon is absent."""
    native_muon = getattr(torch.optim, "Muon", None)
    monkeypatch.setattr(torch.optim, "Muon", None, raising=False)
    torch.manual_seed(77)
    reference = _TinyModel()
    candidate = copy.deepcopy(reference)
    lr, multiplier, beta2, weight_decay, total = 0.001, 6.0, 0.99, 0.01, 8
    reference_names = _named(reference)
    muon_names = [
        name
        for name, parameter in reference_names.items()
        if parameter.ndim == 2
        and not name.startswith("inp.")
        and not name.startswith("head.2.")
        and name not in {"mean", "std"}
    ]
    adam_names = [name for name in reference_names if name not in muon_names]
    reference_muon_cls = _CompatMuon if native_muon is None else native_muon
    if native_muon is None:
        reference_muon = reference_muon_cls(
            [reference_names[name] for name in muon_names],
            lr=lr * multiplier,
            weight_decay=weight_decay,
            ns_dtype=torch.bfloat16,
        )
    else:
        reference_muon = reference_muon_cls(
            [reference_names[name] for name in muon_names],
            lr=lr * multiplier,
            weight_decay=weight_decay,
            momentum=0.95,
            nesterov=True,
            ns_coefficients=(3.4445, -4.775, 2.0315),
            eps=1e-7,
            ns_steps=5,
            adjust_lr_fn="match_rms_adamw",
        )
    reference_adamw = torch.optim.AdamW(
        [reference_names[name] for name in adam_names],
        lr=lr,
        betas=(0.9, beta2),
        weight_decay=weight_decay,
        fused=False,
    )
    reference_muon_schedule = torch.optim.lr_scheduler.OneCycleLR(
        reference_muon,
        max_lr=lr * multiplier,
        total_steps=total,
        pct_start=0.25,
        final_div_factor=100,
    )
    reference_adamw_schedule = torch.optim.lr_scheduler.OneCycleLR(
        reference_adamw,
        max_lr=lr,
        total_steps=total,
        pct_start=0.25,
        final_div_factor=100,
    )
    args = options(
        optim="tl3-muon",
        lr=lr,
        beta2=beta2,
        wd=weight_decay,
        muon_lr_mult=multiplier,
        fused_adam=False,
        precision="bf16",
        warmup_frac=0.25,
    )
    optimizer = build_optimizer(candidate, args)
    scheduler = build_scheduler(optimizer, args, total)
    assert isinstance(optimizer.muon_optimizer, _CompatMuon)
    for step in range(total):
        _set_grads(reference, step)
        _set_grads(candidate, step)
        reference_muon.step()
        reference_adamw.step()
        optimizer.step()
        reference_muon_schedule.step()
        reference_adamw_schedule.step()
        scheduler.step()
        _assert_models_equal(reference, candidate)
        _assert_pair_states_equal(
            (reference_muon, reference_adamw), optimizer, reference, candidate
        )
        assert scheduler.get_last_lr() == pytest.approx(
            reference_muon_schedule.get_last_lr() + reference_adamw_schedule.get_last_lr(),
            rel=0,
            abs=0,
        )
    assert scheduler.last_epoch == total
    assert optimizer.param_groups[0]["lr"] == pytest.approx(
        optimizer.param_groups[1]["lr"] * multiplier
    )
    assert (
        optimizer.state[_named(candidate)["gru.weight_ih_l0"]]["momentum_buffer"].dtype
        == torch.float32
    )


def test_strict_fp32_newton_schulz_keeps_every_iterate_fp32() -> None:
    """The explicit strict path remains FP32 instead of inheriting native BF16 NS."""
    previous_precision = torch.get_float32_matmul_precision()
    tf32_matmul = torch.backends.cuda.matmul.allow_tf32
    tf32_cudnn = torch.backends.cudnn.allow_tf32
    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        matrix = torch.arange(24, dtype=torch.float32).reshape(6, 4) / 10
        result = _orthogonalize(matrix, dtype=torch.float32)
        assert result.dtype == torch.float32
        assert torch.isfinite(result).all()
        model = _TinyModel()
        args = options(
            optim="tl3-muon",
            lr=0.001,
            beta2=0.99,
            wd=0.01,
            muon_lr_mult=3.0,
            fused_adam=False,
            precision="fp32",
        )
        optimizer = build_optimizer(model, args)
        assert isinstance(optimizer.muon_optimizer, _CompatMuon)
        assert optimizer.muon_optimizer.param_groups[0]["ns_dtype"] is torch.float32
        _set_grads(model, 0)
        optimizer.step()
        assert (
            optimizer.state[_named(model)["gru.weight_ih_l0"]]["momentum_buffer"].dtype
            is torch.float32
        )
    finally:
        torch.set_float32_matmul_precision(previous_precision)
        torch.backends.cuda.matmul.allow_tf32 = tf32_matmul
        torch.backends.cudnn.allow_tf32 = tf32_cudnn
