"""Synthetic-only correctness checks for the E481 candidate-shape actor."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
E474_DIR = ROOT / "research/campaign_2026-10-05/activation_placement"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(E474_DIR))
import fit_activation_placement as e474  # noqa: E402
from fit_recurrent_curvature import (  # noqa: E402
    FEATURES,
    HIDDEN,
    MUON_NAMES,
    N_PARAMETERS,
    SEEDS,
    SHAPES,
    CurvatureRNN,
    candidate_transform,
    initialize_arm,
    make_optimizer,
    scripted_shape_scan,
    seed_states,
)


def _close(a: torch.Tensor, b: torch.Tensor, atol: float = 5e-5, rtol: float = 5e-4) -> None:
    assert torch.isfinite(a).all()
    assert torch.isfinite(b).all()
    torch.testing.assert_close(a, b, atol=atol, rtol=rtol)


def test_scalar_shape_bounds_derivatives_and_tails() -> None:
    """Prove both formulae stay bounded/monotone across the closed domain."""
    u = torch.linspace(-1.0, 1.0, 20_001, dtype=torch.float64, requires_grad=True)
    expected = {
        0: u,
        1: u + 0.1 * torch.sin(math.pi * u),
        2: u + 0.1 * math.pi * u * (1 - u.square()),
    }
    minima = {1: 1 - 0.1 * math.pi, 2: 1 - 0.2 * math.pi}
    for code in range(3):
        got = candidate_transform(u, code)
        torch.testing.assert_close(got, expected[code], atol=1e-12, rtol=1e-12)
        assert float(got.detach().min()) >= -1.0 - 1e-12
        assert float(got.detach().max()) <= 1.0 + 1e-12
        derivative = torch.autograd.grad(got.sum(), u, retain_graph=True)[0]
        assert torch.isfinite(derivative).all()
        if code:
            assert float(derivative.min()) >= minima[code] - 1e-12
            assert float(derivative.min()) > 0
    for coordinate in (-1.0, 0.0, 1.0):
        value = torch.tensor([coordinate], dtype=torch.float64)
        assert float(candidate_transform(value, 1)) == coordinate
        assert float(candidate_transform(value, 2)) == coordinate
    center = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)
    for code in (1, 2):
        derivative = torch.autograd.grad(
            candidate_transform(center, code), center, retain_graph=True
        )[0]
        torch.testing.assert_close(
            derivative, torch.tensor([1 + 0.1 * math.pi], dtype=torch.float64)
        )
    for v in (-20.0, -8.0, 8.0, 20.0):
        saturated = torch.tanh(torch.tensor([v], dtype=torch.float64))
        for code in range(3):
            assert torch.isfinite(candidate_transform(saturated, code)).all()


def test_baseline_matches_native_gru_outputs_states_and_gradients() -> None:
    """E481 baseline scan equals the inherited native reset-after GRU."""
    torch.manual_seed(48101)
    baseline = e474.PlacementRNN("gelu", "gelu").train()
    independent = CurvatureRNN("baseline").train()
    independent.load_state_dict(baseline.state_dict(), strict=True)
    x0 = torch.randn(2, 17, FEATURES, requires_grad=True)
    x1 = x0.detach().clone().requires_grad_(True)
    y0, s0 = baseline(x0)
    y1, s1 = independent(x1)
    _close(y0, y1, atol=2e-5, rtol=2e-4)
    _close(s0, s1, atol=2e-5, rtol=2e-4)
    (y0.square().mean() + s0.square().mean()).backward()
    (y1.square().mean() + s1.square().mean()).backward()
    _close(x0.grad, x1.grad)
    base_params, new_params = (
        dict(baseline.named_parameters()),
        dict(independent.named_parameters()),
    )
    assert base_params.keys() == new_params.keys()
    for name in base_params:
        _close(base_params[name].grad, new_params[name].grad)
    assert sum(p.numel() for p in independent.parameters()) == N_PARAMETERS


def test_shared_initialization_chunk_carry_reset_and_causality() -> None:
    """All shapes share the same state dict and causal chunked recurrence."""
    states = seed_states()
    assert set(states) == set(SEEDS)
    for seed in SEEDS:
        models = [initialize_arm(shape, states[seed]).eval() for shape in SHAPES]
        for model in models:
            assert sum(p.numel() for p in model.parameters()) == N_PARAMETERS
            assert torch.equal(model.mean, torch.zeros_like(model.mean))
            assert torch.equal(model.std, torch.ones_like(model.std))
        for model in models[1:]:
            for name, tensor in models[0].state_dict().items():
                assert torch.equal(tensor, model.state_dict()[name]), f"seed={seed}, {name}"
        order = np.arange(24)
        np.random.default_rng(seed).shuffle(order)
        assert np.array_equal(order, np.random.default_rng(seed).permutation(24))
    torch.manual_seed(48102)
    x = torch.randn(2, 37, FEATURES)
    for shape in SHAPES:
        model = CurvatureRNN(shape).eval()
        with torch.no_grad():
            full, full_state = model(x)
            first, carry = model(x[:, :13])
            second, end = model(x[:, 13:], carry)
            reset_second, reset_end = model(x[:, 13:])
            _close(torch.cat((first, second), dim=1), full, atol=2e-6, rtol=2e-5)
            _close(end, full_state, atol=2e-6, rtol=2e-5)
            assert not torch.allclose(reset_second, second)
            changed = x.clone()
            changed[:, 19:] += 4
            changed_y, _ = model(changed)
            torch.testing.assert_close(changed_y[:, :19], full[:, :19], atol=0, rtol=0)
            assert reset_end.shape == (2, 2, HIDDEN)


def _one_step(model, optimizer, x, y):
    optimizer.zero_grad(set_to_none=True)
    prediction, state = model(x)
    loss = prediction.square().mean() + 0.1 * state.square().mean() + 0.01 * (prediction * y).mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_clipped_adamw_and_muon_update_parity_at_baseline() -> None:
    """The alpha-zero shape preserves both optimizer branches after clipping."""
    torch.manual_seed(48103)
    reference = e474.PlacementRNN("gelu", "gelu")
    initial = {name: tensor.detach().clone() for name, tensor in reference.state_dict().items()}
    x = torch.randn(2, 13, FEATURES)
    y = torch.randn(2, 13, 2)
    for optimizer_name in ("adamw", "muon_adamw"):
        ref = e474.PlacementRNN("gelu", "gelu")
        got = CurvatureRNN("baseline")
        ref.load_state_dict(initial, strict=True)
        got.load_state_dict(initial, strict=True)
        opt_ref, opt_got = make_optimizer(ref, optimizer_name), make_optimizer(got, optimizer_name)
        if optimizer_name == "muon_adamw":
            selected = {
                name
                for name, p in got.named_parameters()
                if any(
                    id(p) == id(q)
                    for group in opt_got.param_groups
                    if group.get("muon")
                    for q in group["params"]
                )
            }
            assert selected == MUON_NAMES
        _one_step(ref, opt_ref, x, y)
        _one_step(got, opt_got, x, y)
        for name, value in ref.named_parameters():
            _close(value, dict(got.named_parameters())[name], atol=2e-4, rtol=2e-3)


def test_candidate_scan_formula_and_finite_gradients_for_all_shapes() -> None:
    """Both nonlinear scans keep gradients finite and distinct from the baseline."""
    torch.manual_seed(48104)
    model = CurvatureRNN("baseline")
    torch.randn(2, 8, FEATURES)
    h = torch.randn(2, HIDDEN)
    layer_x = torch.randn(2, 8, HIDDEN, requires_grad=True)
    params = [
        model.gru.weight_ih_l0,
        model.gru.weight_hh_l0,
        model.gru.bias_ih_l0,
        model.gru.bias_hh_l0,
    ]
    results = []
    for code in range(3):
        out, final = scripted_shape_scan(layer_x, h, *params, code)
        loss = out.square().mean() + final.square().mean()
        grad = torch.autograd.grad(loss, (layer_x, *params), retain_graph=True)
        assert all(torch.isfinite(g).all() for g in grad)
        assert any(float(g.abs().sum()) > 0 for g in grad)
        results.append(out.detach())
    assert not torch.allclose(results[0], results[1])
    assert not torch.allclose(results[0], results[2])
    assert not torch.allclose(results[1], results[2])
