from __future__ import annotations

import argparse
import ast
from pathlib import Path

import numpy as np
import pytest
import torch
from objective_math import bootstrap_target, cell_fit, next_pair_change, weighted_wp
from stream_probe import since_from_inputs

from wnn.features import FEATURE_NAMES, event_features_torch
from wnn.nn import weighted_mse_loss

ROOT = Path.cwd()


def executed_branch(
    teach: torch.Tensor | None, mode: str, horizon: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute the actual trainer objective AST, isolated from data/GPU/training."""
    source = (ROOT / "scripts/train_rnn.py").read_text()
    tree = ast.parse(source)
    candidate = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "moments is not None"
        and "norm_ema =" in ast.get_source_segment(source, node)
    )
    y = torch.tensor([[[-8.0, 0.3], [-0.5, 3.0], [1.0, -1.0], [1.0, -1.0]]])
    out = torch.full_like(y, 0.4, requires_grad=True)
    args = argparse.Namespace(
        distill_alpha=0.5,
        distill_alpha_far=None,
        distill_boot_h=horizon,
        td_alpha=0.0,
        loss_norm=mode,
        loss="wmse",
    )
    environment = {
        "torch": torch,
        "moments": None,
        "teach": teach,
        "y": y,
        "out": out,
        "args": args,
        "sl": slice(0, 4),
        "main_valid": torch.tensor([[0.2, 0.4, 0.5, 0.6]]),
        "norm_ema": torch.tensor(37.0),
        "objective": weighted_mse_loss,
        "nxt_label": torch.from_numpy(next_pair_change(y[0].numpy()))[None],
    }
    code = compile(
        ast.fix_missing_locations(ast.Module(body=[candidate], type_ignores=[])),
        "actual_trainer_branch",
        "exec",
    )
    exec(code, environment)
    return environment["loss"], environment["norm_ema"]


def test_running_normalizer_is_bypassed_for_actual_distill_branch() -> None:
    teacher = torch.tensor([[[8.0, 0.2], [-3.0, 1.0], [0.5, -0.5], [0.4, -0.4]]])
    chunk_loss, chunk_ema = executed_branch(teacher, "chunk")
    running_loss, running_ema = executed_branch(teacher, "running")
    assert torch.equal(chunk_loss, running_loss)
    assert chunk_ema.item() == running_ema.item() == 37.0
    y = torch.tensor([[[-2.0, 0.3], [-0.5, 2.0], [1.0, -1.0], [1.0, -1.0]]])
    v = torch.tensor([[0.2, 0.4, 0.5, 0.6]])
    weight = y.abs() * v[..., None]
    expected = (weight * (0.4 - (0.5 * y + 0.5 * teacher.clamp(-2, 2))) ** 2).sum() / weight.sum()
    torch.testing.assert_close(running_loss, expected)


def test_running_branch_updates_only_without_distill() -> None:
    chunk_loss, _ = executed_branch(None, "chunk")
    running_loss, ema = executed_branch(None, "running")
    assert ema.item() != 37.0
    assert not torch.equal(chunk_loss, running_loss)


def test_raw_pair_bootstrap_boundaries_and_clipping() -> None:
    y = np.array([[4.0, -3.0], [4.0, -3.0], [4.0, -3.0], [3.0, -3.0], [3.0, -3.0]], np.float32)
    teacher = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 1.0]], np.float32)
    assert np.array_equal(next_pair_change(y), [3, 3, 3, 5, 5])
    target = bootstrap_target(y, teacher, 2, 0.5)
    expected = 0.5 * np.clip(y, -2, 2) + 0.5 * teacher
    expected[0] = 0.5 * teacher[2] + 0.5 * teacher[0]
    np.testing.assert_allclose(target, expected)
    assert not np.array_equal(next_pair_change(y), next_pair_change(np.clip(y, -2, 2)))


def test_since_reconstruction_matches_actual_input_feature() -> None:
    x = torch.zeros(1, 24, 112)
    x[0, 5:, 0] = 1
    x[0, 5:, 11] = 2
    x[0, 10:, 0] = 2  # bid only: trainer's ev_i0px is AND, not OR
    x[0, 17:, 0] = 3
    x[0, 17:, 11] = 4
    actual = torch.expm1(event_features_torch(x)[0, :, FEATURE_NAMES.index("since_i0px")])
    reconstructed = since_from_inputs(x[0, :, 0].numpy(), x[0, :, 11].numpy())
    np.testing.assert_allclose(reconstructed, actual.numpy(), rtol=2e-6, atol=2e-6)


def test_full_main_and_ema_gradient_factorization() -> None:
    prediction = torch.tensor([[[0.3, 0.4], [0.5, 0.6]]], requires_grad=True)
    y = torch.tensor([[[1.0, -2.0], [0.2, 0.7]]])
    teacher = torch.tensor([[[0.2, -0.2], [0.1, 0.3]]])
    ema = torch.tensor([[[0.4, -0.1], [0.3, 0.2]]])
    v = torch.tensor([[0.2, 0.8]])
    w = y.abs() * v[..., None]
    target = 0.5 * y + 0.5 * teacher
    loss = (w * (prediction - target) ** 2).sum() / w.sum() + (
        v[..., None] * (prediction - ema) ** 2
    ).sum() / v.sum()
    loss.backward()
    expected = (
        2 * w * (prediction.detach() - target) / w.sum()
        + 2 * v[..., None] * (prediction.detach() - ema) / v.sum()
    )
    torch.testing.assert_close(prediction.grad, expected)


def test_equal_mass_estimator_identity_and_wp_affine_feasibility() -> None:
    x = np.tile(np.arange(3), 4)
    y = np.array([-1.0, 0.5, 1.0])[x]
    weight = np.abs(y)
    pooled = cell_fit(x, y, weight, np.ones(len(y)))
    scaled = cell_fit(x, y, weight, np.full(len(y), 7.0))
    np.testing.assert_allclose(pooled, scaled)
    assert weighted_wp(y, y) == pytest.approx(1.0)
    assert weighted_wp(y, 0.5 * y + 0.1) == pytest.approx(1.0)
    # Global positive affine transforms cease being WP-invariant after saturation.
    assert weighted_wp(y, 4 * y) < 0.999


def test_sham_preserves_mass_multiset_and_aligned_pooled_gradient() -> None:
    from estimator_control import mass_multipliers

    mass = np.array([[1.0, 2.0, 3.0], [4.0, 2.0, 5.0]])
    gradient = np.arange(12).reshape(2, 3, 2).astype(float)
    aligned = mass_multipliers(mass)
    shuffled = mass_multipliers(mass, 9000)
    assert np.array_equal(np.sort(aligned.ravel()), np.sort(shuffled.ravel()))
    assert aligned.mean() == pytest.approx(1)
    assert not np.array_equal(aligned, shuffled)
    np.testing.assert_allclose(
        gradient / mass[..., None] * aligned[..., None], gradient / mass.mean()
    )
    np.testing.assert_array_equal(mass_multipliers(np.ones(20), 9000), np.ones(20))
    with pytest.raises(ValueError):
        mass_multipliers(np.array([0.0, 1.0]))


def test_actual_distilled_bootstrap_matches_stream_reconstruction() -> None:
    teacher = torch.tensor([[[8.0, 0.2], [-3.0, 1.0], [0.5, -0.5], [0.4, -0.4]]])
    loss, unchanged = executed_branch(teacher, "running", horizon=1)
    y = np.array([[-8.0, 0.3], [-0.5, 3.0], [1.0, -1.0], [1.0, -1.0]], np.float32)
    target = torch.from_numpy(bootstrap_target(y, teacher[0].numpy(), 1, 0.5))[None]
    clipped = torch.from_numpy(np.clip(y, -2, 2))[None]
    v = torch.tensor([[0.2, 0.4, 0.5, 0.6]])
    weight = clipped.abs() * v[..., None]
    expected = (weight * (0.4 - target) ** 2).sum() / weight.sum()
    torch.testing.assert_close(loss, expected)
    assert unchanged.item() == 37.0
