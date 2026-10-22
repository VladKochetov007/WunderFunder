from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from wnn.training.cli import build_parser
from wnn.training.losses import main_loss
from wnn.training.mean_variance import (
    append_log_variance_output_head,
    configuration_error,
    weighted_label_diagonal_gaussian_nll,
)


def test_label_nll_matches_independent_clipped_masked_oracle() -> None:
    mean = torch.tensor([[[0.5, -0.5], [1.0, 0.0]]], dtype=torch.float64)
    labels = torch.tensor([[[3.0, -1.0], [0.5, -3.0]]], dtype=torch.float64)
    log_variance = torch.tensor([[[5.0, -5.0], [0.0, 0.25]]], dtype=torch.float64)
    valid = torch.tensor([[1.0, 0.0]], dtype=torch.float64)

    actual = weighted_label_diagonal_gaussian_nll(mean, labels, log_variance, valid)
    clipped = [[2.0, -1.0], [0.5, -2.0]]
    predicted = [[0.5, -0.5], [1.0, 0.0]]
    raw_logvar = [[5.0, -5.0], [0.0, 0.25]]
    weights = [[2.0, 1.0], [0.0, 0.0]]
    numerator = 0.0
    denominator = 0.0
    for row in range(2):
        for column in range(2):
            weight = weights[row][column]
            bounded_logvar = min(4.0, max(-4.0, raw_logvar[row][column]))
            residual = predicted[row][column] - clipped[row][column]
            numerator += weight * (math.exp(-bounded_logvar) * residual**2 + bounded_logvar)
            denominator += weight
    expected = numerator / max(denominator, 1e-8)
    assert actual.item() == pytest.approx(expected, abs=1e-6)


def test_main_loss_routes_label_only_nll_without_a_teacher() -> None:
    args = SimpleNamespace(
        mean_variance_objective="nll",
        mean_variance_target_source="labels",
        td_alpha=0.0,
        loss_norm="chunk",
        loss="wmse",
        ce_mse=0.0,
        ce_tilt=False,
        distill_alpha_far=None,
        distill_alpha=0.9,
        distill_boot_h=0,
        distill_reanchor=0.0,
        distill_family_weight=0.0,
        chunk=1,
    )
    labels = torch.tensor([[[1.0, -2.0], [0.5, 1.0]]])
    valid = torch.tensor([[1.0, 0.0]])
    out = torch.tensor([[[0.0, -1.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]]])
    run = SimpleNamespace(
        options=SimpleNamespace(args=args, device=torch.device("cpu")),
        batch=SimpleNamespace(
            ce_w=None,
            codes=None,
            family_targets=None,
            inputs=torch.zeros((1, 2, 1)),
            loss=None,
            main_valid=valid,
            nxt_all=None,
            nxt_label=None,
            out=out,
            out_ema=None,
            s=0,
            sl=slice(0, 2),
            td_a=None,
            teach=None,
            y=labels,
        ),
        network=SimpleNamespace(
            family_at=0,
            initializer_teacher=None,
            moments=None,
        ),
        progress=SimpleNamespace(norm_ema=None),
        optimization=SimpleNamespace(objective=None),
    )

    main_loss(run)

    expected = weighted_label_diagonal_gaussian_nll(out[..., :2], labels, out[..., -2:], valid)
    assert run.batch.loss.item() == pytest.approx(expected.item())


def test_labels_mode_is_explicit_and_preserves_teacher_default() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--name",
            "unit",
            "--mean-variance-head",
            "1",
            "--mean-variance-objective",
            "nll",
            "--mean-variance-target-source",
            "labels",
            "--loss",
            "wmse",
            "--precision",
            "fp32",
        ]
    )
    assert configuration_error(args, controlled_loss=False) is None

    args.distill = "teacher"
    error = configuration_error(args, controlled_loss=False)
    assert error is not None and "forbids teacher" in error

    legacy = parser.parse_args(
        [
            "--name",
            "unit",
            "--mean-variance-head",
            "1",
            "--mean-variance-objective",
            "nll",
            "--loss",
            "wmse",
            "--precision",
            "fp32",
        ]
    )
    assert legacy.mean_variance_target_source == "main_t3_blend"
    error = configuration_error(legacy, controlled_loss=False)
    assert error is not None and "fixed MainT3 blended teacher" in error


def test_output_head_appends_zero_log_variance_rows() -> None:
    torch.manual_seed(4)
    model = torch.nn.Sequential(torch.nn.Identity(), torch.nn.Linear(3, 2))
    inputs = torch.randn(2, 3)
    original = model(inputs).detach().clone()

    append_log_variance_output_head(model)
    extended = model(inputs)

    torch.testing.assert_close(extended[..., :2], original)
    torch.testing.assert_close(extended[..., -2:], torch.zeros((2, 2)))
