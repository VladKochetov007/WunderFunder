from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from wnn.training import losses
from wnn.training.auxiliary_schedule import auxiliary_loss_coefficients
from wnn.training.cli_targets import add_targets_arguments


def test_auxiliary_schedule_defaults_to_constant_coefficients() -> None:
    parser = argparse.ArgumentParser()
    add_targets_arguments(parser)
    args = parser.parse_args(["--aux-struct", "0.3", "--aux-next", "0.3"])

    assert args.aux_anneal == "none"
    annealed_args = parser.parse_args(
        ["--aux-struct", "0.3", "--aux-next", "0.3", "--aux-anneal", "late_linear_to_zero"]
    )
    assert annealed_args.aux_anneal == "late_linear_to_zero"
    assert auxiliary_loss_coefficients("none", args.aux_struct, args.aux_next, 8, 9) == (
        0.3,
        0.3,
    )


def test_late_auxiliary_schedule_holds_then_decays_to_zero() -> None:
    assert auxiliary_loss_coefficients("late_linear_to_zero", 0.3, 0.3, 0, 9) == (0.3, 0.3)
    assert auxiliary_loss_coefficients("late_linear_to_zero", 0.3, 0.3, 4, 9) == (0.3, 0.3)
    assert auxiliary_loss_coefficients("late_linear_to_zero", 0.3, 0.3, 6, 9) == pytest.approx(
        (0.15, 0.15)
    )
    assert auxiliary_loss_coefficients("late_linear_to_zero", 0.3, 0.3, 8, 9) == (0.0, 0.0)


def test_default_auxiliary_loss_matches_the_frozen_coefficient_arithmetic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(losses, "structural_loss", lambda *_args: torch.tensor(2.0))
    outputs = torch.tensor([[[0.1, -0.2, 0.0, 0.5, -0.5], [0.3, 0.4, 0.0, -0.2, 0.2]]])
    next_rows = torch.tensor([[4.0, 2.0]])
    next_up = torch.tensor([[1.0, 0.0]])
    next_weight = torch.tensor([[1.0, 2.0]])
    base_loss = torch.tensor(1.25)
    args = SimpleNamespace(
        arch="gru",
        gate_ce=0.0,
        aux_struct=0.3,
        aux_next=0.3,
        aux_tier=0.0,
        aux_mask=0.0,
        aux_sign=0.0,
        aux_quote=0.0,
        aux_weight=0.5,
        ema_teacher=0.0,
    )
    run = SimpleNamespace(
        options=SimpleNamespace(args=args),
        network=SimpleNamespace(
            mask_at=0,
            n_out=2,
            next_at=3,
            quote_at=0,
            sign_at=0,
            struct_at=2,
            tier_at=0,
        ),
        optimization=SimpleNamespace(objective=lambda *_args: torch.tensor(0.0)),
        batch=SimpleNamespace(
            aux=[],
            br_lab=torch.zeros((1, 2), dtype=torch.long),
            cl_lab=torch.zeros((1, 2), dtype=torch.long),
            kth=[],
            loss=base_loss.clone(),
            main_valid=torch.ones((1, 2)),
            mask_rows=torch.ones((1, 2)),
            nw=next_weight,
            nx_rows=next_rows,
            nx_up=next_up,
            out=outputs,
            out_ema=None,
            quote_label=None,
            scored_dev=torch.ones((1, 2)),
            sign_at=0,
            sl=slice(0, 2),
            sp_lab=torch.zeros((1, 2), dtype=torch.long),
            spread_encoded=None,
            sw=torch.ones((1, 2)),
            tier_label=torch.zeros((1, 2), dtype=torch.long),
            tier_weight=torch.ones((1, 2)),
            y=torch.zeros((1, 2, 2)),
        ),
    )

    losses.auxiliary_loss(run)

    mse = (outputs[..., 3] - (next_rows - 3.0) / 2.0) ** 2
    bce = F.binary_cross_entropy_with_logits(outputs[..., 4], next_up, reduction="none")
    expected = base_loss + 0.3 * torch.tensor(2.0)
    expected = expected + 0.3 * ((mse + bce) * next_weight).sum() / next_weight.sum().clamp_min(
        1e-8
    )
    assert torch.equal(run.batch.loss, expected)


@pytest.mark.parametrize(
    ("schedule", "step", "total_steps"),
    [("unknown", 0, 5), ("late_linear_to_zero", 0, 1), ("late_linear_to_zero", 9, 9)],
)
def test_auxiliary_schedule_rejects_invalid_clock(
    schedule: str, step: int, total_steps: int
) -> None:
    with pytest.raises(ValueError):
        auxiliary_loss_coefficients(schedule, 0.3, 0.3, step, total_steps)
