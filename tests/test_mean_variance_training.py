from __future__ import annotations

from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

from wnn.distillation import append_family_output_head
from wnn.nn import RNNModel
from wnn.training.cli import build_parser
from wnn.training.environment import TrainingEnvironment
from wnn.training.input_representation import extend_input_projection_state
from wnn.training.losses import main_loss
from wnn.training.mean_variance import (
    append_log_variance_output_head,
    configuration_error,
    extend_mean_variance_state,
    weighted_diagonal_gaussian_nll,
)
from wnn.training.setup_model import build_model, load_initializer
from wnn.training.state import TrainingRun


def _options(objective: str = "nll", head: int = 1) -> Namespace:
    return Namespace(
        mean_variance_head=head,
        mean_variance_objective=objective,
        arch="gru",
        cell="gru",
        enc="linear",
        precision="fp32",
        qat="none",
        recurrent_rank=0,
        loss="wmse",
        loss_norm="chunk",
        td_alpha=0.0,
        ema_teacher=0.0,
        eval_ema=False,
        swa_frac=0.0,
        distill="",
        distill_cache_root="target-cache",
        final_distill_initializer=False,
        distill_alpha=0.9,
        distill_alpha_far=None,
        distill_boot_h=0,
        distill_reanchor=0.0,
        distill_cache_scale=1.0,
    )


def _loss_run(objective: str, outputs: torch.Tensor) -> SimpleNamespace:
    labels = torch.tensor([[[0.5, -1.0], [1.0, 0.25]]])
    teacher = torch.tensor([[[0.2, -0.4], [0.7, 0.1]]])
    valid = torch.tensor([[True, False]])
    args = _options(objective)
    args.distill_alpha_far = None
    args.distill_boot_h = 0
    args.distill_reanchor = 0.0
    return SimpleNamespace(
        options=SimpleNamespace(args=args, device=torch.device("cpu")),
        batch=SimpleNamespace(
            ce_w=None,
            codes=None,
            family_targets=None,
            inputs=torch.zeros((1, 2, 1)),
            loss=torch.zeros(()),
            main_valid=valid,
            nxt_all=None,
            nxt_label=None,
            out=outputs,
            out_ema=None,
            s=0,
            sl=slice(0, 2),
            td_a=None,
            teach=teacher,
            y=labels,
        ),
        network=SimpleNamespace(
            family_at=0,
            family_targets=None,
            initializer_teacher=None,
            moments=None,
        ),
        progress=SimpleNamespace(norm_ema=None),
        optimization=SimpleNamespace(objective=None),
    )


def test_configuration_requires_the_head_target_and_supported_recipe() -> None:
    assert configuration_error(_options(), controlled_loss=False) is None
    assert "requires --mean-variance-head" in configuration_error(
        _options(head=0), controlled_loss=False
    )
    no_teacher = _options()
    no_teacher.distill_cache_root = ""
    assert "requires the fixed MainT3" in configuration_error(no_teacher, False)
    changed_target = _options()
    changed_target.distill_alpha = 0.5
    assert "fixed 0.9 teacher" in configuration_error(changed_target, False)
    unsupported = _options()
    unsupported.loss = "corr"
    assert "chunk-normalized --loss wmse" in configuration_error(unsupported, False)
    assert "controlled" in configuration_error(_options(), controlled_loss=True)


def test_zero_logvariance_matches_wmse_and_clips_extreme_values() -> None:
    mean = torch.tensor([[[1.0, -2.0], [0.25, 0.5]]], requires_grad=True)
    target = torch.zeros_like(mean)
    weights = torch.tensor([[[1.0, 2.0], [0.0, 1.0]]])
    zeros = torch.zeros_like(mean)
    expected_wmse = (weights * mean.square()).sum() / weights.sum()
    torch.testing.assert_close(
        weighted_diagonal_gaussian_nll(mean, target, zeros, weights), expected_wmse
    )

    bounded = torch.tensor([[[-100.0, 100.0], [0.0, 0.0]]])
    actual = weighted_diagonal_gaussian_nll(mean, target, bounded, weights)
    clipped = bounded.clamp(-4.0, 4.0)
    expected = (weights * (torch.exp(-clipped) * mean.square() + clipped)).sum() / weights.sum()
    torch.testing.assert_close(actual, expected)


def test_wmse_path_does_not_read_nan_logvariance() -> None:
    output = torch.tensor(
        [[[0.1, -0.2, float("nan"), float("nan")], [0.3, 0.4, float("nan"), float("nan")]]],
        requires_grad=True,
    )
    run = _loss_run("wmse", output)
    main_loss(run)
    assert torch.isfinite(run.batch.loss)
    target = 0.1 * run.batch.y.clamp(-2, 2) + 0.9 * run.batch.teach
    weights = run.batch.y.clamp(-2, 2).abs() * run.batch.main_valid[..., None]
    expected = (weights * (output[..., :2] - target).square()).sum() / weights.sum().clamp_min(1e-8)
    torch.testing.assert_close(run.batch.loss, expected)
    run.batch.loss.backward()
    torch.testing.assert_close(output.grad[..., 2:], torch.zeros_like(output.grad[..., 2:]))


def test_nll_uses_the_same_blended_target_and_original_weights() -> None:
    output = torch.tensor([[[0.1, -0.2, 0.3, -0.4], [0.3, 0.4, 0.1, 0.2]]], requires_grad=True)
    run = _loss_run("nll", output)
    main_loss(run)
    target = 0.1 * run.batch.y.clamp(-2, 2) + 0.9 * run.batch.teach
    weights = run.batch.y.clamp(-2, 2).abs() * run.batch.main_valid[..., None]
    expected = weighted_diagonal_gaussian_nll(output[..., :2], target, output[..., -2:], weights)
    torch.testing.assert_close(run.batch.loss, expected)
    run.batch.loss.backward()
    assert output.grad[..., -2:].abs().sum() > 0


def test_appended_logvariance_rows_are_zero_and_preserve_old_outputs_and_rng() -> None:
    torch.manual_seed(17)
    model = RNNModel(3, hidden=4, layers=1, n_out=6)
    inputs = torch.randn(2, 5, 3)
    before, _ = model(inputs)
    old_state = {name: value.clone() for name, value in model.state_dict().items()}
    rng_before = torch.get_rng_state()
    append_log_variance_output_head(model)
    torch.testing.assert_close(torch.get_rng_state(), rng_before)
    after, _ = model(inputs)
    torch.testing.assert_close(after[..., :6], before)
    torch.testing.assert_close(after[..., -2:], torch.zeros_like(after[..., -2:]))
    for name, value in old_state.items():
        current = model.state_dict()[name]
        if name == "head.2.weight" or name == "head.2.bias":
            torch.testing.assert_close(current[: value.shape[0]], value)
        else:
            torch.testing.assert_close(current, value)


def test_output_state_extension_preserves_family_rows_and_parent_outputs() -> None:
    parent = RNNModel(3, hidden=4, layers=1, n_out=5).state_dict()
    target_model = RNNModel(3, hidden=4, layers=1, n_out=5)
    append_family_output_head(target_model)
    append_log_variance_output_head(target_model)
    initialized = target_model.state_dict()
    merged = extend_mean_variance_state(parent, initialized, family_outputs=8)
    torch.testing.assert_close(merged["head.2.weight"][:5], parent["head.2.weight"])
    torch.testing.assert_close(merged["head.2.bias"][:5], parent["head.2.bias"])
    torch.testing.assert_close(merged["head.2.weight"][5:13], initialized["head.2.weight"][5:13])
    torch.testing.assert_close(
        merged["head.2.weight"][13:], torch.zeros_like(merged["head.2.weight"][13:])
    )
    torch.testing.assert_close(
        merged["head.2.bias"][13:], torch.zeros_like(merged["head.2.bias"][13:])
    )


def test_output_state_extension_rejects_non_output_shape_changes() -> None:
    parent_model = RNNModel(3, hidden=4, layers=1, n_out=4)
    target_model = RNNModel(4, hidden=4, layers=1, n_out=4)
    append_log_variance_output_head(target_model)
    with pytest.raises(ValueError, match="outside output head"):
        extend_mean_variance_state(parent_model.state_dict(), target_model.state_dict())


def test_47_output_parent_extension_preserves_every_existing_auxiliary_row() -> None:
    torch.manual_seed(19)
    parent = RNNModel(211, hidden=8, layers=1, n_out=47)
    child = RNNModel(467, hidden=8, layers=1, n_out=47)
    append_log_variance_output_head(child)
    parent_state = extend_input_projection_state(parent.state_dict(), child.state_dict())
    merged = extend_mean_variance_state(parent_state, child.state_dict())

    inputs = torch.randn(2, 5, 211)
    extended_inputs = torch.cat((inputs, torch.zeros(2, 5, 256)), dim=-1)
    parent_output, _ = parent(inputs)
    child.load_state_dict(merged)
    child_output, _ = child(extended_inputs)
    torch.testing.assert_close(child_output[..., :47], parent_output)
    torch.testing.assert_close(child_output[..., -2:], torch.zeros_like(child_output[..., -2:]))


def test_main_auxiliary_layout_and_representation_warmstart_extend_parent() -> None:
    parser = build_parser()
    parent_args = parser.parse_args(
        [
            "--name",
            "synthetic-parent",
            "--hidden",
            "8",
            "--layers",
            "1",
            "--aux-struct",
            "0.3",
            "--aux-next",
            "0.3",
        ]
    )
    parent_output_width = 2 + 12 + 3 + 4 + 2
    torch.manual_seed(23)
    parent_model = RNNModel(211, hidden=8, layers=1, n_out=parent_output_width)
    checkpoint = {
        "args": vars(parent_args),
        "model": parent_model.state_dict(),
        "epoch": 5,
    }

    child_parser = build_parser()
    child_args = child_parser.parse_args(
        [
            "--name",
            "synthetic-child",
            "--hidden",
            "8",
            "--layers",
            "1",
            "--init",
            "synthetic-parent@ep5",
            "--aux-struct",
            "0.3",
            "--aux-next",
            "0.3",
            "--representation-mode",
            "base",
            "--mean-variance-head",
            "1",
            "--mean-variance-objective",
            "wmse",
            "--loss",
            "wmse",
            "--precision",
            "fp32",
        ]
    )
    run = TrainingRun(TrainingEnvironment())
    run.options.ap = child_parser
    run.options.args = child_args
    run.options.device = torch.device("cpu")
    run.options.init_checkpoint = checkpoint
    run.data.shifts = []
    run.data.provenance = {}
    build_model(run)
    assert run.network.n_in == 467
    assert run.network.n_out == parent_output_width + 2
    assert run.network.family_at == parent_output_width

    inputs = torch.randn(2, 5, 211)
    extended_inputs = torch.cat((inputs, torch.zeros(2, 5, 256)), dim=-1)
    expected, _ = parent_model(inputs)
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr("wnn.training.setup_model.file_sha256", lambda _path: "0" * 64)
        load_initializer(run)
    actual, _ = run.network.model(extended_inputs)
    torch.testing.assert_close(actual[..., :parent_output_width], expected)
    torch.testing.assert_close(actual[..., -2:], torch.zeros_like(actual[..., -2:]))
