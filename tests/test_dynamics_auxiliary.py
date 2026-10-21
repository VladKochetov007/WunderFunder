from __future__ import annotations

from pathlib import Path

import torch

from scripts.package_submission import load_checkpoint_member
from wnn.data import N_FEATURES
from wnn.features import N_EVENT_FEATURES
from wnn.kalman_group_features import FEATURE_GROUPS
from wnn.nn import RNNModel
from wnn.training.dynamics_auxiliary import (
    DYNAMICS_TARGET_WIDTH,
    HORIZONS,
    LOG_EPSILON,
    TARGETS_PER_HORIZON,
    dynamics_huber_loss,
    dynamics_targets_torch,
    strip_dynamics_auxiliary_state,
)
from wnn.training.environment import TrainingEnvironment
from wnn.training.inputs import build_inputs


def _direct_group_targets(
    raw: torch.Tensor,
    input_mean: torch.Tensor,
    input_std: torch.Tensor,
    start: int,
    horizon: int,
) -> torch.Tensor:
    """Compute one horizon directly in float64 as a small-window oracle."""
    normalized = (raw[..., :N_FEATURES].double() - input_mean.double()) / input_std.double()
    changes = normalized[start + 1 : start + horizon + 1] - normalized[start : start + horizon]
    values: list[torch.Tensor] = []
    for _, feature_start, feature_stop in FEATURE_GROUPS:
        group_changes = changes[:, feature_start:feature_stop]
        values.extend(
            (
                group_changes.mean(),
                torch.log(group_changes.square().mean() + LOG_EPSILON),
            )
        )
    return torch.stack(values)


def test_future_targets_have_exact_group_values_and_endpoint_masks() -> None:
    raw = torch.arange(300, dtype=torch.float32)[None, :, None].expand(-1, -1, N_FEATURES)
    valid = torch.arange(300)[None, :] >= 5

    target, mask = dynamics_targets_torch(
        raw, torch.zeros(N_FEATURES), torch.ones(N_FEATURES), valid
    )

    assert target.shape == (1, 300, DYNAMICS_TARGET_WIDTH)
    assert mask.shape == (1, 300, 2)
    torch.testing.assert_close(target[0, 5, 0::2], torch.ones(26))
    torch.testing.assert_close(target[0, 5, 1::2], torch.zeros(26), atol=1e-6, rtol=0)
    assert mask[0, 5].all()
    assert mask[0, 283, 0]
    assert not mask[0, 284, 0]
    assert mask[0, 43, 1]
    assert not mask[0, 44, 1]

    _, compressed_mask = dynamics_targets_torch(
        raw,
        torch.zeros(N_FEATURES),
        torch.ones(N_FEATURES),
        valid,
        lengths=torch.tensor([280]),
    )
    assert compressed_mask[0, 263, 0]
    assert not compressed_mask[0, 264, 0]
    assert compressed_mask[0, 23, 1]
    assert not compressed_mask[0, 24, 1]


def test_late_small_energy_survives_an_early_large_pulse() -> None:
    sequence_length = 20_000
    raw = torch.zeros(1, sequence_length, N_FEATURES)
    raw[:, 1, :] = 1_000.0
    late_steps = torch.arange(sequence_length - 10_000, dtype=torch.float32)
    raw[:, 10_000:, :] = (late_steps.remainder(2) * 0.01)[None, :, None]
    input_mean = torch.zeros(N_FEATURES)
    input_std = torch.ones(N_FEATURES)
    valid_rows = torch.ones(1, sequence_length, dtype=torch.bool)

    target, mask = dynamics_targets_torch(raw, input_mean, input_std, valid_rows)

    position = 16_000
    for horizon_index, horizon in enumerate(HORIZONS):
        assert mask[0, position, horizon_index]
        expected = _direct_group_targets(raw[0], input_mean, input_std, position, horizon)
        start = horizon_index * TARGETS_PER_HORIZON
        stop = start + TARGETS_PER_HORIZON
        torch.testing.assert_close(
            target[0, position, start:stop].double(), expected, atol=2e-6, rtol=2e-6
        )


def test_local_window_sums_match_direct_oracle_at_block_and_mask_boundaries() -> None:
    sequence_length = 541
    generator = torch.Generator(device="cpu").manual_seed(7301)
    raw = torch.randn(2, sequence_length, N_FEATURES, generator=generator)
    feature_scale = torch.linspace(0.25, 2.5, N_FEATURES)
    raw *= feature_scale
    input_mean = torch.linspace(-0.4, 0.6, N_FEATURES)
    input_std = torch.linspace(0.35, 1.8, N_FEATURES)
    valid_rows = torch.ones(2, sequence_length, dtype=torch.bool)
    valid_rows[0, 17] = False
    valid_rows[1, 40] = False
    lengths = torch.tensor([sequence_length, 300])

    target, mask = dynamics_targets_torch(
        raw, input_mean, input_std, valid_rows, lengths=lengths
    )
    positions = torch.arange(sequence_length)

    for horizon_index, horizon in enumerate(HORIZONS):
        expected_mask = valid_rows & (positions[None, :] + horizon < sequence_length)
        expected_mask &= positions[None, :] + horizon < lengths[:, None]
        assert torch.equal(mask[..., horizon_index], expected_mask)
        start = horizon_index * TARGETS_PER_HORIZON
        stop = start + TARGETS_PER_HORIZON
        for batch_index, length in enumerate(lengths.tolist()):
            boundary = length - horizon - 1
            if boundary >= 0:
                assert mask[batch_index, boundary, horizon_index]
                assert not mask[batch_index, boundary + 1, horizon_index]
            selected = {
                0,
                horizon - 1,
                horizon,
                horizon + 1,
                2 * horizon - 1,
                2 * horizon,
                sequence_length - horizon - 1,
                boundary,
            }
            for position in selected:
                if (
                    position < 0
                    or position + horizon >= length
                    or not valid_rows[batch_index, position]
                ):
                    continue
                expected = _direct_group_targets(
                    raw[batch_index], input_mean, input_std, position, horizon
                )
                torch.testing.assert_close(
                    target[batch_index, position, start:stop].double(),
                    expected,
                    atol=2e-6,
                    rtol=2e-6,
                )

    short_target, short_mask = dynamics_targets_torch(
        raw[:1, :12],
        input_mean,
        input_std,
        valid_rows[:1, :12],
    )
    assert not short_mask.any()
    assert torch.isfinite(short_target).all()


def test_future_changes_do_not_change_causal_inputs_or_model_prefix() -> None:
    torch.manual_seed(812)
    model = RNNModel(
        N_FEATURES + N_EVENT_FEATURES,
        hidden=4,
        layers=1,
        n_out=2,
        dynamics_aux_head=True,
    ).eval()
    raw = torch.randn(1, 2300, N_FEATURES)
    changed = raw.clone()
    changed[:, 2001:] += 3.0
    environment = TrainingEnvironment()
    first_inputs = build_inputs(raw, environment=environment)
    changed_inputs = build_inputs(changed, environment=environment)
    first_output, _ = model(first_inputs)
    changed_output, _ = model(changed_inputs)
    first_target, first_mask = dynamics_targets_torch(
        raw,
        model.mean[:N_FEATURES],
        model.std[:N_FEATURES],
        torch.ones(raw.shape[:2], dtype=torch.bool),
    )
    changed_target, changed_mask = dynamics_targets_torch(
        changed,
        model.mean[:N_FEATURES],
        model.std[:N_FEATURES],
        torch.ones(raw.shape[:2], dtype=torch.bool),
    )

    torch.testing.assert_close(first_inputs[:, :2001], changed_inputs[:, :2001])
    torch.testing.assert_close(first_output[:, :2001], changed_output[:, :2001])
    assert not torch.equal(first_target[:, 1990], changed_target[:, 1990])
    assert first_mask[0, 1990, 1] and changed_mask[0, 1990, 1]
    assert first_target.shape[1] == 2300
    assert first_target[0, 1990, 26] != changed_target[0, 1990, 26]
    assert first_output.shape[-1] == 2


def test_huber_mean_ignores_invalid_horizon_rows() -> None:
    prediction = torch.zeros(1, 3, DYNAMICS_TARGET_WIDTH)
    target = torch.full_like(prediction, 100.0)
    target[0, 0, :26] = 1.0
    target[0, 1, 26:] = 1.0
    valid = torch.zeros(1, 3, 2, dtype=torch.bool)
    valid[0, 0, 0] = True
    valid[0, 1, 1] = True

    loss = dynamics_huber_loss(prediction, target, valid)

    torch.testing.assert_close(loss, torch.tensor(0.5))


def test_training_head_keeps_same_seed_main_output_and_is_stripped_for_export(
    tmp_path: Path,
) -> None:
    model_args = {"readout_activation": "gelu"}
    torch.manual_seed(2718)
    control = RNNModel(211, hidden=4, layers=1, n_out=2).train()
    torch.manual_seed(2718)
    treatment = RNNModel(211, hidden=4, layers=1, n_out=2, dynamics_aux_head=True).train()
    inputs = torch.randn(1, 8, 211)
    control_output, _ = control(inputs)
    treatment_output, _ = treatment(inputs)
    torch.testing.assert_close(control_output, treatment_output[..., :2])
    assert treatment_output.shape[-1] == 2 + DYNAMICS_TARGET_WIDTH

    restored = RNNModel(211, hidden=4, layers=1, n_out=2, dynamics_aux_head=True)
    restored.load_state_dict(treatment.state_dict(), strict=True)
    assert set(restored.state_dict()) == set(treatment.state_dict())
    stripped = strip_dynamics_auxiliary_state(treatment.state_dict())
    assert not any("dynamics_aux_head" in name for name in stripped)
    assert "dynamics_target_mean" not in stripped
    assert "dynamics_target_std" not in stripped
    serving = RNNModel(211, hidden=4, layers=1, n_out=2)
    serving.load_state_dict(stripped, strict=True)
    serving.eval()
    treatment.eval()
    torch.testing.assert_close(treatment(inputs)[0], serving(inputs)[0])

    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "candidate"
    run_dir.mkdir(parents=True)
    torch.save(
        {"args": model_args, "n_in": 211, "model": treatment.state_dict()},
        run_dir / "model.pt",
    )
    exported, _ = load_checkpoint_member("candidate", runs_dir, require_provenance=False)

    assert not any("dynamics_aux_head" in name for name in exported)
    assert "dynamics_target_mean" not in exported
    assert exported["head.2.weight"].shape[0] == 2
    assert exported["head.2.bias"].shape[0] == 2
