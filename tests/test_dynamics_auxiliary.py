from __future__ import annotations

from pathlib import Path

import torch

from scripts.package_submission import load_checkpoint_member
from wnn.data import N_FEATURES
from wnn.features import N_EVENT_FEATURES
from wnn.nn import RNNModel
from wnn.training.dynamics_auxiliary import (
    DYNAMICS_TARGET_WIDTH,
    dynamics_huber_loss,
    dynamics_targets_torch,
    strip_dynamics_auxiliary_state,
)
from wnn.training.environment import TrainingEnvironment
from wnn.training.inputs import build_inputs


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
