from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from wnn.data import I0, N_FEATURES
from wnn.features import N_EVENT_FEATURES, event_features_torch
from wnn.nn import RNNModel
from wnn.optim import muon_param_groups
from wnn.tiedlead import (
    LEAD_WIDTH,
    TIEDLEAD_INPUT_WIDTH,
    TIEDLEAD_SCHEMA,
    tiedlead_i1_at_last_i0_price_event,
)
from wnn.training.cli import build_parser
from wnn.training.environment import InputConfiguration, TrainingEnvironment
from wnn.training.inputs import build_inputs
from wnn.training.normalization import fit_normalization
from wnn.training.state import TrainingRun
from wnn.training.validation_training import validate_models


def _raw_with_events(rows: int = 9) -> torch.Tensor:
    raw = torch.zeros(2, rows, N_FEATURES, dtype=torch.float32)
    raw[..., 52:104] = torch.arange(rows * LEAD_WIDTH).reshape(1, rows, LEAD_WIDTH)
    # An i0 price event requires both first bid and ask price coordinates to change.
    raw[0, 3:, I0.bid_px.start] = raw[0, 3:, I0.ask_px.start] = 1.0
    raw[0, 6:, I0.bid_px.start] = raw[0, 6:, I0.ask_px.start] = 2.0
    raw[1, 4:, I0.bid_px.start] = raw[1, 4:, I0.ask_px.start] = 5.0
    return raw


def test_tiedlead_feature_is_causal_and_resets_per_sequence() -> None:
    raw = _raw_with_events()
    events = event_features_torch(raw)
    lead = tiedlead_i1_at_last_i0_price_event(raw, events)

    torch.testing.assert_close(lead[0, :3], raw[0, :1, 52:104].expand(3, -1))
    torch.testing.assert_close(lead[0, 3:6], raw[0, 3:4, 52:104].expand(3, -1))
    torch.testing.assert_close(lead[0, 6:], raw[0, 6:7, 52:104].expand(3, -1))
    torch.testing.assert_close(lead[1, :4], raw[1, :1, 52:104].expand(4, -1))
    torch.testing.assert_close(lead[1, 4:], raw[1, 4:5, 52:104].expand(5, -1))

    changed_future = raw.clone()
    changed_future[:, 7:, :104] += 100.0
    changed_lead = tiedlead_i1_at_last_i0_price_event(
        changed_future, event_features_torch(changed_future)
    )
    torch.testing.assert_close(changed_lead[:, :7], lead[:, :7])

    # Resetting the feature clock at a chunk boundary would lose the row-3 event.
    second_chunk = tiedlead_i1_at_last_i0_price_event(
        raw[:, 4:], event_features_torch(raw[:, 4:])
    )
    assert not torch.equal(second_chunk[0, 0], lead[0, 4])
    torch.testing.assert_close(lead[0, 4], raw[0, 3, 52:104])


def test_tiedlead_input_builder_preserves_raw_event_prefix_and_schema() -> None:
    raw = _raw_with_events()
    env = TrainingEnvironment(inputs=InputConfiguration(tiedlead=True))
    assembled = build_inputs(raw, environment=env)

    assert assembled.shape == (2, raw.shape[1], TIEDLEAD_INPUT_WIDTH)
    assert TIEDLEAD_INPUT_WIDTH == N_FEATURES + N_EVENT_FEATURES + LEAD_WIDTH == 263
    assert TIEDLEAD_SCHEMA == "raw112-event99-lead52-v1"
    torch.testing.assert_close(assembled[..., :N_FEATURES], raw)
    torch.testing.assert_close(
        assembled[..., N_FEATURES : N_FEATURES + N_EVENT_FEATURES],
        event_features_torch(raw),
    )
    torch.testing.assert_close(
        assembled[..., -LEAD_WIDTH:],
        tiedlead_i1_at_last_i0_price_event(raw, event_features_torch(raw)),
    )

    differentiable_raw = raw.clone().requires_grad_(True)
    differentiable = build_inputs(differentiable_raw, environment=env)
    differentiable[..., -LEAD_WIDTH:].sum().backward()
    assert differentiable_raw.grad is not None
    assert torch.isfinite(differentiable_raw.grad).all()


def test_tiedlead_normalization_uses_the_assembled_training_batch() -> None:
    raw = _raw_with_events()
    environment = TrainingEnvironment(inputs=InputConfiguration(tiedlead=True))
    model = RNNModel(TIEDLEAD_INPUT_WIDTH, hidden=8, layers=1, n_out=2, enc="tiedlead")
    loader = DataLoader(TensorDataset(raw), batch_size=2, shuffle=False)

    fit_normalization(model, loader, torch.device("cpu"), batches=1, environment=environment)
    assembled = build_inputs(raw, environment=environment)
    sampled = assembled.reshape(-1, TIEDLEAD_INPUT_WIDTH)[::7]

    assert model.mean.shape == model.std.shape == (TIEDLEAD_INPUT_WIDTH,)
    torch.testing.assert_close(model.mean, sampled.mean(dim=0))
    torch.testing.assert_close(model.std, sampled.std(dim=0).clamp_min(0.001))


def test_tiedlead_model_uses_source_shaped_residuals() -> None:
    model = RNNModel(TIEDLEAD_INPUT_WIDTH, hidden=8, layers=2, n_out=23, enc="tiedlead")
    values = torch.randn(2, 11, TIEDLEAD_INPUT_WIDTH)
    output, state = model(values)

    assert output.shape == (2, 11, 23)
    assert state.shape == (2, 2, 8)
    assert torch.count_nonzero(model.tied_out.weight) == 0
    assert torch.count_nonzero(model.lead_out.weight) == 0
    with pytest.raises(ValueError, match="263-column"):
        RNNModel(211, hidden=8, layers=2, n_out=2, enc="tiedlead")
    with pytest.raises(ValueError, match="263 input columns"):
        model(torch.randn(2, 11, 211))

    groups = muon_param_groups(model)
    muon_ids = {id(parameter) for group in groups if group.get("muon") for parameter in group["params"]}
    adam_ids = {id(parameter) for group in groups if not group.get("muon") for parameter in group["params"]}
    named_parameters = dict(model.named_parameters())
    assert muon_ids.isdisjoint(adam_ids)
    assert muon_ids | adam_ids == {id(parameter) for parameter in model.parameters()}
    assert {id(parameter) for name, parameter in named_parameters.items() if name.startswith("gru.weight_")} <= muon_ids
    assert {id(parameter) for name, parameter in named_parameters.items() if name.startswith(("tied.", "tied_out.", "lead_out."))} <= adam_ids


def test_tiedlead_encoder_is_selectable_from_the_canonical_cli() -> None:
    args = build_parser().parse_args(
        [
            "--name",
            "bounded-check",
            "--device",
            "cpu",
            "--precision",
            "fp32",
            "--enc",
            "tiedlead",
        ]
    )
    assert args.enc == "tiedlead"

    run = TrainingRun(TrainingEnvironment())
    run.options.ap = build_parser()
    run.options.args = args
    validate_models(run)
    assert run.environment.inputs.tiedlead

    half_precision = build_parser().parse_args(
        ["--name", "bounded-check", "--device", "cpu", "--enc", "tiedlead"]
    )
    half_precision_run = TrainingRun(TrainingEnvironment())
    half_precision_run.options.ap = build_parser()
    half_precision_run.options.args = half_precision
    with pytest.raises(SystemExit):
        validate_models(half_precision_run)

    incompatible = build_parser().parse_args(
        [
            "--name",
            "bounded-check",
            "--device",
            "cpu",
            "--enc",
            "tiedlead",
            "--precision",
            "fp32",
            "--hidden",
            "64",
        ]
    )
    invalid_run = TrainingRun(TrainingEnvironment())
    invalid_run.options.ap = build_parser()
    invalid_run.options.args = incompatible
    with pytest.raises(SystemExit):
        validate_models(invalid_run)

    no_propensity_args = build_parser().parse_args(
        [
            "--name",
            "bounded-check",
            "--device",
            "cpu",
            "--precision",
            "fp32",
            "--enc",
            "tiedlead",
            "--propensity",
            "old-propensity",
        ]
    )
    propensity_run = TrainingRun(TrainingEnvironment())
    propensity_run.options.ap = build_parser()
    propensity_run.options.args = no_propensity_args
    with pytest.raises(SystemExit):
        validate_models(propensity_run)
