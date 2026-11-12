from __future__ import annotations

import pytest
import torch

from wnn.training.sequence_weight_ema import SequenceWeightEMA


def test_weight_ema_updates_after_each_optimizer_step_and_swaps_back() -> None:
    model = torch.nn.Linear(2, 1, bias=True)
    ema = SequenceWeightEMA(model, decay=0.75)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(2.0)
    after = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    ema.update(model)

    expected = {
        name: before[name] * 0.75 + after[name] * 0.25
        for name in before
    }
    assert ema.updates == 1
    assert all(torch.equal(ema.shadow[name], expected[name]) for name in expected)

    raw = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    with ema.apply(model):
        assert all(torch.equal(parameter, expected[name]) for name, parameter in model.named_parameters())
    assert all(torch.equal(parameter, raw[name]) for name, parameter in model.named_parameters())


def test_weight_ema_state_round_trip_binds_model_and_update_count() -> None:
    model = torch.nn.Linear(3, 2)
    ema = SequenceWeightEMA(model, decay=0.999)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.5)
    ema.update(model)
    state = ema.state_dict()

    restored = SequenceWeightEMA(model, decay=0.999)
    restored.load_state_dict(state, model, expected_updates=1)
    assert restored.updates == 1
    assert all(torch.equal(restored.shadow[name], ema.shadow[name]) for name in ema.shadow)

    with pytest.raises(ValueError, match="update count"):
        restored.load_state_dict(state, model, expected_updates=2)


def test_weight_ema_rejects_non_fp32_trainable_parameters() -> None:
    model = torch.nn.Linear(2, 2).to(dtype=torch.float64)
    with pytest.raises(TypeError, match="finite FP32"):
        SequenceWeightEMA(model, decay=0.999)
