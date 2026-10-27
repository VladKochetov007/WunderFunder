from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch

from wnn.galam import GALAM, connectome_112_to_2_config
from wnn.training.galam_adapter import (
    GALAMSequenceAdapter,
    galam_model_settings,
    validate_galam_batch_plan,
)
from wnn.training.sequence_config import load_config
from wnn.training.sequence_model_helpers import detach_state
from wnn.training.sequence_models import build_sequence_model


def _settings() -> dict[str, object]:
    config = replace(
        connectome_112_to_2_config(),
        width=8,
        attention_heads=2,
        attention_history=4,
        feed_forward_width=16,
        relative_bias_width=4,
        conv_kernel_size=3,
    )
    return {"architecture": "galam", "galam": asdict(config)}


def _json_default(value: object) -> list[str]:
    if isinstance(value, frozenset):
        return sorted(value)
    raise TypeError(f"not serializable: {type(value).__name__}")


def test_runtime_adapter_matches_direct_model_and_chunking() -> None:
    torch.manual_seed(92)
    settings = _settings()
    adapter = build_sequence_model(settings).eval()
    assert isinstance(adapter, GALAMSequenceAdapter)
    x = torch.randn(2, 9, 112)
    direct = GALAM(adapter.galam.config).eval()
    direct.load_state_dict(adapter.galam.state_dict())
    with torch.no_grad():
        expected = direct(x)
        prediction, hints, state = adapter(x)
        first, _, carry = adapter(x[:, :3])
        rest, _, carry = adapter(x[:, 3:], detach_state(carry))
    torch.testing.assert_close(prediction, expected, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(torch.cat((first, rest), dim=1), expected, atol=1e-5, rtol=1e-5)
    assert hints.shape == (2, 9, 0)
    assert state.steps_seen == carry.steps_seen == 9


def test_galam_factory_uses_explicit_settings_and_rejects_ignored_options() -> None:
    for extra in ({"hidden": 768}, {"compile_blocks": True}, {"precision_mode": "bf16"}):
        with pytest.raises(ValueError):
            galam_model_settings(_settings() | extra)
    with pytest.raises(ValueError, match="explicit"):
        build_sequence_model({"architecture": "galam"})


def test_config_loader_and_fresh_checkpoint_factory_agree(tmp_path: Path) -> None:
    path = tmp_path / "recipe.json"
    recipe = {"model": _settings(), "training_mask": "all_after_warmup", "batch_size": 2}
    path.write_text(json.dumps(recipe, default=_json_default), encoding="utf-8")
    loaded = load_config(path)
    assert loaded["model"]["hidden"] == 8
    assert loaded["model"]["heads"] == 2
    assert isinstance(build_sequence_model(loaded["model"]), GALAMSequenceAdapter)
    recipe["optimizer"] = "muon"
    path.write_text(json.dumps(recipe, default=_json_default), encoding="utf-8")
    with pytest.raises(ValueError, match="AdamW"):
        load_config(path)


def test_galam_admission_rejects_singleton_full_and_tail_batches(tmp_path: Path) -> None:
    path = tmp_path / "singleton.json"
    path.write_text(json.dumps({"model": _settings()}, default=_json_default), encoding="utf-8")
    with pytest.raises(ValueError, match="batch_size"):
        load_config(path)
    config = {"model": _settings(), "batch_size": 2}
    with pytest.raises(ValueError, match="singleton"):
        validate_galam_batch_plan(config, fit_sequences=3, world_size=1)
    with pytest.raises(ValueError, match="singleton"):
        validate_galam_batch_plan(config, fit_sequences=8, world_size=8)
    with pytest.raises(ValueError, match="padding"):
        validate_galam_batch_plan(config, fit_sequences=7, world_size=2)
    validate_galam_batch_plan(config, fit_sequences=8, world_size=2)
    validate_galam_batch_plan(config | {"batch_size": 3}, fit_sequences=6, world_size=1)


def test_future_suffix_does_not_change_prefix_in_eval_or_training() -> None:
    torch.manual_seed(31)
    original = torch.randn(3, 8, 112)
    changed = original.clone()
    changed[:, 4:] = 10 * torch.randn_like(changed[:, 4:])
    for training in (False, True):
        baseline = build_sequence_model(_settings()).train(training)
        perturbed = build_sequence_model(_settings()).train(training)
        perturbed.load_state_dict(baseline.state_dict())
        expected, _, _ = baseline(original)
        observed, _, _ = perturbed(changed)
        torch.testing.assert_close(expected[:, :4], observed[:, :4], atol=1e-5, rtol=1e-5)


def test_state_detachment_is_explicit_and_reset_has_no_previous_sequence_memory() -> None:
    torch.manual_seed(7)
    adapter = build_sequence_model(_settings()).eval()
    x = torch.randn(2, 3, 112, requires_grad=True)
    _, _, state = adapter(x)
    assert state.lstm_hidden.grad_fn is not None
    detached = detach_state(state)
    assert detached.lstm_hidden.grad_fn is None
    assert detached.steps_seen == state.steps_seen
    with torch.no_grad():
        before, _, _ = adapter(x.detach())
        adapter(10 * x.detach(), detached)
        after, _, _ = adapter(x.detach())
    torch.testing.assert_close(before, after, atol=1e-5, rtol=1e-5)


def test_gradients_reach_both_memories_and_relative_bias_without_fitting() -> None:
    torch.manual_seed(85)
    adapter = build_sequence_model(_settings()).eval()
    output, _, _ = adapter(torch.randn(2, 5, 112))
    output.square().mean().backward()
    parameters = dict(adapter.named_parameters())
    checked = (
        "galam.lstm.weight_ih_l0",
        "galam.global_projection.weight",
        "galam.global_attention.0.relative_bias.hidden.weight",
        "galam.local_attention.0.relative_bias.hidden.weight",
        "galam.fusion.gates.0.weight",
    )
    for name in checked:
        gradient = parameters[name].grad
        assert gradient is not None, name
        assert torch.isfinite(gradient).all(), name
        assert gradient.abs().sum() > 0, name
