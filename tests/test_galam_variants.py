from __future__ import annotations

from dataclasses import asdict, replace

import pytest
import torch
from torch import nn

from wnn.galam import GALAM, connectome_112_to_2_config
from wnn.training.galam_adapter import GALAMSequenceAdapter, galam_model_settings
from wnn.training.sequence_model_helpers import detach_state
from wnn.training.sequence_models import build_sequence_model


def _settings(variant: str | None = None) -> dict[str, object]:
    """Return a small explicit 112-to-2 GALAM runtime config for CPU checks."""
    config = replace(
        connectome_112_to_2_config(),
        width=8,
        attention_heads=2,
        attention_history=4,
        feed_forward_width=16,
        relative_bias_width=4,
        conv_hidden_width=16,
        conv_kernel_size=3,
    )
    nested: dict[str, object] = asdict(config)
    if variant is not None:
        nested["variant"] = variant
    return {"architecture": "galam", "galam": nested}


@pytest.mark.parametrize(
    ("variant", "recurrent_name", "has_global", "has_cell"),
    [
        ("schematic_gru_full", "gru", True, False),
        ("schematic_lstm_local_only", "lstm", False, True),
    ],
)
def test_variant_streaming_prefix_reset_and_state_detach(
    variant: str, recurrent_name: str, has_global: bool, has_cell: bool
) -> None:
    """Check causal chunk equivalence, sequence reset, and TBPTT state detachment."""
    torch.manual_seed(103)
    adapter = build_sequence_model(_settings(variant)).eval()
    assert isinstance(adapter, GALAMSequenceAdapter)
    assert adapter.variant == variant
    assert isinstance(getattr(adapter.galam, recurrent_name), nn.GRU if has_global else nn.LSTM)
    assert hasattr(adapter.galam, "global_projection") is has_global
    assert hasattr(adapter.galam, "cell_projection") is has_cell

    values = torch.randn(2, 7, 112, requires_grad=True)
    prediction, hints, state = adapter(values)
    first, _, carried = adapter(values[:, :3])
    rest, _, carried = adapter(values[:, 3:], detach_state(carried))
    torch.testing.assert_close(torch.cat((first, rest), dim=1), prediction, atol=1e-5, rtol=1e-5)
    assert prediction.shape == (2, 7, 2)
    assert hints.shape == (2, 7, 0)
    assert state.steps_seen == carried.steps_seen == 7
    assert (state.lstm_cell is not None) is has_cell
    assert state.lstm_hidden.grad_fn is not None
    assert state.detach().lstm_hidden.grad_fn is None
    if not has_cell:
        assert state.lstm_cell is None
        assert state.detach().lstm_cell is None

    changed = values.detach().clone()
    changed[:, 4:] = 20 * torch.randn_like(changed[:, 4:])
    perturbed, _, _ = adapter(changed)
    torch.testing.assert_close(prediction[:, :4], perturbed[:, :4], atol=1e-5, rtol=1e-5)
    fresh_again, _, fresh_state = adapter(values.detach())
    torch.testing.assert_close(prediction.detach(), fresh_again, atol=1e-5, rtol=1e-5)
    assert fresh_state.steps_seen == 7


def test_full_gru_has_real_three_branch_fusion_and_backward_paths() -> None:
    """Check the GRU, both memories, relative biases, ConvMixer norm, and decoder gradients."""
    torch.manual_seed(104)
    adapter = build_sequence_model(_settings("schematic_gru_full")).eval()
    output, _, state = adapter(torch.randn(2, 5, 112))
    output.square().mean().backward()
    parameters = dict(adapter.named_parameters())
    for name in (
        "galam.gru.weight_ih_l0",
        "galam.global_projection.weight",
        "galam.global_attention.0.relative_bias.hidden.weight",
        "galam.local_attention.0.relative_bias.hidden.weight",
        "galam.fusion.gates.0.weight",
    ):
        gradient = parameters[name].grad
        assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    assert isinstance(adapter.galam.global_mixer_norms[0], nn.LayerNorm)
    assert adapter.galam.local_attention[0].query_norm is not None
    assert adapter.galam.local_attention[0].key_norm is not None
    assert state.lstm_cell is None
    assert not hasattr(adapter.galam, "cell_projection")


def test_local_only_has_no_global_parameters_and_three_branch_fusion() -> None:
    """Check the local-only ablation omits global modules and trains all local branches."""
    torch.manual_seed(105)
    adapter = build_sequence_model(_settings("schematic_lstm_local_only")).eval()
    output, _, state = adapter(torch.randn(2, 5, 112))
    output.square().mean().backward()
    parameters = dict(adapter.named_parameters())
    assert not any("global_" in name for name in parameters)
    assert not hasattr(adapter.galam, "global_projection")
    for name in (
        "galam.lstm.weight_ih_l0",
        "galam.local_attention.0.relative_bias.hidden.weight",
        "galam.hidden_projection.0.weight",
        "galam.cell_projection.0.weight",
        "galam.fusion.gates.0.weight",
    ):
        gradient = parameters[name].grad
        assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    assert state.global_attention == ()
    assert state.global_mixers == ()
    assert state.lstm_cell is not None


def test_variant_selector_is_optional_nested_and_strict() -> None:
    """Keep old configs at the historical LSTM default and reject false or unknown selectors."""
    legacy = _settings()
    resolved = galam_model_settings(legacy)
    assert resolved["galam"] == legacy["galam"]
    model = build_sequence_model(legacy)
    assert isinstance(model, GALAMSequenceAdapter)
    assert isinstance(model.galam, GALAM)
    assert model.variant == "schematic_lstm_full"

    with pytest.raises(ValueError, match="variant"):
        galam_model_settings(_settings("geglu_exact"))
    with pytest.raises(ValueError, match="unknown GALAM runtime model field"):
        galam_model_settings(legacy | {"variant": "schematic_gru_full"})


def test_ablation_changes_decoder_and_parameter_count() -> None:
    """Record that the three-branch adaptations change decoder and total parameter counts."""
    baseline = build_sequence_model(_settings())
    gru = build_sequence_model(_settings("schematic_gru_full"))
    local_only = build_sequence_model(_settings("schematic_lstm_local_only"))
    assert isinstance(baseline.galam.fusion.gates[0], nn.Linear)
    assert baseline.galam.fusion.gates[0].in_features == 4 * 8
    assert gru.galam.fusion.gates[0].in_features == 3 * 8
    assert local_only.galam.fusion.gates[0].in_features == 3 * 8
    parameter_counts = {
        sum(parameter.numel() for parameter in model.parameters())
        for model in (baseline, gru, local_only)
    }
    assert len(parameter_counts) == 3
