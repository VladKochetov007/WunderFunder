from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, replace
from typing import Literal

import pytest
import torch
from torch.nn import functional as F

from wnn.galam import (
    GALAM,
    GALAMConfig,
    connectome_112_to_2_config,
    galam_config_from_mapping,
    historical_32_config,
)
from wnn.galam.attention import TemporalSelfAttention
from wnn.galam.fusion import DiagramChannelProductGate
from wnn.galam.mixer import CausalConvMixer, PerTimeBatchNorm1d


@contextmanager
def _single_threaded() -> Iterator[None]:
    """Bound synthetic CPU checks to one Torch thread and restore the caller setting."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _small_config(timestamp_source: Literal["ordinal", "provided"] = "ordinal") -> GALAMConfig:
    """Return a small explicitly assumed config for causal component checks."""
    return replace(
        historical_32_config(),
        width=8,
        attention_heads=2,
        attention_history=4,
        conv_kernel_size=3,
        conv_hidden_width=16,
        relative_bias_width=4,
        feed_forward_width=12,
        timestamp_source=timestamp_source,
    )


def test_configs_keep_historical_and_connectome_shapes_distinct() -> None:
    historical = historical_32_config()
    adaptation = connectome_112_to_2_config()
    assert (historical.input_features, historical.output_features) == (32, 32)
    assert (adaptation.input_features, adaptation.output_features) == (112, 2)
    assert historical.assumed_details
    assert adaptation.assumed_details
    assert "historical_32_input_and_output_widths" in historical.faithful_details
    assert "historical_32_input_and_output_widths" not in adaptation.faithful_details
    assert "connectome_output_adaptation" in adaptation.assumed_details
    assert "connectome_output_adaptation" not in historical.assumed_details
    assert galam_config_from_mapping(asdict(historical)) == historical


def test_config_rejects_invalid_dimensions_scales_roles_and_fidelity_claims() -> None:
    config = historical_32_config()
    with pytest.raises(ValueError, match="dimensions"):
        replace(config, width=True)
    for scale in (math.nan, math.inf, -math.inf, 0.0):
        with pytest.raises(ValueError, match="finite and positive"):
            replace(config, timestamp_scale=scale)
    with pytest.raises(ValueError, match="requires 32 inputs"):
        replace(config, input_features=112)
    with pytest.raises(ValueError, match="role"):
        replace(config, role="unknown")
    with pytest.raises(ValueError, match="timestamp_source"):
        replace(config, timestamp_source="relative")
    with pytest.raises(ValueError, match="fusion_mode"):
        replace(config, fusion_mode="two_stream_weighted_sum")
    with pytest.raises(ValueError, match="fusion_channel_order"):
        replace(config, fusion_channel_order=("global_memory",) * 4)
    with pytest.raises(ValueError, match="immutable frozensets"):
        replace(config, faithful_details=set(config.faithful_details))
    false_assumptions = config.assumed_details - {"width_and_attention_heads"}
    false_fidelity = config.faithful_details | {"width_and_attention_heads"}
    with pytest.raises(ValueError, match="unresolved details"):
        replace(config, faithful_details=false_fidelity, assumed_details=false_assumptions)


def test_source_schematic_global_norms_and_conv_channel_path() -> None:
    config = _small_config()
    model = GALAM(config)
    assert all(isinstance(norm, torch.nn.LayerNorm) for norm in model.global_attention_norms)
    assert all(isinstance(norm, torch.nn.LayerNorm) for norm in model.global_mixer_norms)
    mixer = model.global_mixers[0]
    assert mixer.conv1.conv.in_channels == config.width
    assert mixer.conv1.conv.out_channels == config.conv_hidden_width
    assert mixer.norm1.norm.num_features == config.conv_hidden_width
    assert mixer.conv2.conv.in_channels == config.conv_hidden_width
    assert mixer.conv2.conv.out_channels == config.width
    assert mixer.norm2.norm.num_features == config.width
    assert config.fusion_mode == "diagram_channel_product"
    assert "fusion_channel_order" in config.assumed_details
    assert model.fusion.gates[0].weight.shape == (4 * config.width, 4 * config.width)
    assert model.fusion_norm.normalized_shape == (4 * config.width,)
    assert model.feed_forward[0].in_features == 4 * config.width
    assert model.feed_forward[-1].out_features == 4 * config.width
    assert model.readout.in_features == 4 * config.width


def test_decoder_gate_multiplies_the_same_ordered_channel_bypass() -> None:
    width = 2
    order = ("global_memory", "local_memory", "hidden_projection", "cell_projection")
    gate = DiagramChannelProductGate(width, "diagram_channel_product", order).eval()
    with torch.no_grad():
        for parameter in gate.parameters():
            parameter.zero_()
        logits = torch.linspace(-1.5, 1.5, 4 * width)
        gate.gates[-1].bias.copy_(logits)
    sources = {
        "global_memory": torch.tensor([[1.0, 2.0]]),
        "local_memory": torch.tensor([[3.0, 4.0]]),
        "hidden_projection": torch.tensor([[5.0, 6.0]]),
        "cell_projection": torch.tensor([[7.0, 8.0]]),
    }
    bypass = torch.cat(tuple(sources[name] for name in order), dim=-1)
    actual = gate(**sources)
    expected = bypass * torch.sigmoid(logits).unsqueeze(0)
    assert actual.shape == (1, 4 * width)
    torch.testing.assert_close(actual, expected)


def test_conv_mixer_uses_schematic_conv_gelu_bn_order() -> None:
    with _single_threaded():
        mixer = CausalConvMixer(width=1, hidden_width=1, kernel_size=1).train()
        with torch.no_grad():
            mixer.conv1.conv.weight.fill_(1.0)
            mixer.conv1.conv.bias.zero_()
            mixer.conv2.conv.weight.fill_(1.0)
            mixer.conv2.conv.bias.zero_()
        values = torch.tensor([[-2.0], [0.25], [2.0]])
        actual, _ = mixer.forward_step(values, None)

        first = F.gelu(values)
        first = (first - first.mean(dim=0)) / torch.sqrt(first.var(dim=0, unbiased=False) + 1e-5)
        expected = F.gelu(first)
        expected = (expected - expected.mean(dim=0)) / torch.sqrt(
            expected.var(dim=0, unbiased=False) + 1e-5
        )
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_attention_does_not_attend_beyond_declared_total_context() -> None:
    with _single_threaded(), torch.no_grad():
        torch.manual_seed(37)
        attention = TemporalSelfAttention(8, 2, history=3, bias_width=4, timestamp_scale=1.0).eval()
        shared = torch.randn(3, 1, 8)
        sequences = (
            torch.cat((torch.randn(6, 1, 8), shared)),
            torch.cat((torch.randn(2, 1, 8), shared)),
        )
        outputs = []
        for sequence in sequences:
            cache = None
            for index, value in enumerate(sequence):
                time = torch.tensor([float(index)])
                output, cache = attention.forward_step(value, time, cache)
            outputs.append(output)
        torch.testing.assert_close(outputs[0], outputs[1], rtol=1e-5, atol=1e-6)


def test_training_batch_norm_has_clear_minimum_batch_contract() -> None:
    norm = PerTimeBatchNorm1d(3).train()
    with pytest.raises(ValueError, match="batch size >= 2"):
        norm(torch.randn(1, 3))


def test_empty_chunk_is_a_noop_and_state_detach_cuts_history_graph() -> None:
    with _single_threaded():
        config = _small_config()
        model = GALAM(config).eval()
        values = torch.randn(2, config.input_features, requires_grad=True)
        state = model.reset_state(2, values.device, values.dtype)
        empty, unchanged = model.forward_chunk(
            values.new_empty((2, 0, config.input_features)), state
        )
        assert empty.shape == (2, 0, config.output_features)
        assert unchanged is state

        _, state = model.stream_step(values, state)
        assert state.lstm_hidden.grad_fn is not None
        detached = state.detach()
        assert detached.steps_seen == state.steps_seen
        assert detached.lstm_hidden.grad_fn is None
        assert detached.lstm_cell.grad_fn is None
        for cache in detached.local_attention + detached.global_attention:
            assert cache is not None
            assert cache.keys.grad_fn is None
            assert cache.values.grad_fn is None
        for mixer in detached.global_mixers:
            assert mixer is not None
            assert mixer.first_inputs.grad_fn is None
            assert mixer.second_inputs.grad_fn is None


def test_full_prefix_chunk_and_step_outputs_match_with_bounded_state() -> None:
    with _single_threaded(), torch.no_grad():
        torch.manual_seed(47)
        config = _small_config()
        model = GALAM(config).eval()
        values = torch.randn(3, 11, config.input_features)
        full = model(values)
        prefix = model(values[:, :5])
        torch.testing.assert_close(full[:, :5], prefix, rtol=1e-5, atol=1e-6)

        state = model.reset_state(3, values.device, values.dtype)
        chunks = []
        for start, end in ((0, 3), (3, 7), (7, 11)):
            prediction, state = model.forward_chunk(values[:, start:end], state)
            chunks.append(prediction)
        torch.testing.assert_close(full, torch.cat(chunks, dim=1), rtol=1e-5, atol=1e-6)

        state = model.reset_state(3, values.device, values.dtype)
        steps = []
        for index in range(values.shape[1]):
            prediction, state = model.stream_step(values[:, index], state)
            steps.append(prediction.unsqueeze(1))
        torch.testing.assert_close(full, torch.cat(steps, dim=1), rtol=1e-5, atol=1e-6)
        assert state.steps_seen == values.shape[1]
        assert all(
            cache is not None and cache.keys.shape[2] == config.attention_history
            for cache in state.local_attention
        )
        assert all(
            cache is not None and cache.keys.shape[2] == config.attention_history
            for cache in state.global_attention
        )
        assert all(mixer is not None for mixer in state.global_mixers)
        assert state.global_mixers[0].first_inputs.shape[1] == config.width
        assert state.global_mixers[0].first_inputs.shape[-1] == config.conv_kernel_size - 1
        assert state.global_mixers[0].second_inputs.shape[1] == config.conv_hidden_width
        reset = model.reset_state(3, values.device, values.dtype)
        assert reset.steps_seen == 0
        assert all(cache is None for cache in reset.local_attention + reset.global_attention)


def test_provided_absolute_timestamps_preserve_chunk_parity() -> None:
    with _single_threaded(), torch.no_grad():
        torch.manual_seed(11)
        config = _small_config("provided")
        model = GALAM(config).eval()
        values = torch.randn(2, 7, config.input_features)
        times = torch.tensor([[0.0, 0.2, 1.0, 4.5, 5.0, 9.0, 10.0]] * 2)
        full = model(values, times)
        state = model.reset_state(2, values.device, values.dtype)
        left, state = model.forward_chunk(values[:, :4], state, times[:, :4])
        right, _ = model.forward_chunk(values[:, 4:], state, times[:, 4:])
        torch.testing.assert_close(full, torch.cat((left, right), dim=1), rtol=1e-5, atol=1e-6)


def test_training_normalization_does_not_use_future_timestamps() -> None:
    with _single_threaded():
        torch.manual_seed(19)
        config = _small_config()
        whole_model = GALAM(config).train()
        prefix_model = GALAM(config).train()
        prefix_model.load_state_dict(whole_model.state_dict())
        values = torch.randn(3, 8, config.input_features)
        whole = whole_model(values)
        prefix = prefix_model(values[:, :4])
        torch.testing.assert_close(whole[:, :4], prefix, rtol=1e-5, atol=1e-6)
