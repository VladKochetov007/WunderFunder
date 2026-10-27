from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, replace
from typing import Literal

import torch

from wnn.galam import (
    GALAM,
    connectome_112_to_2_config,
    galam_config_from_mapping,
    historical_32_config,
)


@contextmanager
def _single_threaded() -> Iterator[None]:
    """Bound synthetic CPU checks to one Torch thread and restore the caller setting."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _small_config(timestamp_source: Literal["ordinal", "provided"] = "ordinal"):
    """Return a small explicitly assumed config for causal component checks."""
    return replace(
        historical_32_config(),
        width=8,
        attention_heads=2,
        attention_history=4,
        conv_kernel_size=3,
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
            cache is not None and cache.keys.shape[2] <= config.attention_history
            for cache in state.local_attention
        )
        assert all(
            cache is not None and cache.keys.shape[2] <= config.attention_history
            for cache in state.global_attention
        )
        assert all(mixer is not None for mixer in state.global_mixers)
        assert state.global_mixers[0].first_inputs.shape[-1] == config.conv_kernel_size - 1
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
