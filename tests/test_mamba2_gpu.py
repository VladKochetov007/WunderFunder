from __future__ import annotations

import pytest
import torch

from wnn.mamba2_regressor import Mamba2Regressor, Mamba2State

pytestmark = pytest.mark.gpu


def _model() -> Mamba2Regressor:
    """Build a small Mamba-2 model with dimensions supported by the SSD kernels."""
    return Mamba2Regressor(
        n_in=9,
        hidden=64,
        layers=2,
        state_size=16,
        head_dim=16,
        expand=2,
        conv_width=4,
        chunk_size=16,
        n_out=2,
    ).cuda().eval()


def _assert_state_close(left: Mamba2State, right: Mamba2State) -> None:
    """Compare every causal convolution and SSD state tensor."""
    assert len(left.layers) == len(right.layers)
    for left_layer, right_layer in zip(left.layers, right.layers, strict=True):
        torch.testing.assert_close(left_layer.conv, right_layer.conv, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(left_layer.ssm, right_layer.ssm, atol=5e-4, rtol=5e-4)


def test_full_sequence_matches_steps_and_ragged_chunks() -> None:
    """Require full, tokenwise, and uneven-chunk CUDA outputs to agree."""
    if not torch.cuda.is_available() or torch.version.cuda is None:
        pytest.skip("Mamba-2 tests require an NVIDIA CUDA host")

    torch.manual_seed(401)
    model = _model()
    values = torch.randn(2, 65, 9, device="cuda")
    full, full_state = model(values)

    step_outputs = []
    step_state = None
    for index in range(values.shape[1]):
        output, step_state = model(values[:, index : index + 1], step_state)
        step_outputs.append(output)
    stepped = torch.cat(step_outputs, dim=1)

    chunk_outputs = []
    chunk_state = None
    offset = 0
    for size in (1, 3, 7, 2, 11, 5, 36):
        output, chunk_state = model(values[:, offset : offset + size], chunk_state)
        chunk_outputs.append(output)
        offset += size
    assert offset == values.shape[1]
    chunked = torch.cat(chunk_outputs, dim=1)

    torch.testing.assert_close(stepped, full, atol=5e-4, rtol=5e-4)
    torch.testing.assert_close(chunked, full, atol=5e-4, rtol=5e-4)
    _assert_state_close(step_state, full_state)
    _assert_state_close(chunk_state, full_state)


def test_batch_members_are_independent_and_none_starts_fresh_state() -> None:
    """Keep both batch members and repeated sequences isolated by explicit state."""
    if not torch.cuda.is_available() or torch.version.cuda is None:
        pytest.skip("Mamba-2 tests require an NVIDIA CUDA host")

    torch.manual_seed(407)
    model = _model()
    values = torch.randn(2, 31, 9, device="cuda")
    batched, batched_state = model(values)
    separate = [model(values[index : index + 1]) for index in range(2)]
    separate_predictions = [result[0] for result in separate]
    torch.testing.assert_close(
        torch.cat(separate_predictions), batched, atol=5e-4, rtol=5e-4
    )
    for layer_index, batched_layer in enumerate(batched_state.layers):
        for batch_index, (_, single_state) in enumerate(separate):
            single_layer = single_state.layers[layer_index]
            torch.testing.assert_close(
                batched_layer.conv[batch_index : batch_index + 1],
                single_layer.conv,
                atol=1e-6,
                rtol=1e-6,
            )
            torch.testing.assert_close(
                batched_layer.ssm[batch_index : batch_index + 1],
                single_layer.ssm,
                atol=5e-4,
                rtol=5e-4,
            )

    # Calling with None after another sequence must discard all prior history.
    _, first_sequence_state = model(values[:1, :13])
    fresh_expected, expected_state = model(values[1:2, 13:])
    fresh_after_prior, actual_state = model(values[1:2, 13:], state=None)
    contaminated, _ = model(values[1:2, 13:], state=first_sequence_state)
    torch.testing.assert_close(fresh_after_prior, fresh_expected, atol=1e-6, rtol=1e-6)
    _assert_state_close(actual_state, expected_state)
    assert not torch.allclose(contaminated, fresh_expected, atol=1e-6, rtol=1e-6)


def test_streaming_scan_matches_official_mamba2_full_and_step() -> None:
    """Compare the wrapper's SSD path to pinned upstream full and token APIs."""
    if not torch.cuda.is_available() or torch.version.cuda is None:
        pytest.skip("Mamba-2 tests require an NVIDIA CUDA host")

    torch.manual_seed(409)
    model = _model()
    block = model.layers[0]
    mixer = block.mixer
    values = torch.randn(2, 19, model.hidden, device="cuda")
    normalized = block.norm(values)

    streamed, _ = block(values, None)
    wrapper_mixer = streamed - values
    official_full = mixer(normalized)
    torch.testing.assert_close(wrapper_mixer, official_full, atol=5e-4, rtol=5e-4)

    conv_state, ssm_state = mixer.allocate_inference_cache(2, values.shape[1])
    step_outputs = []
    with torch.no_grad():
        for index in range(values.shape[1]):
            output, conv_state, ssm_state = mixer.step(
                normalized[:, index : index + 1], conv_state, ssm_state
            )
            step_outputs.append(output)
    official_steps = torch.cat(step_outputs, dim=1)
    torch.testing.assert_close(official_steps, official_full, atol=5e-4, rtol=5e-4)


def test_streaming_path_has_finite_nonzero_ssd_gradients_and_detachable_state() -> None:
    """Exercise differentiability through SSD, convolution, projection, and head."""
    if not torch.cuda.is_available() or torch.version.cuda is None:
        pytest.skip("Mamba-2 tests require an NVIDIA CUDA host")

    torch.manual_seed(419)
    model = _model().train()
    values = torch.randn(2, 23, 9, device="cuda", requires_grad=True)
    prefix, carry = model(values[:, :7])
    carry.layers[0].conv.retain_grad()
    carry.layers[0].ssm.retain_grad()
    suffix, state = model(values[:, 7:], carry)
    predictions = torch.cat((prefix, suffix), dim=1)
    predictions.square().mean().backward()

    parameters = (
        model.project.weight,
        model.layers[0].mixer.in_proj.weight,
        model.layers[0].mixer.conv1d.weight,
        model.layers[0].mixer.A_log,
        model.layers[0].mixer.D,
        model.head[2].weight,
    )
    for parameter in parameters:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    assert values.grad is not None
    assert torch.isfinite(values.grad).all()
    for carried_tensor in (carry.layers[0].conv, carry.layers[0].ssm):
        assert carried_tensor.grad is not None
        assert torch.isfinite(carried_tensor.grad).all()
        assert carried_tensor.grad.abs().sum() > 0

    assert carry.layers[0].conv.requires_grad
    assert carry.layers[0].ssm.requires_grad
    detached = state.detach()
    for layer in detached.layers:
        assert not layer.conv.requires_grad
        assert not layer.ssm.requires_grad
