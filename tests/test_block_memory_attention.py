from __future__ import annotations

import torch

from wnn.block_memory_attention import BlockMemoryAttention


def tiny_model() -> BlockMemoryAttention:
    torch.manual_seed(41)
    torch.set_num_threads(1)
    return BlockMemoryAttention(n_in=5, width=8, layers=2, heads=2, block_rows=4)


def test_uneven_chunks_and_row_stream_match_full_sequence() -> None:
    model = tiny_model().eval()
    data = torch.randn(2, 13, 5)
    expected, _ = model(data)
    for chunks in ([3, 6, 4], [1] * 13):
        state = None
        start = 0
        outputs = []
        for rows in chunks:
            output, state = model(data[:, start : start + rows], state)
            outputs.append(output)
            start += rows
        torch.testing.assert_close(torch.cat(outputs, 1), expected, rtol=1e-5, atol=1e-6)


def test_future_changes_cannot_affect_prefix_and_none_resets_state() -> None:
    model = tiny_model().eval()
    data = torch.randn(2, 13, 5)
    expected, _ = model(data)
    changed = data.clone()
    changed[:, 7:] = torch.randn_like(changed[:, 7:]) * 100
    actual, _ = model(changed)
    torch.testing.assert_close(actual[:, :7], expected[:, :7], rtol=0, atol=0)
    model(torch.randn_like(data))
    reset, _ = model(data, None)
    torch.testing.assert_close(reset, expected, rtol=0, atol=0)


def test_block_memory_learns_through_complete_blocks() -> None:
    model = tiny_model()
    data = torch.randn(2, 13, 5)
    output, states = model(data)
    output[:, -1].square().mean().backward()
    for layer in model.layers:
        gradient = layer.memory_update.weight_ih.grad
        assert gradient is not None and torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0
    for state in states:
        assert state.inputs.shape[1] == 1
        detached = state.detach()
        assert not detached.memory.requires_grad
        assert not detached.inputs.requires_grad
        assert not detached.output_sum.requires_grad


def test_full_and_chunked_gradients_match_without_detachment() -> None:
    whole, chunked = tiny_model(), tiny_model()
    chunked.load_state_dict(whole.state_dict())
    data = torch.randn(2, 11, 5)
    expected, _ = whole(data)
    a, state = chunked(data[:, :3])
    b, _ = chunked(data[:, 3:], state)
    expected.square().mean().backward()
    torch.cat([a, b], 1).square().mean().backward()
    for left, right in zip(whole.parameters(), chunked.parameters(), strict=True):
        torch.testing.assert_close(left.grad, right.grad, rtol=2e-4, atol=2e-6)
