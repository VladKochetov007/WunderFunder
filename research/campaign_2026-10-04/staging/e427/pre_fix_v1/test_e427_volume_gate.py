"""Synthetic CPU checks for E427; run only in the admitted root test worker."""

from __future__ import annotations

import torch

from wnn.e427_volume_gate import E427VolumeGateModel
from wnn.nn import RNNModel
from wnn.optim import MuonAdamW


def make_parent(n_in: int = 211, hidden: int = 24) -> RNNModel:
    return RNNModel(n_in, hidden=hidden, layers=2, dropout=0.0, n_out=23)


def make_overlay(parent: RNNModel, arm: str = "noop") -> E427VolumeGateModel:
    model = E427VolumeGateModel(
        parent.mean.numel(), hidden=parent.gru.hidden_size, layers=2, dropout=0.0,
        n_out=23, arm=arm, fallback_mean=[0.0] * 28,
    )
    model.load_parent_state_dict(parent.state_dict())
    return model


def raw_input(batch: int = 2, rows: int = 47, n_in: int = 211) -> torch.Tensor:
    value = torch.randn(batch, rows, n_in)
    value[..., 110] = 0.0
    return value


def test_noop_matches_parent_full_and_chunked_state_carry() -> None:
    torch.manual_seed(17)
    parent = make_parent().eval()
    model = make_overlay(parent).eval()
    inputs = raw_input()
    expected, expected_state = parent(inputs)
    actual, actual_state = model(inputs)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(actual_state, expected_state, rtol=2e-6, atol=2e-6)

    state = None
    pieces = []
    for start, stop in ((0, 3), (3, 18), (18, 19), (19, 47)):
        output, state = model.forward_chunk(inputs, slice(start, stop), state)
        pieces.append(output)
    torch.testing.assert_close(torch.cat(pieces, dim=1), expected, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(state, expected_state, rtol=2e-6, atol=2e-6)


def test_three_arms_have_identical_tensor_topology_and_budget() -> None:
    torch.manual_seed(16)
    parent = make_parent(hidden=16)
    models = [make_overlay(parent, arm) for arm in ("aligned", "stale", "noop")]
    reference = [(name, tuple(value.shape), value.numel()) for name, value in models[0].named_parameters()]
    for model in models[1:]:
        assert [(name, tuple(value.shape), value.numel()) for name, value in model.named_parameters()] == reference
        assert model.added_parameter_counts == models[0].added_parameter_counts
    production_allocated = 3 * 256 * 8 + 28 * 8 + 8
    production_fixed = 2 * 256 * 8
    assert (production_allocated, production_fixed, production_allocated - production_fixed) == (
        6376, 4096, 2280,
    )


def test_stale_context_fallback_and_endpoint_boundaries() -> None:
    torch.manual_seed(18)
    raw = raw_input(batch=1, rows=25)
    raw[..., 110] = 0.0
    raw[:, 3, 110] = -5.199337482452393
    raw[:, 12, 110] = -5.199337482452393
    raw[:, 21, 110] = -5.199337482452393
    raw[:, :, 22:44] += torch.arange(25)[None, :, None]
    model = E427VolumeGateModel(211, hidden=16, arm="stale", fallback_mean=[-7.0] * 28)
    features = torch.cat((raw, torch.zeros(1, 25, 99)), dim=-1)
    aligned, stale = model._full_contexts(features)
    torch.testing.assert_close(stale[:, :12], torch.full_like(stale[:, :12], -7.0))
    torch.testing.assert_close(stale[:, 12:21], aligned[:, 11:12].expand(-1, 9, -1))
    torch.testing.assert_close(stale[:, 21:], aligned[:, 20:21].expand(-1, 4, -1))


def test_future_suffix_cannot_change_aligned_or_stale_prefix() -> None:
    torch.manual_seed(19)
    first = raw_input(batch=1, rows=31)
    first[:, [2, 13, 24], 110] = -5.199337482452393
    changed = first.clone()
    changed[:, 18:, :112] = torch.randn_like(changed[:, 18:, :112])
    left = E427VolumeGateModel(211, hidden=16, arm="aligned", fallback_mean=[0.0] * 28)
    right = E427VolumeGateModel(211, hidden=16, arm="aligned", fallback_mean=[0.0] * 28)
    _, stale_left = left._full_contexts(first)
    aligned_left, _ = left._full_contexts(first)
    aligned_right, stale_right = right._full_contexts(changed)
    torch.testing.assert_close(aligned_left[:, :18], aligned_right[:, :18], rtol=0, atol=0)
    torch.testing.assert_close(stale_left[:, :18], stale_right[:, :18], rtol=0, atol=0)


def test_context_changes_only_update_gate_input_block() -> None:
    torch.manual_seed(20)
    parent = make_parent(hidden=16).eval()
    model = make_overlay(parent, "aligned").train()
    with torch.no_grad():
        model.gate_context.weight.fill_(0.05)
        model.gate_context.bias.fill_(0.1)
        model.gru.weight_ih_l0[16:32, 16:].fill_(0.02)
    inputs = raw_input(batch=2, rows=20)
    expected, _ = parent(inputs)
    actual, _ = model(inputs)
    assert not torch.allclose(actual, expected)
    assert model.constraints_hold()
    actual.square().mean().backward()
    extension_grad = model.gru.weight_ih_l0.grad[:, 16:]
    assert torch.count_nonzero(extension_grad[16:32]) > 0
    assert torch.count_nonzero(extension_grad[:16]) == 0
    assert torch.count_nonzero(extension_grad[32:]) == 0


def test_post_muon_projection_restores_exact_structural_zeros() -> None:
    torch.manual_seed(21)
    parent = make_parent(hidden=16)
    model = make_overlay(parent, "aligned")
    weight = model.gru.weight_ih_l0
    optimizer = MuonAdamW([{"params": [weight], "muon": True, "blocks": 3}], lr=1e-2)
    optimizer.state[weight]["momentum_buffer"] = torch.randn_like(weight)
    weight.grad = torch.randn_like(weight)
    optimizer.step()
    # This exercises the actual optimizer implementation with nonzero incoming
    # momentum in constrained coordinates, then the production projection hook.
    assert torch.count_nonzero(weight[:16, 16:]) + torch.count_nonzero(weight[32:, 16:]) > 0
    model.project_constraints_()
    assert model.constraints_hold()
    assert model.added_parameter_counts == {
        "allocated_added": 616, "structurally_fixed": 256, "effective_added": 360,
    }
