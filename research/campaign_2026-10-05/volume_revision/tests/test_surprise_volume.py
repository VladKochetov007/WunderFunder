"""Tiny CPU checks for causal surprise-volume features and matched model arms."""

from __future__ import annotations

import unittest

import torch
from wnn.surprise_volume import SurpriseVolumeGRU, surprise_volume_features

from wnn.event_content import ENDPOINT
from wnn.nn import RNNModel


def make_raw(batch: int = 2, steps: int = 31, n_in: int = 211) -> torch.Tensor:
    """Create deterministic volume paths with endpoint markers."""
    raw = torch.zeros(batch, steps, n_in, dtype=torch.float32)
    raw[..., 110] = 0.0
    raw[:, [4, 15, 26], 110] = float(ENDPOINT)
    path = torch.arange(steps, dtype=torch.float32)[None, :, None]
    raw[..., 22:44] = (path * torch.linspace(0.01, 0.22, 22)[None, None, :]).expand(batch, -1, -1)
    return raw


class SurpriseVolumeTests(unittest.TestCase):
    def test_features_are_causal_and_endpoint_reset(self) -> None:
        raw = make_raw(batch=1)
        features = surprise_volume_features(raw[..., :112])
        self.assertEqual(tuple(features.shape), (1, 31, 22))
        torch.testing.assert_close(features[:, 4], torch.zeros_like(features[:, 4]))
        torch.testing.assert_close(features[:, 15], torch.zeros_like(features[:, 15]))

        changed = raw.clone()
        changed[:, 20:, 22:44] += 100.0
        updated = surprise_volume_features(changed[..., :112])
        torch.testing.assert_close(features[:, :20], updated[:, :20], rtol=0, atol=0)
        self.assertFalse(torch.allclose(features[:, 20:], updated[:, 20:]))

    def test_zero_context_overlay_matches_parent_and_shapes(self) -> None:
        torch.manual_seed(27)
        parent = RNNModel(211, hidden=16, layers=2, n_out=23).eval()
        model = SurpriseVolumeGRU(211, hidden=16, layers=2, n_out=23, arm="noop").eval()
        model.load_parent_state_dict(parent.state_dict())
        inputs = make_raw()
        parent_output, parent_state = parent(inputs)
        output, state = model(inputs)
        self.assertEqual(tuple(output.shape), (2, 31, 23))
        torch.testing.assert_close(output, parent_output, rtol=2e-6, atol=2e-6)
        torch.testing.assert_close(state, parent_state, rtol=2e-6, atol=2e-6)

        aligned = SurpriseVolumeGRU(
            211, hidden=16, layers=2, n_out=23, arm="aligned"
        ).eval()
        aligned.load_parent_state_dict(parent.state_dict())
        aligned_output, aligned_state = aligned(inputs)
        torch.testing.assert_close(aligned_output, parent_output, rtol=2e-6, atol=2e-6)
        torch.testing.assert_close(aligned_state, parent_state, rtol=2e-6, atol=2e-6)

    def test_stale_arm_uses_previous_run_terminal_row(self) -> None:
        raw = make_raw(batch=1)
        context = surprise_volume_features(raw[..., :112])
        stale = SurpriseVolumeGRU.stale_features(context, raw[..., :112])
        torch.testing.assert_close(stale[:, :15], torch.zeros_like(stale[:, :15]))
        torch.testing.assert_close(stale[:, 15:26], context[:, 14:15].expand(-1, 11, -1))
        torch.testing.assert_close(stale[:, 26:], context[:, 25:26].expand(-1, 5, -1))

    def test_arm_topology_and_update_gate_only_gradient(self) -> None:
        torch.manual_seed(28)
        parent = RNNModel(211, hidden=16, layers=2, n_out=23)
        arms = [
            SurpriseVolumeGRU(211, hidden=16, layers=2, n_out=23, arm=arm)
            for arm in SurpriseVolumeGRU.ARMS
        ]
        parameter_layout = [
            (name, tuple(value.shape), value.numel()) for name, value in arms[0].named_parameters()
        ]
        for model in arms:
            model.load_parent_state_dict(parent.state_dict())
            self.assertEqual(
                [
                    (name, tuple(value.shape), value.numel())
                    for name, value in model.named_parameters()
                ],
                parameter_layout,
            )
            self.assertTrue(model.constraints_hold())

        model = arms[0].train()
        with torch.no_grad():
            model.surprise_projection.weight.fill_(0.05)
            model.surprise_projection.bias.fill_(0.1)
            model.gru.weight_ih_l0[16:32, 16:].fill_(0.02)
        output, _ = model(make_raw())
        output.square().mean().backward()
        extra_gradient = model.gru.weight_ih_l0.grad[:, 16:]
        self.assertGreater(torch.count_nonzero(extra_gradient[16:32]).item(), 0)
        self.assertEqual(torch.count_nonzero(extra_gradient[:16]).item(), 0)
        self.assertEqual(torch.count_nonzero(extra_gradient[32:]).item(), 0)

    def test_uneven_chunks_use_full_sequence_features(self) -> None:
        torch.manual_seed(29)
        model = SurpriseVolumeGRU(211, hidden=16, layers=2, n_out=23, arm="aligned").eval()
        with torch.no_grad():
            model.surprise_projection.weight.fill_(0.03)
            model.surprise_projection.bias.fill_(0.05)
            model.gru.weight_ih_l0[16:32, 16:].fill_(0.01)
        inputs = make_raw()
        context = surprise_volume_features(inputs[..., :112])
        full, final = model(inputs, full_context=context)
        state = None
        pieces = []
        for start, stop in ((0, 6), (6, 13), (13, 14), (14, 31)):
            piece, state = model.forward_chunk(
                inputs, slice(start, stop), state, full_context=context
            )
            pieces.append(piece)
        torch.testing.assert_close(torch.cat(pieces, dim=1), full, rtol=2e-6, atol=2e-6)
        torch.testing.assert_close(state, final, rtol=2e-6, atol=2e-6)


if __name__ == "__main__":
    unittest.main()
