"""Synthetic source checks for E488 duration accounting and optimizer parity."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
E474_DIR = ROOT / "research/campaign_2026-10-05/activation_placement"
sys.path[:0] = [str(HERE), str(E474_DIR), str(ROOT / "src")]

import fit_activation_placement as e474  # noqa: E402
import fit_duration as actor  # noqa: E402


class E488DurationTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(1)

    def test_epoch_shuffle_is_zero_based_and_reproducible(self) -> None:
        for seed in actor.SEEDS:
            expected = []
            for epoch in range(3):
                order = np.arange(24)
                np.random.default_rng(seed + epoch).shuffle(order)
                expected.append(order)
                np.testing.assert_array_equal(actor.epoch_order(seed, epoch), order)
            self.assertTrue(all(len(set(order.tolist())) == 24 for order in expected))
            self.assertFalse(np.array_equal(expected[0], expected[1]))
        with self.assertRaises(ValueError):
            actor.epoch_order(actor.SEEDS[0], 3)

    def test_initialization_shares_common_bytes_and_gate_uses_no_random_draw(self) -> None:
        torch.manual_seed(442)
        initial = e474.PlacementRNN("gelu", "gelu").state_dict()
        state = {key: value.detach().clone() for key, value in initial.items()}
        torch.manual_seed(987)
        _ = actor.make_arm("gelu", state)
        expected_next = torch.rand(5)
        torch.manual_seed(987)
        mixture = actor.make_arm("recurrent_mix", state)
        observed_next = torch.rand(5)
        torch.testing.assert_close(observed_next, expected_next, rtol=0, atol=0)
        actor.e484.check_common_parameter_bytes(mixture, state)
        self.assertEqual(mixture.head[1].gate_coeff.shape, (64,))
        self.assertTrue(torch.equal(mixture.head[1].gate_coeff, torch.zeros(64)))
        self.assertEqual(sum(p.numel() for p in mixture.parameters()), actor.GATED_PARAMETERS)

    def test_three_mixture_initial_functions_match_and_gelu_anchor_differs(self) -> None:
        torch.manual_seed(31)
        state = {
            key: value.detach().clone()
            for key, value in e474.PlacementRNN("gelu", "gelu").state_dict().items()
        }
        x = torch.randn(2, 37, 42)
        outputs = {}
        for arm in actor.ARMS:
            model = actor.make_arm(arm, state).eval()
            with torch.no_grad():
                outputs[arm] = model(x)[0]
        for arm in ("fixed_mix", "static_mix", "recurrent_mix"):
            self.assertTrue(torch.equal(outputs["fixed_mix"], outputs[arm]))
        self.assertFalse(torch.equal(outputs["gelu"], outputs["fixed_mix"]))

    def test_custom_clipped_update_matches_frozen_e474_helper(self) -> None:
        torch.manual_seed(1234)
        state = {
            key: value.detach().clone()
            for key, value in e474.PlacementRNN("gelu", "gelu").state_dict().items()
        }
        candidate = actor.make_arm("recurrent_mix", state).train()
        reference = actor.make_arm("recurrent_mix", state).train()
        candidate_optimizer = actor.e484.make_optimizer(candidate)
        reference_optimizer = actor.e484.make_optimizer(reference)
        x = torch.randn(4, 41, 42)
        y = torch.randn(4, 41, 2)
        need = torch.ones(4, 41, dtype=torch.bool)
        for step in (0, 1):
            for model, optimizer in (
                (candidate, candidate_optimizer),
                (reference, reference_optimizer),
            ):
                optimizer.zero_grad(set_to_none=True)
                prediction, _state = model(x)
                e470_loss = actor.e470.loss_for_batch(prediction, y, need)
                e470_loss.backward()
            left = actor.apply_duration_update(
                candidate,
                candidate_optimizer,
                actor.e484.parameter_families(candidate),
                step,
                "candidate",
            )
            right = e474.apply_optimizer_update(
                reference,
                reference_optimizer,
                actor.e484.parameter_families(reference),
                step,
                "reference",
            )
            if step == 0:
                self.assertEqual(left["preclip_gradient_rms"], right["preclip_gradient_rms"])
                self.assertEqual(left["parameter_update_rms"], right["parameter_update_rms"])
            for a, b in zip(candidate.parameters(), reference.parameters(), strict=True):
                self.assertTrue(torch.equal(a, b))
        self.assertEqual(actor.optimizer_state_steps(candidate_optimizer), [2])

    def test_registered_trajectory_and_snapshot_counts(self) -> None:
        self.assertEqual(len(actor.SEEDS) * len(actor.ARMS), 12)
        self.assertEqual(actor.TOTAL_UPDATES, 1422)
        self.assertEqual(actor.TELEMETRY_STEPS, (0, 46, 236, 473, 947, 1421))
        self.assertEqual(len(actor.CONTRASTS_PER_SEED) * len(actor.SEEDS), 27)
        self.assertEqual(actor.epoch_order(actor.SEEDS[0], 0).shape, (24,))


if __name__ == "__main__":
    unittest.main()
