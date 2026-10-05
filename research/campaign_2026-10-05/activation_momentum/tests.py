"""Synthetic source checks for the isolated E483 Muon momentum actor."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research/campaign_2026-10-05/activation_optimizer"))
sys.path.insert(0, str(ROOT / "research/campaign_2026-10-05/activation_placement"))
sys.path.insert(0, str(ROOT / "research/campaign_2026-10-05/recurrent_activation_curvature"))

import fit_momentum as actor  # noqa: E402
import fit_recurrent_curvature as e481  # noqa: E402

from wnn.optim import MuonAdamW, orthogonalize  # noqa: E402


def optimizer_with_beta(beta: float, muon: nn.Parameter, adam: nn.Parameter) -> MuonAdamW:
    """Create one isolated two-family optimizer for the synthetic reference."""
    return MuonAdamW(
        [
            {"params": [adam], "betas": actor.BETAS},
            {"params": [muon], "muon": True, "blocks": 2, "betas": (beta, 0.999)},
        ],
        lr=0.01,
        weight_decay=0.0,
        betas=actor.BETAS,
    )


def frozen_muon_reference(
    initial: torch.Tensor, first: torch.Tensor, second: torch.Tensor, beta: float
) -> torch.Tensor:
    """Independently apply the frozen two-step Muon recurrence to a test matrix."""
    parameter = initial.clone()
    momentum = torch.zeros_like(parameter)
    for gradient in (first, second):
        momentum = momentum * beta + gradient * (1.0 - beta)
        effective = gradient * (1.0 - beta) + momentum * beta
        blocks, rows_per_block, columns = 2, parameter.shape[0] // 2, parameter.shape[1]
        orthogonal = orthogonalize(
            effective.reshape(blocks, rows_per_block, columns), steps=5
        ).reshape_as(parameter)
        parameter = parameter - 0.01 * 0.2 * max(rows_per_block, columns) ** 0.5 * orthogonal
    return parameter


class MomentumActorTests(unittest.TestCase):
    """Exercise optimizer isolation and frozen equations without data or a GPU."""

    def test_registered_matrix_and_seed_initializations(self) -> None:
        self.assertEqual(len(actor.ARMS), 18)
        self.assertEqual(len(set(actor.SEEDS)), 3)
        states = actor.make_seed_states()
        self.assertEqual(len({actor.state_hash(state) for state in states.values()}), 3)
        for _seed, state in states.items():
            baseline = actor.initialize_arm("baseline", state)
            sine = actor.initialize_arm("sine", state)
            for name, value in baseline.state_dict().items():
                torch.testing.assert_close(value, sine.state_dict()[name], rtol=0, atol=0)
            for beta in actor.MOMENTA:
                optimizer = actor.make_optimizer(baseline, beta)
                receipt = actor.optimizer_group_receipt(baseline, optimizer, beta)
                recurrent = [group for group in receipt if group["muon"]]
                adam = [group for group in receipt if not group["muon"]]
                self.assertEqual(len(recurrent), 1)
                self.assertEqual(set(recurrent[0]["names"]), actor.MUON_NAMES)
                self.assertEqual(recurrent[0]["betas"], [beta, 0.999])
                self.assertTrue(adam)
                self.assertTrue(all(group["betas"] == list(actor.BETAS) for group in adam))
                self.assertEqual(sum(len(group["names"]) for group in receipt), len(state) - 2)

    def test_noncollinear_two_step_muon_reference_and_adam_invariance(self) -> None:
        initial = torch.tensor(
            [[0.4, -0.1], [0.2, 0.7], [-0.3, 0.5], [0.9, -0.6]], dtype=torch.float32
        )
        first = torch.tensor(
            [[1.0, 0.2], [-0.1, 0.8], [0.4, -0.3], [0.2, 0.9]], dtype=torch.float32
        )
        second = torch.tensor(
            [[0.1, 1.0], [0.7, -0.2], [-0.8, 0.1], [0.3, -0.4]], dtype=torch.float32
        )
        adam_initial = torch.tensor([0.3, -0.8], dtype=torch.float32)
        adam_grads = (torch.tensor([1.0, -2.0]), torch.tensor([-0.5, 0.25]))
        results = {}
        adam_results = {}
        for beta in actor.MOMENTA:
            muon_parameter = nn.Parameter(initial.clone())
            adam_parameter = nn.Parameter(adam_initial.clone())
            optimizer = optimizer_with_beta(beta, muon_parameter, adam_parameter)
            for muon_grad, adam_grad in zip((first, second), adam_grads, strict=True):
                muon_parameter.grad = muon_grad.clone()
                adam_parameter.grad = adam_grad.clone()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            expected = frozen_muon_reference(initial, first, second, beta)
            torch.testing.assert_close(muon_parameter, expected, rtol=1e-6, atol=1e-7)
            results[beta] = muon_parameter.detach().clone()
            adam_results[beta] = adam_parameter.detach().clone()
        self.assertFalse(torch.equal(results[0.8], results[0.9]))
        self.assertFalse(torch.equal(results[0.95], results[0.9]))
        for beta in (0.8, 0.95):
            torch.testing.assert_close(adam_results[beta], adam_results[0.9], rtol=0, atol=0)

    def test_local_clipping_matches_frozen_e474_helper(self) -> None:
        states = actor.make_seed_states()
        initial = states[actor.SEEDS[0]]
        left = actor.initialize_arm("baseline", initial)
        right = actor.initialize_arm("baseline", initial)
        left_optimizer = actor.make_optimizer(left, 0.9)
        right_optimizer = actor.make_optimizer(right, 0.9)
        torch.manual_seed(951)
        x = torch.randn(2, 13, actor.FEATURES)
        target = torch.randn(2, 13, 2)
        for optimizer in (left_optimizer, right_optimizer):
            optimizer.zero_grad(set_to_none=True)
        prediction, state = left(x)
        (torch.nn.functional.mse_loss(prediction, target) + state.square().mean()).backward()
        for left_parameter, right_parameter in zip(
            left.parameters(), right.parameters(), strict=True
        ):
            right_parameter.grad = left_parameter.grad.detach().clone()
        local = actor.apply_optimizer_update(left, left_optimizer, {}, 1, "test/local")
        e481.e474.apply_optimizer_update(right, right_optimizer, {}, 1, "test/frozen")
        self.assertTrue(torch.isfinite(torch.tensor(local["preclip_gradient_norm"])))
        for actual, expected in zip(left.parameters(), right.parameters(), strict=True):
            self.assertTrue(torch.equal(actual, expected))

    def test_native_reset_after_output_state_gradient_parity_on_cpu(self) -> None:
        initial = actor.make_seed_states()[actor.SEEDS[0]]
        receipt = actor.run_native_and_optimizer_parity(torch.device("cpu"), initial)
        self.assertEqual(receipt["native_gru"], "output_state_gradient_close")
        self.assertEqual(
            receipt["optimizer_step"],
            "exact_beta_0.9_clipped_muonadamw_parameter_match",
        )


if __name__ == "__main__":
    unittest.main()
