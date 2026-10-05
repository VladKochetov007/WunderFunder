"""Synthetic CPU controls for E487's optimizer-precision comparison."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import torch
from precision_replay import (
    PRECISIONS,
    compare_updates,
    orthogonalize_fp32,
    replay,
    replay_crossed_gradients,
    shared_gradient_control,
)
from precision_replay import (
    e485 as imported_e485,
)

from wnn.nn import RNNModel
from wnn.optim import orthogonalize


class PrecisionReplayTests(unittest.TestCase):
    """Exercise fixed-gradient arithmetic and fail-closed comparisons."""

    def setUp(self) -> None:
        torch.manual_seed(271836)
        model = RNNModel(42, hidden=imported_e485.HIDDEN, layers=2, dropout=0.0, n_out=2)
        self.initial = {name: value.detach().clone() for name, value in model.state_dict().items()}

    def test_fp32_orthogonalizer_matches_source_cpu_byte_exact(self) -> None:
        """The FP32 diagnostic differs only where CUDA source uses BF16."""
        generator = torch.Generator().manual_seed(811)
        for shape in ((3, 64, 64), (3, 128, 64), (3, 64, 128)):
            matrix = torch.randn(shape, generator=generator, dtype=torch.float32)
            self.assertTrue(torch.equal(orthogonalize_fp32(matrix), orthogonalize(matrix)))

    def test_cpu_replay_paths_match_frozen_optimizer_bytes(self) -> None:
        """CPU BF16-source dispatch is FP32 and must equal diagnostic FP32."""
        model = imported_e485.initialize_arm(0.0, self.initial)
        gradients = {}
        for index, (name, parameter) in enumerate(model.named_parameters()):
            sequence = torch.arange(parameter.numel(), dtype=parameter.dtype).reshape_as(parameter)
            gradients[name] = (sequence.remainder(31) - 15.0) * 0.01 + (index + 1) * 0.001
        reference = replay(
            self.initial, gradients, "bf16_source", torch.device("cpu"), "test/source"
        )
        candidate = replay(
            self.initial, gradients, "fp32_diagnostic", torch.device("cpu"), "test/fp32"
        )
        self.assertEqual(reference["updated_state"].keys(), candidate["updated_state"].keys())
        for name in reference["updated_state"]:
            self.assertTrue(
                torch.equal(reference["updated_state"][name], candidate["updated_state"][name]),
                name,
            )
        self.assertTrue(reference["clip_applied"])
        self.assertTrue(candidate["clip_applied"])

    def test_same_gradient_native_custom_and_frozen_helper_exact(self) -> None:
        """Identical gradients cannot produce a difference between model wrappers."""
        control = shared_gradient_control(self.initial, torch.device("cpu"))
        self.assertEqual(set(control), set(PRECISIONS))
        self.assertTrue(control["bf16_source"]["same_gradient_delta_bytes_exact"])
        self.assertTrue(control["bf16_source"]["same_gradient_model_state_bytes_exact"])
        self.assertTrue(control["bf16_source"]["frozen_e474_equal"])
        self.assertIsNone(control["fp32_diagnostic"]["frozen_e474_equal"])

    def test_strict_gate_failure_is_reported_not_raised(self) -> None:
        """A finite expected native/custom gate miss remains a complete diagnostic."""
        model = imported_e485.initialize_arm(0.0, self.initial)
        names = [name for name, _parameter in model.named_parameters()]
        reference = {
            name: torch.ones_like(parameter) for name, parameter in model.named_parameters()
        }
        candidate = {name: value.clone() for name, value in reference.items()}
        candidate["gru.weight_hh_l0"][0, 0] += 3e-5
        gate = compare_updates(reference, candidate)
        self.assertFalse(gate["passed"])
        self.assertEqual(len(gate["metrics"]), 7)
        self.assertFalse(gate["metrics"]["matrix:gru.weight_hh_l0"]["passed"])
        self.assertEqual(set(names), set(reference))

    def test_expected_crossed_gate_failure_keeps_all_four_replays(self) -> None:
        """A finite BF16 gate miss cannot prevent the FP32 arm from being recorded."""
        model = imported_e485.initialize_arm(0.0, self.initial)
        gradients = {
            name: torch.full_like(parameter, 0.01) for name, parameter in model.named_parameters()
        }
        captured = {
            "native": {"raw_gradients": gradients},
            "custom": {"raw_gradients": gradients},
        }
        with patch(
            "precision_replay.compare_updates",
            return_value={"metrics": {"registered": {"passed": False}}, "passed": False},
        ):
            result = replay_crossed_gradients(self.initial, captured, torch.device("cpu"))
        self.assertEqual(len(result["crossed_replays"]), 4)
        self.assertEqual(set(result["native_custom_strict_gates"]), set(PRECISIONS))
        self.assertTrue(result["expected_gate_failures_are_recorded_not_raised"])
        self.assertFalse(result["native_custom_strict_gates"]["bf16_source"]["passed"])


if __name__ == "__main__":
    unittest.main()
