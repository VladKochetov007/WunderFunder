"""Synthetic invariants for the E484 context-conditioned activation actor."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research/campaign_2026-10-05/activation_optimizer"))
sys.path.insert(0, str(ROOT / "research/campaign_2026-10-05/activation_placement"))
sys.path.insert(0, str(ROOT / "research/campaign_2026-10-05/activation_head_mixture"))

import fit_activation_head_mixture as e477  # noqa: E402
import fit_context_mixture_v2 as actor  # noqa: E402


class ContextMixtureTests(unittest.TestCase):
    """Test formula, initialization, context gradients and causal independence."""

    def test_common_state_identity_uses_exact_canonical_cpu_bytes(self) -> None:
        value = torch.tensor([0.0, 1.25, -2.0], dtype=torch.float32)
        self.assertEqual(actor.tensor_cpu_identity(value), actor.tensor_cpu_identity(value.clone()))
        negative_zero = torch.tensor([-0.0], dtype=torch.float32)
        positive_zero = torch.tensor([0.0], dtype=torch.float32)
        self.assertNotEqual(
            actor.tensor_cpu_identity(negative_zero), actor.tensor_cpu_identity(positive_zero)
        )
        states = actor.make_initial_states()
        model = actor.initialize_arm("gelu", states[actor.SEEDS[0]])
        actor.check_common_parameter_bytes(model, states[actor.SEEDS[0]])

    def test_parameter_counts_shared_bytes_and_initial_functions(self) -> None:
        states = actor.make_initial_states()
        self.assertEqual(len(set(actor.tensor_state_sha256(s) for s in states.values())), 2)
        for seed, state in states.items():
            reference_gelu = e477.initialize_arm("gelu", 0.0, state).eval()
            reference_mix = e477.initialize_arm("same_channel_mix", 0.0, state).eval()
            models = {arm: actor.initialize_arm(arm, state).eval() for arm in actor.ARMS}
            for arm, model in models.items():
                actor.check_common_parameter_bytes(model, state)
                expected_count = (
                    actor.GATED_PARAMETERS
                    if arm in ("static_mix", *actor.CONTEXT_MODES)
                    else actor.BASE_PARAMETERS
                )
                self.assertEqual(sum(p.numel() for p in model.parameters()), expected_count)
                if arm in ("static_mix", *actor.CONTEXT_MODES):
                    self.assertTrue(torch.equal(model.head[1].gate_coeff, torch.zeros(64)))
            torch.manual_seed(seed)
            gelu_model = actor.ContextMixtureRNN("gelu")
            gelu_rng = torch.random.get_rng_state()
            torch.manual_seed(seed)
            static_model = actor.ContextMixtureRNN("static_mix")
            static_rng = torch.random.get_rng_state()
            self.assertTrue(torch.equal(gelu_rng, static_rng))
            self.assertFalse(hasattr(gelu_model.head[1], "gate_coeff"))
            self.assertTrue(torch.equal(static_model.head[1].gate_coeff, torch.zeros(64)))

            torch.manual_seed(48402 + seed % 97)
            inputs = torch.randn(2, 19, actor.FEATURES)
            gelu_expected, gelu_state = reference_gelu(inputs)
            mix_expected, mix_state = reference_mix(inputs)
            for arm, model in models.items():
                prediction, state_out, gate, _context = model.forward_details(inputs)
                if arm == "gelu":
                    torch.testing.assert_close(prediction, gelu_expected, atol=0, rtol=0)
                    torch.testing.assert_close(state_out, gelu_state, atol=0, rtol=0)
                else:
                    torch.testing.assert_close(prediction, mix_expected, atol=0, rtol=0)
                    torch.testing.assert_close(state_out, mix_state, atol=0, rtol=0)
                    self.assertTrue(torch.equal(gate, torch.full_like(gate, 0.5)))

    def test_zero_and_nonzero_context_rms_formula_and_gate_values(self) -> None:
        zero = torch.zeros(2, 3, actor.HIDDEN)
        self.assertTrue(torch.equal(actor.normalize_context(zero), zero))
        values = torch.linspace(-2.0, 3.0, 2 * 3 * actor.HIDDEN).reshape(2, 3, actor.HIDDEN)
        expected_normalized = values / torch.sqrt(
            (values * values).mean(dim=-1, keepdim=True) + 1e-6
        )
        torch.testing.assert_close(
            actor.normalize_context(values), expected_normalized, atol=0, rtol=0
        )
        states = actor.make_initial_states()
        torch.manual_seed(48403)
        inputs = torch.randn(2, 17, actor.FEATURES)
        for arm in actor.CONTEXT_MODES:
            model = actor.initialize_arm(arm, states[actor.SEEDS[0]])
            with torch.no_grad():
                model.head[1].gate_coeff.copy_(torch.linspace(-0.65, 0.65, actor.HIDDEN))
            _prediction, _state, gate, context = model.forward_details(inputs)
            expected_gate = torch.sigmoid(
                model.head[1].gate_coeff.view(1, 1, -1)
                * (context / torch.sqrt((context * context).mean(-1, keepdim=True) + 1e-6))
            )
            torch.testing.assert_close(gate, expected_gate, atol=0, rtol=0)
            self.assertTrue(torch.isfinite(gate).all())

    def test_nonzero_gate_and_context_gradients_are_finite_and_nonzero(self) -> None:
        states = actor.make_initial_states()
        torch.manual_seed(48404)
        inputs = torch.randn(2, 17, actor.FEATURES, requires_grad=True)
        weights = torch.randn(2, 17, 2)
        static = actor.initialize_arm("static_mix", states[actor.SEEDS[0]])
        static_prediction, _ = static(inputs)
        (static_prediction * weights).sum().backward()
        static_gradient = static.head[1].gate_coeff.grad
        self.assertIsNotNone(static_gradient)
        self.assertTrue(torch.isfinite(static_gradient).all())
        self.assertGreater(float(static_gradient.abs().sum()), 0.0)
        for arm in actor.CONTEXT_MODES:
            model = actor.initialize_arm(arm, states[actor.SEEDS[0]])
            with torch.no_grad():
                model.head[1].gate_coeff.copy_(torch.linspace(-0.55, 0.55, actor.HIDDEN))
            prediction, _state, gate, context = model.forward_details(inputs)
            self.assertIsNotNone(gate)
            self.assertIsNotNone(context)
            gate.retain_grad()
            context.retain_grad()
            (prediction * weights).sum().backward(retain_graph=True)
            coeff_grad = model.head[1].gate_coeff.grad
            self.assertIsNotNone(coeff_grad)
            self.assertTrue(torch.isfinite(coeff_grad).all())
            self.assertGreater(float(coeff_grad.abs().sum()), 0.0)
            self.assertIsNotNone(gate.grad)
            self.assertTrue(torch.isfinite(gate.grad).all())
            self.assertGreater(float(gate.grad.abs().sum()), 0.0)
            self.assertIsNotNone(context.grad)
            self.assertTrue(torch.isfinite(context.grad).all())
            self.assertGreater(float(context.grad.abs().sum()), 0.0)
            model.zero_grad(set_to_none=True)
        self.assertTrue(torch.isfinite(inputs.grad).all())

    def test_nonzero_context_gate_is_causal_and_batch_independent(self) -> None:
        states = actor.make_initial_states()
        torch.manual_seed(48405)
        inputs = torch.randn(2, 31, actor.FEATURES)
        cutoff = 17
        future_changed = inputs.clone()
        future_changed[:, cutoff:] += torch.randn_like(future_changed[:, cutoff:]) * 5
        other_sample_changed = inputs.clone()
        other_sample_changed[1] += torch.randn_like(other_sample_changed[1]) * 3
        for arm in actor.CONTEXT_MODES:
            model = actor.initialize_arm(arm, states[actor.SEEDS[0]]).eval()
            with torch.no_grad():
                model.head[1].gate_coeff.copy_(torch.linspace(-0.7, 0.7, actor.HIDDEN))
            pred, _state, gate, _context = model.forward_details(inputs)
            future_pred, _state, future_gate, _context = model.forward_details(future_changed)
            other_pred, _state, other_gate, _context = model.forward_details(other_sample_changed)
            torch.testing.assert_close(pred[:, :cutoff], future_pred[:, :cutoff], atol=0, rtol=0)
            torch.testing.assert_close(gate[:, :cutoff], future_gate[:, :cutoff], atol=0, rtol=0)
            torch.testing.assert_close(pred[0], other_pred[0], atol=0, rtol=0)
            torch.testing.assert_close(gate[0], other_gate[0], atol=0, rtol=0)

    def test_frozen_e477_output_state_and_gradient_parity(self) -> None:
        state = actor.make_initial_states()[actor.SEEDS[0]]
        result = actor.check_reference_parity(torch.device("cpu"), state)
        self.assertEqual(result["device"], "cpu")
        self.assertEqual(len(result["checks"]), 2)
        self.assertTrue(
            all(
                item["exact_forward_state_input_and_parameter_gradients"]
                and item["clipping_exercised"]
                and item["exact_clipped_adamw_step"]
                and item["preclip_gradient_norm"] > actor.CLIP_NORM
                for item in result["checks"]
            )
        )

    def test_initial_gate_mix_and_gelu_anchor_are_distinct(self) -> None:
        state = actor.make_initial_states()[actor.SEEDS[0]]
        gelu = actor.initialize_arm("gelu", state).eval()
        fixed = actor.initialize_arm("fixed_mix", state).eval()
        torch.manual_seed(48406)
        inputs = torch.randn(2, 13, actor.FEATURES)
        gelu_out, _ = gelu(inputs)
        fixed_out, _ = fixed(inputs)
        self.assertFalse(torch.equal(gelu_out, fixed_out))
        self.assertTrue(torch.isfinite(F.gelu(inputs)).all())


if __name__ == "__main__":
    unittest.main()
