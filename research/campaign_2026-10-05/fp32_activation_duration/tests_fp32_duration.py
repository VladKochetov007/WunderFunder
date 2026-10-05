"""Bounded synthetic checks for the E489 FP32 duration candidate."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [
    str(ROOT / "research/campaign_2026-10-05/activation_optimizer"),
    str(ROOT / "research/campaign_2026-10-05/activation_placement"),
    str(ROOT / "research/campaign_2026-10-05/recurrent_activation"),
    str(ROOT / "research/campaign_2026-10-05/muon_orthogonalization_precision"),
    str(ROOT / "src"),
]

import fit_activation_placement as e474  # noqa: E402
import fit_fp32_duration as actor  # noqa: E402
import torch  # noqa: E402

from wnn.nn import RNNModel  # noqa: E402


class E489DurationTests(unittest.TestCase):
    def test_root_gate_requires_fp32_success_and_finite_deltas(self) -> None:
        proof_path = (
            actor.ROOT / "research/campaign_2026-10-05/checks/e487_saved_delta_reproduction.json"
        )
        proof = json.loads(proof_path.read_text())
        for key in ("strict_gate_fp32_pass", "all_delta_arrays_finite"):
            altered = dict(proof)
            altered[key] = False
            with tempfile.TemporaryDirectory() as directory:
                candidate = Path(directory) / "proof.json"
                candidate.write_text(json.dumps(altered))
                with self.assertRaises(ValueError):
                    actor.validate_e487_saved_delta_gate(candidate)

    def test_alpha_zero_matches_native_outputs_state_and_gradients(self) -> None:
        torch.manual_seed(271836)
        initial_model = RNNModel(42, hidden=128, layers=2, dropout=0.0, n_out=2)
        initial = {
            name: value.detach().clone() for name, value in initial_model.state_dict().items()
        }
        native = RNNModel(42, hidden=128, layers=2, dropout=0.0, n_out=2).train()
        native.load_state_dict(initial, strict=True)
        candidate = actor.initialize_arm(0.0, initial).train()
        native_input = torch.randn(2, 13, 42, requires_grad=True)
        candidate_input = native_input.detach().clone().requires_grad_(True)
        native_prediction, native_state = native(native_input)
        candidate_prediction, candidate_state, _ = candidate.forward_details(candidate_input)
        torch.testing.assert_close(candidate_prediction, native_prediction, atol=2e-5, rtol=2e-4)
        torch.testing.assert_close(candidate_state, native_state, atol=2e-5, rtol=2e-4)
        native_prediction.square().mean().backward()
        candidate_prediction.square().mean().backward()
        torch.testing.assert_close(candidate_input.grad, native_input.grad, atol=5e-5, rtol=5e-4)
        for name, parameter in native.named_parameters():
            torch.testing.assert_close(
                dict(candidate.named_parameters())[name].grad,
                parameter.grad,
                atol=5e-5,
                rtol=5e-4,
            )

    def test_optimizer_groups_are_exact_and_clipped_step_changes_parameters(self) -> None:
        torch.manual_seed(271836)
        native = RNNModel(42, hidden=128, layers=2, dropout=0.0, n_out=2)
        initial = {name: value.detach().clone() for name, value in native.state_dict().items()}
        model = actor.initialize_arm(0.1, initial)
        optimizer = actor.make_optimizer(model)
        named = dict(model.named_parameters())
        self.assertEqual(
            set(actor.MUON_NAMES), {name for name in named if name in actor.MUON_NAMES}
        )
        self.assertEqual(len(actor.MUON_NAMES), 4)
        x = torch.randn(4, 256, 42)
        prediction, state, candidate = model.forward_details(x)
        loss = prediction.square().mean() * 1e4
        loss.backward()
        before = {name: parameter.detach().clone() for name, parameter in named.items()}
        clipped, _ = actor.optimizer_update(
            model,
            optimizer,
            0,
            0,
            "synthetic-test",
            alpha=0.1,
            state=state,
            prediction=prediction,
            candidate=candidate,
        )
        self.assertTrue(clipped)
        self.assertTrue(
            any(not torch.equal(before[name], value) for name, value in model.named_parameters())
        )
        self.assertEqual(
            sorted(
                {
                    int(
                        state["step"].item()
                        if isinstance(state["step"], torch.Tensor)
                        else state["step"]
                    )
                    for state in optimizer.state.values()
                    if "step" in state
                }
            ),
            [1],
        )

    def test_epoch_shuffle_is_deterministic_and_covers_all_fit_groups(self) -> None:
        for seed in actor.SEEDS:
            first = actor.canonical_epoch_order(seed, 0)
            second = actor.canonical_epoch_order(seed, 0)
            self.assertTrue((first == second).all())
            self.assertEqual(sorted(first.tolist()), list(range(24)))
            self.assertFalse((first == actor.canonical_epoch_order(seed, 1)).all())


if __name__ == "__main__":
    with e474.full_ieee_float32():
        unittest.main()
