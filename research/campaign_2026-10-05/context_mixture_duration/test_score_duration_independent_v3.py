"""Synthetic CPU tests for the independent E488 evidence/scoring helpers."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "research/campaign_2026-10-05/activation_placement"))

import fit_activation_placement as e474  # noqa: E402
import score_duration_independent_v3 as scorer  # noqa: E402


class E488IndependentScorerTests(unittest.TestCase):
    """Exercise model identity, mixture equations, schedules and moment math."""

    def test_repository_root_resolves_from_scorer_file_and_required_inputs_exist(self):
        self.assertEqual(scorer.ROOT, Path(__file__).resolve().parents[3])
        self.assertTrue(scorer.DATA_PATH.is_file())
        self.assertTrue((scorer.ROOT / "src/wnn/nn.py").is_file())
        self.assertTrue(
            (
                scorer.ROOT
                / "research/campaign_2026-10-05/activation_context_mixture/fit_context_mixture_v2.py"
            ).is_file()
        )
        self.assertFalse((scorer.ROOT / "research/research").exists())

    def test_assembled_real_fit_and_probe_specs_have_exact_dependency_roles(self):
        fit_spec = json.loads((HERE / "e488_twelve_fit_v3.sealed.json").read_text())
        probe_spec = json.loads((HERE / "e488_gpu_probe_v2.sealed.json").read_text())
        scorer.validate_fit_spec(fit_spec)
        scorer.validate_probe_spec_roles(probe_spec)
        fit_code = {entry["path"] for entry in fit_spec["code"]}
        fit_inputs = {entry["path"] for entry in fit_spec["inputs"]}
        eval_path = "src/wnn/evaluation.py"
        self.assertNotIn(eval_path, fit_code)
        self.assertNotIn(eval_path, fit_inputs)
        self.assertEqual(
            scorer.EXPECTED_HASHES["wnn.evaluation.py"],
            scorer.sha256_file(scorer.ROOT / eval_path),
        )
        self.assertIn(
            "research/campaign_2026-10-05/context_mixture_duration/E488_DESIGN.md",
            fit_code,
        )
        self.assertNotIn(
            "research/campaign_2026-10-05/context_mixture_duration/E488_DESIGN.md",
            fit_inputs,
        )
        altered = json.loads(json.dumps(fit_spec))
        doc_path = "research/campaign_2026-10-05/context_mixture_duration/E488_DESIGN.md"
        moved = next(entry for entry in altered["code"] if entry["path"] == doc_path)
        altered["code"].remove(moved)
        altered["inputs"].append(moved)
        with self.assertRaisesRegex(ValueError, "code/input role inventory"):
            scorer.validate_fit_spec(altered)
        altered_probe = json.loads(json.dumps(probe_spec))
        altered_probe["inputs"].pop()
        with self.assertRaisesRegex(ValueError, "probe code/input role inventory"):
            scorer.validate_probe_spec_roles(altered_probe)

    def test_baseline_initial_bytes_match_frozen_e474_model(self):
        for seed in scorer.SEEDS:
            torch.manual_seed(seed)
            reference = e474.PlacementRNN("gelu", "gelu")
            torch.manual_seed(seed)
            candidate = scorer.IndependentE488Model("gelu")
            reference_state = reference.state_dict()
            candidate_state = candidate.state_dict()
            self.assertEqual(reference_state.keys(), candidate_state.keys())
            for key in reference_state:
                self.assertTrue(torch.equal(reference_state[key], candidate_state[key]), key)
            self.assertEqual(
                scorer.state_tensor_sha256(reference_state),
                scorer.state_tensor_sha256(candidate_state),
            )

    def test_all_head_mixture_formulas_match_frozen_e484_source(self):
        sys.path.insert(0, str(ROOT / "research/campaign_2026-10-05/activation_context_mixture"))
        import fit_context_mixture_v2 as e484

        hidden = torch.linspace(-3, 3, 2 * scorer.HIDDEN).reshape(1, 2, scorer.HIDDEN)
        recurrent = torch.flip(hidden, dims=(1,))
        for arm in scorer.ARMS:
            torch.manual_seed(271839)
            reference = e484.ContextMixtureActivation(arm)
            candidate = scorer.ManualGate(arm)
            if arm in ("static_mix", "recurrent_mix"):
                candidate.gate_coeff.data.copy_(reference.gate_coeff.data)
            ref_value, ref_gate, _context = reference.forward_details(hidden, recurrent, hidden)
            value, gate = candidate(hidden, recurrent)
            self.assertTrue(torch.equal(ref_value, value), arm)
            if ref_gate is None:
                self.assertIsNone(gate)
            else:
                self.assertTrue(torch.equal(ref_gate, gate), arm)

    def test_canonical_epoch_shuffles_cover_fit_positions_and_are_seed_paired(self):
        for seed in scorer.SEEDS:
            orders = [scorer.canonical_epoch_order(seed, epoch) for epoch in range(3)]
            for order in orders:
                self.assertEqual(sorted(order.tolist()), list(range(24)))
            self.assertFalse(np.array_equal(orders[0], orders[1]))
            self.assertTrue(np.array_equal(orders[0], scorer.canonical_epoch_order(seed, 0)))
        self.assertFalse(
            np.array_equal(
                scorer.canonical_epoch_order(scorer.SEEDS[0], 0),
                scorer.canonical_epoch_order(scorer.SEEDS[1], 0),
            )
        )

    def test_telemetry_epoch_batch_chunk_mapping_at_boundaries(self):
        fit_nodes = [{"row_group": 1000 + index} for index in range(24)]
        for step in (0, 46, 236, 473, 947, 1421):
            epoch, local = divmod(step, scorer.UPDATES_PER_EPOCH)
            batch, chunk = divmod(local, 79)
            expected_positions = scorer.canonical_epoch_order(scorer.SEEDS[0], epoch)
            expected_groups = [
                fit_nodes[int(index)]["row_group"]
                for index in expected_positions[batch * scorer.BATCH : (batch + 1) * scorer.BATCH]
            ]
            self.assertEqual(len(expected_groups), 4)
            self.assertEqual(chunk * scorer.CHUNK_ROWS, (local % 79) * 256)
            gate = (
                None
                if scorer.ARMS[0] == "gelu"
                else {
                    "min": 0.5,
                    "max": 0.5,
                    "mean": 0.5,
                    "saturated_fraction_at_0_01": 0.0,
                }
            )
            row = {
                "global_update_zero_based": step,
                "update_index_zero_based": step,
                "epoch_index_zero_based": epoch,
                "batch_index_zero_based": batch,
                "chunk_start": chunk * scorer.CHUNK_ROWS,
                "fit_row_groups": expected_groups,
                "loss": 1.0,
                "preclip_global_norm": 2.0,
                "clip_factor": 1.0 / (2.0 + 1e-6),
                "head_update_rms": 0.01,
                "state_abs_max": 0.5,
                "gradient_rms_preclip": 0.1,
                "preclip_gradient_rms": {"inp": 0.1, "gru": 0.2, "head": 0.3},
                "parameter_update_rms": {"inp": 0.001, "gru": 0.002, "head": 0.003},
                "gate": gate,
                "gate_coefficient_rms": None,
            }
            scorer.validate_sample_telemetry(row, scorer.SEEDS[0], "gelu", fit_nodes, step)

    def test_original_and_actual_charged_projection_are_recomputed(self):
        probe = {
            "probe_seconds": 5.645548842992866,
            "full_fit_projection_seconds": 633.0783077389933,
            "projection_safety_multiplier": 1.5,
            "fixed_full_fit_overhead_margin_seconds": 120,
            "projected_study_seconds_with_margin": 1075.2630104514828,
        }
        attempt = {"elapsed_seconds": 11.659668447}
        result = scorer.verify_reviewed_projection(probe, attempt)
        self.assertAlmostEqual(result["wrapper_projection_seconds"], 1075.2630104514828)
        self.assertAlmostEqual(result["charged_projection_seconds"], 1081.277130055489)
        probe["projected_study_seconds_with_margin"] += 1
        with self.assertRaisesRegex(ValueError, "wrapper projection arithmetic"):
            scorer.verify_reviewed_projection(probe, attempt)

    def test_target_moments_are_invariant_to_component_order_and_translation(self):
        rng = np.random.default_rng(2701)
        targets = rng.normal(size=(16, 2, 6))
        targets[:, :, 0] = 1
        targets[:, :, 1:3] = rng.normal(size=(16, 2, 2))
        targets[:, :, 3:5] = 0.5
        targets[:, :, 5] = 0.2
        counts = rng.integers(1, 10, size=16)
        expected = scorer.score_target_moments(targets, counts)
        order = rng.permutation(16)
        observed = scorer.score_target_moments(targets[order], counts[order])
        self.assertTrue(np.allclose(expected, observed, atol=1e-14, rtol=0))

    def test_registered_snapshot_contrast_inventory(self):
        names = scorer.expected_contrast_names()
        self.assertEqual(len(names), 9)
        self.assertEqual(names[-1], "(recurrent_3 - static_3) - (recurrent_1 - static_1)")
        self.assertEqual(len(scorer.SEEDS) * len(names), 27)

    def test_pointwise_model_prediction_is_causal_and_chunk_state_carries(self):
        torch.manual_seed(271839)
        model = scorer.IndependentE488Model("recurrent_mix").eval()
        x = torch.randn(1, 512, scorer.FEATURES_DIM)
        with torch.no_grad():
            full, full_state = model(x)
            first, state = model(x[:, :256])
            second, chunk_state = model(x[:, 256:], state)
            changed = x.clone()
            changed[:, 400:] += 7.0
            changed_output, _ = model(changed)
        self.assertTrue(
            torch.allclose(full, torch.cat((first, second), dim=1), atol=1e-6, rtol=1e-6)
        )
        self.assertTrue(torch.allclose(full_state, chunk_state, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.equal(full[:, :400], changed_output[:, :400]))


if __name__ == "__main__":
    unittest.main()
