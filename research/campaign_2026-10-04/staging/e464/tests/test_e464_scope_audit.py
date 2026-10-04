"""Pure standard-library tests for E464 architecture scopes and state audit gates."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_head_fit.py"
SPEC = importlib.util.spec_from_file_location("e464_run_head_fit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class E464ScopeTests(unittest.TestCase):
    def test_specialist_freezes_router_expert_grus_but_keeps_both_heads(self) -> None:
        frozen, heads = module.classify_parameters(
            [
                "inp.weight", "gru.weight_ih_l0", "experts.0.weight_ih_l0",
                "prefix_proj.weight", "router.0.weight", "universal_head.0.weight",
                "expert_heads.0.weight",
            ],
            "specialist",
        )
        self.assertEqual(frozen, ["inp.weight", "gru.weight_ih_l0", "experts.0.weight_ih_l0",
                                  "prefix_proj.weight", "router.0.weight"])
        self.assertEqual(heads, ["universal_head.0.weight", "expert_heads.0.weight"])

    def test_specialist_expert_heads_do_not_match_experts_freeze_prefix(self) -> None:
        frozen, heads = module.classify_parameters(
            ["experts.0.weight_ih_l0", "expert_heads.0.weight"], "specialist"
        )
        self.assertEqual(frozen, ["experts.0.weight_ih_l0"])
        self.assertEqual(heads, ["expert_heads.0.weight"])

    def test_adapter_freezes_all_adapter_parameters_and_prefix_prior(self) -> None:
        frozen, heads = module.classify_parameters(
            ["inp.weight", "gru.weight_ih_l0", "adapter_input.weight",
             "adapter_context.weight", "adapter_output.weight", "adapter_scale",
             "prefix_prior", "head.0.weight"],
            "prefix-adapter",
        )
        self.assertEqual(len(frozen), 7)
        self.assertEqual(heads, ["head.0.weight"])

    def test_ordinary_gru_has_only_head_outside_frozen_backbone(self) -> None:
        frozen, heads = module.classify_parameters(
            ["inp.weight", "gru.weight_ih_l0", "head.0.weight"], "gru"
        )
        self.assertEqual(frozen, ["inp.weight", "gru.weight_ih_l0"])
        self.assertEqual(heads, ["head.0.weight"])

    def test_unclassified_parameter_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside the E464 freeze contract"):
            module.classify_parameters(["router.weight", "unmapped_adapter.weight"], "gru")
        with self.assertRaisesRegex(ValueError, "outside the E464 freeze contract"):
            module.classify_parameters(["input_projection.weight", "head.weight"], "gru")

    def test_non_head_parameter_or_buffer_mutation_is_rejected(self) -> None:
        before = {
            "inp.weight": {"dtype": "torch.float32", "shape": [2, 2], "sha256": "a"},
            "mean": {"dtype": "torch.float32", "shape": [2], "sha256": "b"},
            "hash_projection": {"dtype": "torch.float32", "shape": [2, 4], "sha256": "c"},
            "head.weight": {"dtype": "torch.float32", "shape": [23, 2], "sha256": "d"},
        }
        after = {name: dict(value) for name, value in before.items()}
        after["head.weight"]["sha256"] = "head-changed-allowed"
        self.assertEqual(module.compare_non_head_state(before, after, ["head.weight"]), [])
        after["mean"]["sha256"] = "normalization-changed"
        mismatches = module.compare_non_head_state(before, after, ["head.weight"])
        self.assertIn({"kind": "non_head_state_changed", "name": "mean"}, mismatches)
        after["mean"]["sha256"] = "b"
        after["hash_projection"]["sha256"] = "router-buffer-changed"
        mismatches = module.compare_non_head_state(before, after, ["head.weight"])
        self.assertIn({"kind": "non_head_state_changed", "name": "hash_projection"}, mismatches)

    def test_missing_state_key_fails_closed(self) -> None:
        before = {"gru.weight": {"shape": [2, 2], "dtype": "float32", "sha256": "x"}}
        self.assertEqual(
            module.compare_non_head_state(before, {}, ["head.weight"]),
            [{"kind": "state_key_set", "missing_after": ["gru.weight"], "unexpected_after": []}],
        )

    def test_trainable_head_shape_change_is_rejected(self) -> None:
        before = {"head.weight": {"shape": [23, 2], "dtype": "torch.float32", "sha256": "a"}}
        after = {"head.weight": {"shape": [2, 23], "dtype": "torch.float32", "sha256": "b"}}
        mismatch = module.compare_head_metadata(before, after, ["head.weight"])
        self.assertEqual(mismatch[0]["kind"], "head_tensor_layout_changed")


if __name__ == "__main__":
    unittest.main()
