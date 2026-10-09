from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import recipe_sampler as sampler  # noqa: E402

CHOICES = {
    "optimizer": {"adamw", "muon", "sgd"},
    "muon_scope": {"rnn", "hidden"},
    "schedule": {"onecycle", "wsd", "sgdr"},
    "loss": {"wmse", "huber", "mse"},
    "activation": {"gelu", "fixed_mix"},
    "architecture": {"gru", "dual", "tcn"},
    "precision": {"fp16", "bf16", "fp32"},
    "batch_order": {"random", "seeded", "qmc"},
    "train_source": {"train", "valid_devA", "train+valid", "valid"},
    "cell": {"gru", "lstm"},
    "encoder": {"linear", "mlp", "tokens", "conv"},
    "recurrent_initialization": {"default", "chrono", "orth", "chrono_orth"},
}
HEX = "a" * 64
MUON_POLICY = {
    "scope": "rnn",
    "gatewise": "all",
    "orthogonalization_dtype_policy": "fp32",
    "selected_orthogonalization_dtype": "float32",
    "ns_steps": 5,
    "rms": 0.2,
    "lr_multiplier": 1.0,
}


def recipe() -> dict:
    return {
        "model": {
            "arch": "gru",
            "hidden": 256,
            "layers": 2,
            "cell": "gru",
            "encoder": "linear",
            "activation": "gelu",
            "activation_placement": "readout_only",
            "recurrent_initialization": "default",
        },
        "optimizer": {
            "name": "adamw",
            "learning_rate": 0.001,
            "weight_decay": 0.01,
            "gradient_clip_norm": 1.0,
            "momentum": None,
        },
        "loss": {"name": "wmse", "huber_delta": None},
        "schedule": {
            "name": "onecycle",
            "implementation": (
                "wnn.training_dynamics.build_scheduler: OneCycleLR with D02 R2 warmup, "
                "divisor and final-divisor settings"
            ),
            "parameters": {
                "total_steps": 20,
                "peak_lr": 0.001,
                "warmup_frac": 0.05,
                "anneal_strategy": "cos",
                "div_factor": 25.0,
                "final_div_factor": 100.0,
                "three_phase": False,
                "cycle_momentum": True,
                "beta1_trajectory": "OneCycleLR cycles beta1; record the observed per-update trace",
                "base_momentum": 0.85,
                "max_momentum": 0.95,
            },
        },
        "sampling": {
            "membership_mode": "frozen_file",
            "membership_sha256": HEX,
            "fold": 0,
            "seed": 7,
            "batch_order": "seeded",
            "train_source": "train",
        },
        "data": {"split_manifest_sha256": HEX, "sample_membership_sha256": HEX},
        "ancestry": {"initializer": "fresh_random_seeded", "parents": []},
        "fit": {"epochs": 2, "update_budget": 20},
        "implementation_dependency": {"state": "pending_recipe_aware_runner"},
        "precision": "fp32",
        "identity_hashes": {
            "source_sha256": HEX,
            "data_sha256": HEX,
            "ancestry_sha256": HEX,
        },
    }


def candidate(candidate_id: str, raw_recipe: dict, *tags: str) -> dict:
    return {"candidate_id": candidate_id, "recipe": raw_recipe, "mechanism_tags": list(tags)}


class RecipeSamplerTests(unittest.TestCase):
    def test_seed_reproduces_unique_sample_and_preserves_matched_block(self) -> None:
        base = recipe()
        rows = [
            candidate("A", base, "optimizer", "target"),
            candidate(
                "B",
                _change(base, "loss.name", "huber", "loss.huber_delta", 1.0),
                "optimizer",
                "target",
            ),
            candidate(
                "C",
                _change(
                    base,
                    "optimizer.name",
                    "muon",
                    "optimizer.muon_scope",
                    "rnn",
                    "optimizer.muon_policy",
                    MUON_POLICY,
                ),
                "optimizer",
                "target",
            ),
            candidate(
                "D",
                _change(
                    base,
                    "optimizer.name",
                    "muon",
                    "optimizer.muon_scope",
                    "rnn",
                    "optimizer.muon_policy",
                    MUON_POLICY,
                    "loss.name",
                    "huber",
                    "loss.huber_delta",
                    1.0,
                ),
                "optimizer",
                "target",
            ),
            candidate("E", _change(base, "model.hidden", 128), "representation"),
            candidate("F", _change(base, "schedule.name", "wsd"), "schedule"),
            candidate("G", _change(base, "sampling.fold", 1), "sample"),
        ]
        first = sampler.select_candidates(
            rows, 7, 20261009, ["A", "B", "C", "D"], ["representation", "schedule", "sample"]
        )
        second = sampler.select_candidates(
            rows, 7, 20261009, ["A", "B", "C", "D"], ["representation", "schedule", "sample"]
        )
        self.assertEqual(
            [item["recipe_hash"] for item in first], [item["recipe_hash"] for item in second]
        )
        self.assertEqual(first[0]["candidate_ids"], ["A"])
        self.assertEqual(first[1]["candidate_ids"], ["B"])
        self.assertEqual(first[2]["candidate_ids"], ["C"])
        self.assertEqual(first[3]["candidate_ids"], ["D"])
        self.assertEqual(len({item["recipe_hash"] for item in first}), 7)
        tags = {tag for item in first for tag in item["mechanism_tags"]}
        self.assertTrue({"representation", "schedule", "sample"}.issubset(tags))
        expanded = rows + [candidate("H", _change(base, "model.hidden", 512))]
        seeded_fills = {
            sampler.select_candidates(expanded, 5, seed, ["A", "B", "C", "D"], [])[-1][
                "recipe_hash"
            ]
            for seed in range(32)
        }
        self.assertGreater(len(seeded_fills), 1)
        with self.assertRaises(sampler.RecipeError):
            sampler.select_candidates(rows, 25, 1, ["A", "B", "C", "D"], [])

    def test_aliases_canonicalize_to_one_recipe_hash(self) -> None:
        plain = recipe()
        alias = copy.deepcopy(plain)
        alias["optimizer"]["name"] = "AdamW"
        alias["loss"] = {"name": "weighted_mse", "huber_delta": 0.5}
        self.assertEqual(sampler.recipe_hash(plain), sampler.recipe_hash(alias))
        huber_alias = copy.deepcopy(plain)
        huber_alias["loss"] = {"name": "huber_d1"}
        huber_explicit = copy.deepcopy(plain)
        huber_explicit["loss"] = {"name": "huber", "huber_delta": 1.0}
        self.assertEqual(sampler.recipe_hash(huber_alias), sampler.recipe_hash(huber_explicit))

        rows = [
            candidate("A", plain, "optimizer", "target"),
            candidate("A_alias", alias, "alias"),
            candidate("B", _change(plain, "loss.name", "huber", "loss.huber_delta", 1.0)),
            candidate(
                "C",
                _change(
                    plain,
                    "optimizer.name",
                    "muon",
                    "optimizer.muon_scope",
                    "rnn",
                    "optimizer.muon_policy",
                    MUON_POLICY,
                ),
            ),
            candidate(
                "D",
                _change(
                    plain,
                    "optimizer.name",
                    "muon",
                    "optimizer.muon_scope",
                    "rnn",
                    "optimizer.muon_policy",
                    MUON_POLICY,
                    "loss.name",
                    "huber",
                    "loss.huber_delta",
                    1.0,
                ),
            ),
        ]
        selected = sampler.select_candidates(rows, 4, 3, ["A", "B", "C", "D"], [])
        self.assertEqual(selected[0]["candidate_ids"], ["A", "A_alias"])

    def test_invalid_cli_value_and_incompatible_muon_architecture_fail_closed(self) -> None:
        bad = recipe()
        bad["model"]["arch"] = "transformer"
        with self.assertRaises(sampler.RecipeError):
            sampler.validate_recipe(bad, CHOICES)
        bad = recipe()
        bad["model"]["arch"] = "tcn"
        bad["optimizer"] = {"name": "muon", "muon_scope": "rnn"}
        with self.assertRaises(sampler.RecipeError):
            sampler.validate_recipe(bad, CHOICES)

    def test_huber_delta_is_required_and_part_of_recipe_identity(self) -> None:
        missing = recipe()
        missing["loss"] = {"name": "huber"}
        with self.assertRaises(sampler.RecipeError):
            sampler.validate_recipe(missing, CHOICES)
        delta_one = recipe()
        delta_one["loss"] = {"name": "huber", "huber_delta": 1.0}
        delta_two = copy.deepcopy(delta_one)
        delta_two["loss"]["huber_delta"] = 2.0
        sampler.validate_recipe(delta_one, CHOICES)
        sampler.validate_recipe(delta_two, CHOICES)
        self.assertNotEqual(sampler.recipe_hash(delta_one), sampler.recipe_hash(delta_two))

    def test_distinct_fold_changes_recipe_identity(self) -> None:
        fold_zero = recipe()
        fold_one = copy.deepcopy(fold_zero)
        fold_one["sampling"]["fold"] = 1
        fold_one["data"]["split_manifest_sha256"] = "b" * 64
        self.assertNotEqual(sampler.recipe_hash(fold_zero), sampler.recipe_hash(fold_one))

    def test_old_onecycle_schedule_does_not_claim_d02_r2_source_fidelity(self) -> None:
        changed = recipe()
        changed["schedule"]["parameters"]["warmup_frac"] = 0.3
        changed["schedule"]["parameters"]["final_div_factor"] = 10000.0
        with self.assertRaises(sampler.RecipeError):
            sampler.validate_recipe(changed, CHOICES)

    def test_matched_quartet_keeps_implementation_ancestry_in_signature(self) -> None:
        base = recipe()
        rows = [
            candidate("A", base),
            candidate("B", _change(base, "loss.name", "huber", "loss.huber_delta", 1.0)),
            candidate(
                "C",
                _change(
                    base,
                    "optimizer.name",
                    "muon",
                    "optimizer.muon_scope",
                    "rnn",
                    "optimizer.muon_policy",
                    MUON_POLICY,
                ),
            ),
            candidate(
                "D",
                _change(
                    base,
                    "optimizer.name",
                    "muon",
                    "optimizer.muon_scope",
                    "rnn",
                    "optimizer.muon_policy",
                    MUON_POLICY,
                    "loss.name",
                    "huber",
                    "loss.huber_delta",
                    1.0,
                ),
            ),
        ]
        rows[2]["recipe"]["source"] = {"backend": "different implementation"}
        with self.assertRaises(sampler.RecipeError):
            sampler.select_candidates(rows, 4, 3, ["A", "B", "C", "D"], [])

    def test_pending_runner_blocks_admission(self) -> None:
        with self.assertRaisesRegex(sampler.RecipeError, "recipe-aware runner"):
            sampler.assert_recipe_runner_ready(recipe())


def _change(raw: dict, *pairs: object) -> dict:
    changed = copy.deepcopy(raw)
    for path, value in zip(pairs[::2], pairs[1::2], strict=True):
        target = changed
        fields = str(path).split(".")
        for field in fields[:-1]:
            target = target.setdefault(field, {})
        target[fields[-1]] = value
    return changed


if __name__ == "__main__":
    unittest.main()
