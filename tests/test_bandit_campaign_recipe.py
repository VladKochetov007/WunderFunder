from __future__ import annotations

import copy
import hashlib
import json
import unittest
from pathlib import Path

from wnn.bandit_campaign_recipe import sha256_bytes, validate_bandit_recipe

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "research/campaign_2026-10-06/bandit_continuation"
SOURCE_PATH = CAMPAIGN / "SOURCE_FAMILIES.json"
SOURCE_BYTES = SOURCE_PATH.read_bytes()
SOURCE_HASH = hashlib.sha256(SOURCE_BYTES).hexdigest()
SOURCE_MANIFEST = json.loads(SOURCE_BYTES)
SPLIT_PATH = ROOT / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
SPLIT_MANIFEST = json.loads(SPLIT_PATH.read_text())


def make_realized_snapshot() -> dict[str, object]:
    """Create deterministic E510 baseline identities for a metadata-only test."""
    families = copy.deepcopy(SOURCE_MANIFEST["families"])
    for entry in families:
        if entry["status"] == "planned_fit":
            entry["status"] = "produced_by_e510"
            entry["checkpoint_sha256"] = str(entry["seed"]) * 64
            entry["provenance_sha256"] = "a" * 64
    return {
        "format": "wnn.bandit.source-families-realized",
        "version": 1,
        "base_source_families_sha256": SOURCE_HASH,
        "e508_profile_review_receipt_sha256": "e" * 64,
        "families": families,
    }


def make_cache_metadata(snapshot: dict[str, object]) -> dict[str, object]:
    """Create cache source records with exact family/run/epoch/coefficient bindings."""
    families = snapshot["families"]
    coefficients = [
        {
            "family": entry["family"],
            "run": entry["run"],
            "epoch": entry["epoch_zero_based"],
            "coefficient": entry["main_coefficient"],
        }
        for entry in families
    ]
    sources = [
        {
            "family": entry["family"],
            "run": entry["run"],
            "epoch": entry["epoch_zero_based"],
            "coefficient": entry["main_coefficient"],
            "checkpoint_sha256": entry["checkpoint_sha256"],
        }
        for entry in families
    ]
    keys = (
        "main_source_train",
        "main_source_valid",
        "main_normalized_train",
        "main_normalized_valid",
        "main_homogeneous_train",
        "main_homogeneous_valid",
        "families_train",
        "families_valid",
    )
    return {
        "format": "wnn.heterogeneous-distillation-cache",
        "version": 1,
        "fold": 0,
        "split_manifest_sha256": SOURCE_MANIFEST.get(
            "fold0_split_sha256", "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d"
        ),
        "fit_groups": SPLIT_MANIFEST["fit"],
        "evaluation_valid_groups": SPLIT_MANIFEST["evaluation"]["valid"],
        "family_order": ["strong", "full_finetuned", "raw_wmse", "samefold_baseline"],
        "homogeneous_family": "strong",
        "source_coefficients": list(reversed(coefficients)),
        "coefficient_sum": 0.8125,
        "sources": sources,
        "files": {key: {"path": f"{key}.npy"} for key in keys},
        "epoch_convention": {
            "friend_at_ep1": "second additional epoch",
            "student_epochs": 2,
            "student_checkpoint_policy": "fixed",
            "student_checkpoint_epoch": 1,
        },
    }


def make_artifact_hashes(snapshot: dict[str, object]) -> dict[str, dict[str, str]]:
    """Represent checkpoint and provenance hashes independently of cache metadata."""
    return {
        entry["run"]: {
            "checkpoint_sha256": entry["checkpoint_sha256"],
            "provenance_sha256": entry["provenance_sha256"],
        }
        for entry in snapshot["families"]
    }


class BanditRecipeValidatorTests(unittest.TestCase):
    """Exercise exact prereg/cache/source binding without loading any model data."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.snapshot = make_realized_snapshot()
        snapshot_bytes = (json.dumps(cls.snapshot, indent=2) + "\n").encode()
        cls.snapshot_sha = sha256_bytes(snapshot_bytes)
        cls.cache = make_cache_metadata(cls.snapshot)
        cls.artifact_hashes = make_artifact_hashes(cls.snapshot)

    def recipe(self, arm: str) -> dict[str, object]:
        path = CAMPAIGN / "local" / f"E50{4 + int(arm[1:]) - 1}_{arm}_PAIRED.json"
        if arm == "C1":
            path = CAMPAIGN / "local/E504_C1_PAIRED.json"
        elif arm == "C2":
            path = CAMPAIGN / "local/E505_C2_PAIRED.json"
        elif arm == "C3":
            path = CAMPAIGN / "local/E506_C3_PAIRED.json"
        elif arm == "C4":
            path = CAMPAIGN / "local/E507_C4_PAIRED.json"
        return json.loads(path.read_text())

    def validate(
        self, recipe: dict[str, object], cache: dict[str, object] | None = None
    ) -> dict[str, object]:
        return validate_bandit_recipe(
            recipe,
            SOURCE_MANIFEST,
            self.snapshot,
            self.cache if cache is None else cache,
            split_manifest=SPLIT_MANIFEST,
            sealed_families_sha256=SOURCE_HASH,
            realized_snapshot_sha256=self.snapshot_sha,
            expected_realized_snapshot_sha256=self.snapshot_sha,
            actual_artifact_hashes=self.artifact_hashes,
        )

    def test_all_sealed_preregistrations_match_their_treatment(self) -> None:
        expected = {
            "C1": ("homogeneous", 0.0),
            "C2": ("source", 0.0),
            "C3": ("source", 0.1),
            "C4": ("normalized", 0.1),
        }
        for arm, treatment in expected.items():
            with self.subTest(arm=arm):
                report = self.validate(self.recipe(arm))
                self.assertEqual((report["variant"], report["family_weight"]), treatment)
                self.assertEqual(report["source_count"], 8)
                self.assertEqual(report["source_coefficient_sum"], 0.8125)
                self.assertEqual(len(report["preregistered_commands"]), 2)

    def test_cache_checkpoint_hash_mutation_fails(self) -> None:
        cache = copy.deepcopy(self.cache)
        cache["sources"][0]["checkpoint_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "checkpoint hash"):
            self.validate(self.recipe("C2"), cache)

    def test_cache_epoch_coefficient_and_family_mutations_fail(self) -> None:
        for field, replacement in (("epoch", 2), ("coefficient", 0.2), ("family", "raw_wmse")):
            cache = copy.deepcopy(self.cache)
            cache["source_coefficients"][0][field] = replacement
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.validate(self.recipe("C2"), cache)

    def test_cache_membership_must_equal_the_sealed_split(self) -> None:
        cache = copy.deepcopy(self.cache)
        cache["fit_groups"]["train"].pop()
        with self.assertRaisesRegex(ValueError, "group identities"):
            self.validate(self.recipe("C2"), cache)

    def test_unpinned_realized_snapshot_fails(self) -> None:
        recipe = self.recipe("C1")
        with self.assertRaisesRegex(ValueError, "controller-pinned"):
            validate_bandit_recipe(
                recipe,
                SOURCE_MANIFEST,
                self.snapshot,
                self.cache,
                split_manifest=SPLIT_MANIFEST,
                sealed_families_sha256=SOURCE_HASH,
                realized_snapshot_sha256=self.snapshot_sha,
                expected_realized_snapshot_sha256="0" * 64,
                actual_artifact_hashes=self.artifact_hashes,
            )

    def test_command_or_treatment_mutation_fails(self) -> None:
        recipe = self.recipe("C3")
        recipe["commands"][0][recipe["commands"][0].index("--lr") + 1] = "0.0002"
        with self.assertRaisesRegex(ValueError, "command 0"):
            self.validate(recipe)

        recipe = self.recipe("C4")
        recipe["distillation_alpha"] = 0.8
        with self.assertRaisesRegex(ValueError, "alpha"):
            self.validate(recipe)

    def test_artifact_provenance_hash_mutation_fails(self) -> None:
        hashes = copy.deepcopy(self.artifact_hashes)
        run = next(iter(hashes))
        hashes[run]["provenance_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "provenance bytes"):
            validate_bandit_recipe(
                self.recipe("C2"),
                SOURCE_MANIFEST,
                self.snapshot,
                self.cache,
                split_manifest=SPLIT_MANIFEST,
                sealed_families_sha256=SOURCE_HASH,
                realized_snapshot_sha256=self.snapshot_sha,
                expected_realized_snapshot_sha256=self.snapshot_sha,
                actual_artifact_hashes=hashes,
            )


if __name__ == "__main__":
    unittest.main()
