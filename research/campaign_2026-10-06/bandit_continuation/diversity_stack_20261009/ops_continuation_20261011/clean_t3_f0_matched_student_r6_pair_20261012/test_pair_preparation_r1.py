from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from validate_pair_preparation_r1 import (  # noqa: E402
    BindingError,
    inspect_pending_readiness,
    verify_command_derivation,
    verify_manifest_bindings,
)


PACKET = Path(__file__).resolve().parent
PLAN = json.loads((PACKET / "PAIR_PREPARATION_R1.json").read_text())
PROJECT_ROOT = Path(str(PLAN["project_root"]))
BASE_RECIPE = PROJECT_ROOT / PLAN["immutable_bindings"]["base_recipe"]["path"]


class PairedPreparationTests(unittest.TestCase):
    """Exercise the pending gate against the actual small producer receipts."""

    def test_actual_parent_and_source_bindings_block_before_fit(self) -> None:
        result = inspect_pending_readiness(PLAN, PROJECT_ROOT)
        self.assertEqual(result["status"], "BLOCKED_PENDING_CACHE_REVIEW_PROFILE_AND_ROOT_RELEASE")
        self.assertGreaterEqual(result["static_file_bindings_verified"], 20)
        self.assertFalse(result["launch_allowed"])
        self.assertFalse(result["cache_materialized"])
        self.assertFalse(result["student_fit_started"])
        self.assertFalse(result["torch_imported"])
        self.assertTrue(any("cache" in item.lower() for item in result["blockers"]))
        self.assertTrue(any("seed 5" in item for item in result["blockers"]))

    def test_parent_digest_mutation_fails_closed(self) -> None:
        changed = copy.deepcopy(PLAN)
        changed["immutable_bindings"]["parents"][0]["checkpoint"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(BindingError, "SHA256 differs"):
            verify_manifest_bindings(changed, PROJECT_ROOT)

    def test_frozen_command_derivation_and_profile_window(self) -> None:
        recipe = json.loads(BASE_RECIPE.read_text())
        verify_command_derivation(PLAN, recipe)
        profile = PLAN["profile"]["argv_template"]
        self.assertEqual(profile[profile.index("--batch") + 1], "16")
        self.assertEqual(profile[profile.index("--chunk") + 1], "2000")
        self.assertEqual(profile[profile.index("--precision") + 1], "fp16")


if __name__ == "__main__":
    unittest.main()
