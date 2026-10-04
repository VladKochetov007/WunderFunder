"""Source-only consistency checks for the three E427 production smoke drafts."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
CAMPAIGN = ROOT / "research/campaign_2026-10-04"
SMOKE = CAMPAIGN / "staging/e427_smoke_review"
FULL = CAMPAIGN / "staging/e427/specs"


def load(path: Path) -> dict:
    return json.loads(path.read_text())


class E427SmokePacketTests(unittest.TestCase):
    def test_three_smokes_are_full_shape_matched_recipe_prefixes(self) -> None:
        for arm in ("aligned", "stale", "noop"):
            with self.subTest(arm=arm):
                name = f"e427_smoke_{arm}_s3_b32x2_v1"
                smoke = load(SMOKE / f"specs/{name}.draft.json")
                full = load(FULL / f"c20261004_e427_{arm}_s3_1ep.draft.json")
                smoke_sep = smoke["argv"].index("--")
                child = smoke["argv"][smoke_sep + 1:]
                self.assertEqual(child[0], "--name")
                self.assertEqual(child[child.index("--name") + 1], name)
                self.assertEqual(child[child.index("--e427-arm") + 1], arm)
                self.assertEqual(child[child.index("--profile-steps") + 1], "2")
                self.assertEqual(child[child.index("--profile-warmup") + 1], "0")
                expected_child = full["argv"][2:]
                expected_child[expected_child.index("--name") + 1] = name
                expected_child.remove("--save-training-state")
                expected_child += ["--profile-steps", "2", "--profile-warmup", "0",
                                   "--no-eval-during-fit"]
                self.assertEqual(child, expected_child)
                for flag in ("--batch", "--chunk", "--epochs", "--init", "--distill",
                             "--distill-alpha", "--distill-boot-h", "--aux-struct",
                             "--aux-next", "--propensity", "--optim", "--precision",
                             "--checkpoint-epoch"):
                    self.assertEqual(child[child.index(flag) + 1],
                                     full["argv"][full["argv"].index(flag) + 1], flag)
                self.assertIn("--no-eval-during-fit", child)
                self.assertNotIn("--save-training-state", child)
                self.assertEqual((smoke["limits"]["cpus"], smoke["limits"]["ram_bytes"],
                                  smoke["limits"]["timeout_seconds"],
                                  smoke["limits"]["gpu_memory_bytes"],
                                  smoke["limits"]["gpu_stop_bytes"]),
                                 (8, 25_769_803_776, 900, 4_200_000_000, 4_800_000_000))
                self.assertEqual(smoke["inputs"], full["inputs"])
                self.assertEqual(len(smoke["artifacts"]), 1)
                self.assertTrue(smoke["artifacts"][0].endswith(f"/{name}.json"))


if __name__ == "__main__":
    unittest.main()
