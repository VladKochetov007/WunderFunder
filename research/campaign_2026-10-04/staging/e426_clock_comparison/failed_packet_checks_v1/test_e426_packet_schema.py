"""Stateless guards for E426 exporter-to-scorer cache-name/provenance joins."""

from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
PACKET = ROOT / "research/campaign_2026-10-04/staging/e426_clock_comparison"
SCRIPT = PACKET / "scripts/score_e426_clocks.py"
SPEC = importlib.util.spec_from_file_location("e426_score_packet", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SCORE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCORE)


class PacketSchemaTests(unittest.TestCase):
    def test_exporter_screen_validator_cache_names_join(self) -> None:
        screen = json.loads((PACKET / "specs/e426_clock_screen_v1.json").read_text())
        score = json.loads(
            (PACKET / "specs/e426_clock_fold0_paired_score_v1.draft.json").read_text()
        )
        for clock in SCORE.RUNS:
            expected = f"e426_clock_{clock}_fp32"
            with self.subTest(clock=clock):
                self.assertEqual(SCORE.PREDICTIONS[clock], expected)
                self.assertIn(expected, screen["arms"][clock])
                self.assertIn(f"cache/pred_{expected}_valid.npy", score["inputs"])
                export = json.loads(
                    (PACKET / f"specs/e426_clock_export_{clock}_ep3_fp32_v1.draft.json").read_text()
                )
                self.assertEqual(
                    export["argv"][export["argv"].index("--output-name") + 1], expected
                )
                self.assertIn(f"cache/pred_{expected}_valid.npy", export["artifacts"])
                self.assertIn(f"cache/pred_{expected}_valid.json", export["artifacts"])

    def test_training_state_must_match_attempt_artifact_bytes(self) -> None:
        SCORE.require_artifact_hash(
            {"artifacts/runs/a/training_state.pt": "abc"},
            "artifacts/runs/a/training_state.pt",
            "abc",
            "training state",
        )
        with self.assertRaisesRegex(ValueError, "training state"):
            SCORE.require_artifact_hash(
                {}, "artifacts/runs/a/training_state.pt", "abc", "training state"
            )
        with self.assertRaisesRegex(ValueError, "training state"):
            SCORE.require_artifact_hash(
                {"artifacts/runs/a/training_state.pt": "old"},
                "artifacts/runs/a/training_state.pt",
                "abc",
                "training state",
            )

    def test_score_job_argv_has_only_the_validator_and_output_option(self) -> None:
        score = json.loads(
            (PACKET / "specs/e426_clock_fold0_paired_score_v1.draft.json").read_text()
        )
        self.assertEqual(
            score["argv"],
            [
                "{python}",
                "research/campaign_2026-10-04/staging/e426_clock_comparison/scripts/score_e426_clocks.py",
                "--output",
                "research/campaign_2026-10-04/staging/e426_clock_comparison/results/e426_clock_screen_v1.json",
            ],
        )
        self.assertIn(
            "research/campaign_2026-10-04/staging/e426_clock_comparison/tests/test_e426_packet_schema.py",
            score["code"],
        )


if __name__ == "__main__":
    unittest.main()
