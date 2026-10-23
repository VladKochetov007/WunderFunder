"""Verify the V6 renderer consumes the exact exporter V4 review schema."""

from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path

PACKET = Path(__file__).resolve().parent
CAMPAIGN_PREPARATION = PACKET.parent
EXPORT_PACKET = CAMPAIGN_PREPARATION / "first_cell_evaluation_export_v4"
REVIEW_PATH = EXPORT_PACKET / "EXPORT_SOURCE_REVIEW_V4.json"
REVIEW_BYTES = 8_712
REVIEW_SHA256 = "64c93d8279a2daeb15f2750e6366a0fe94a268c7b8e9a04dbdb28e4a36a56e73"
RENDERER = PACKET / "render_admission_remote_v6.py"


def _load_renderer():
    """Load only the standard-library local renderer module."""
    spec = importlib.util.spec_from_file_location("admission_v6_renderer_fixture", RENDERER)
    if spec is None or spec.loader is None:
        raise AssertionError("could not load the V6 renderer source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AdmissionV6ExportReviewContractTests(unittest.TestCase):
    def test_exact_actual_review_rows_join_to_all_three_producer_inputs(self) -> None:
        import hashlib

        raw = REVIEW_PATH.read_bytes()
        self.assertEqual(len(raw), REVIEW_BYTES)
        self.assertEqual(hashlib.sha256(raw).hexdigest(), REVIEW_SHA256)
        review = json.loads(raw)
        renderer = _load_renderer()
        rows = renderer.reviewed_file_map(review)
        self.assertEqual(len(rows), 12)
        joined = renderer.checked_producer_review_pins(review)
        self.assertEqual(
            joined,
            {
                "export_first_cell_v4.py": {
                    "path": (
                        "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/"
                        "first_cell_evaluation_export_v4/export_first_cell_v4.py"
                    ),
                    "bytes": 66_790,
                    "sha256": "87b6d3626bdafe6353865d95b1d8116bd187cb4da46888815658db6fc192738e",
                },
                "EXPORT_BINDING_V4.json": {
                    "path": (
                        "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/"
                        "first_cell_evaluation_export_v4/EXPORT_BINDING_V4.json"
                    ),
                    "bytes": 15_796,
                    "sha256": "4a8ae22204971ac07d6e5cdfd9f65a62e9f8440831035db7271e4e0da3d83abd",
                },
                "EXPORT_SOURCE_MANIFEST.json": {
                    "path": (
                        "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/"
                        "first_cell_evaluation_export_v4/EXPORT_SOURCE_MANIFEST.json"
                    ),
                    "bytes": 27_569,
                    "sha256": "a44f8a3fd3fda1efdda578c8694036dffaf72ce6999e4e9095f6298365b35621",
                },
            },
        )

    def test_duplicate_or_malformed_review_rows_are_rejected(self) -> None:
        renderer = _load_renderer()
        review = json.loads(REVIEW_PATH.read_bytes())
        duplicate = json.loads(json.dumps(review))
        duplicate["reviewed_files"].append(duplicate["reviewed_files"][0])
        with self.assertRaises(ValueError):
            renderer.reviewed_file_map(duplicate)

        malformed = json.loads(json.dumps(review))
        malformed["reviewed_files"][0]["extra"] = "unexpected"
        with self.assertRaises(ValueError):
            renderer.reviewed_file_map(malformed)

    def test_reviewed_producer_pin_mismatch_is_rejected(self) -> None:
        renderer = _load_renderer()
        review = json.loads(REVIEW_PATH.read_bytes())
        exporter_row = next(
            row
            for row in review["reviewed_files"]
            if row["path"].endswith("/export_first_cell_v4.py")
        )
        exporter_row["bytes"] += 1
        with self.assertRaises(ValueError):
            renderer.checked_producer_review_pins(review)

    def test_first_failed_export_is_pinned_as_root_only_history(self) -> None:
        renderer = _load_renderer()
        historical = renderer.bind_previous_export_failure()
        self.assertEqual(
            historical["classification"],
            "FIRST_EXPORT_INVOCATION_FAILED_IN_SOURCE_METADATA_GATE",
        )
        self.assertIs(historical["historical_only"], True)
        self.assertIs(historical["root_only_not_staged_to_B"], True)
        self.assertIs(historical["old_unit_must_not_be_queried_stopped_or_retried"], True)
        self.assertEqual(
            historical["unit"], "computer-vision-homework-fg19-first-cell-export-v1.service"
        )
        self.assertEqual(historical["invocation_id"], "da6c9e1f5897478f9f3f686de58e39cc")
        self.assertEqual(
            set(historical["files"]),
            {
                "launch_acceptance",
                "root_transport",
                "startup_observation",
                "startup_transport",
                "unit_log_tail",
                "failure_review",
            },
        )


if __name__ == "__main__":
    unittest.main()
