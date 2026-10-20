"""Focused local regressions for the exact dictionary launch-acceptance pin."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bind_nll_fit_v2 as binder


class ExactDictionaryLaunchAcceptanceTests(unittest.TestCase):
    def test_exact_original_launch_acceptance_is_accepted(self) -> None:
        self.assertEqual(
            binder._require_exact_launch_acceptance(
                {"launch_acceptance_sha256": binder.DICTIONARY_LAUNCH_ACCEPTANCE_SHA256}
            ),
            binder.DICTIONARY_LAUNCH_ACCEPTANCE_SHA256,
        )

    def test_missing_launch_acceptance_pin_is_rejected_by_local_validator(self) -> None:
        with self.assertRaisesRegex(ValueError, "exact original launch acceptance"):
            binder._validate_receipts(json.dumps({"status": "ACCEPTED"}).encode(), b"{}")

    def test_mismatched_launch_acceptance_pin_is_rejected_by_local_validator(self) -> None:
        with self.assertRaisesRegex(ValueError, "exact original launch acceptance"):
            binder._validate_receipts(
                json.dumps({"launch_acceptance_sha256": "0" * 64}).encode(), b"{}"
            )

    def test_remote_validator_contains_same_exact_field_gate(self) -> None:
        source = binder._derive_template().decode("utf-8")
        self.assertIn(binder.REMOTE_PATCH, source)
        self.assertEqual(source.count('acceptance.get("launch_acceptance_sha256")'), 1)


if __name__ == "__main__":
    unittest.main()
