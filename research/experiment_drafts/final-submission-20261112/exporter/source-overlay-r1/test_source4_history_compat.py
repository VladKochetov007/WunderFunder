"""Focused regression checks for the Source4 history compatibility overlay."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path


PROJECT = Path(os.environ["WNN_FINAL_SUBMIT_PROJECT"]).resolve(strict=True)
OVERLAY = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(OVERLAY / "scripts"))

from package_final_tl3 import validate_source4_final_history


def valid_history() -> list[dict[str, int | bool]]:
    """Return the observed complete eight-epoch Source4 fit work counters."""
    return [
        {
            "epoch": epoch,
            "updates": 980,
            "complete": True,
            "valid_rows": 248_364_480,
            "forwarded_rows": 249_600_000,
            "real_rows": 249_600_000,
        }
        for epoch in range(8)
    ]


class Source4HistoryCompatibilityTests(unittest.TestCase):
    def test_accepts_only_exact_complete_training_counters(self) -> None:
        self.assertIsNone(validate_source4_final_history(valid_history()))

    def test_rejects_validation_loss(self) -> None:
        history = valid_history()
        history[3]["valid_loss"] = 0  # type: ignore[typeddict-item]
        with self.assertRaisesRegex(ValueError, "evaluation metrics"):
            validate_source4_final_history(history)

    def test_rejects_train_eval_metric(self) -> None:
        history = valid_history()
        history[4]["train_eval_wp"] = 0  # type: ignore[typeddict-item]
        with self.assertRaisesRegex(ValueError, "evaluation metrics"):
            validate_source4_final_history(history)

    def test_rejects_wrong_training_mask_count(self) -> None:
        history = valid_history()
        history[0]["valid_rows"] = 248_364_479
        with self.assertRaisesRegex(ValueError, "valid_rows training counter"):
            validate_source4_final_history(history)

    def test_rejects_boolean_or_float_training_counter(self) -> None:
        for bad_value in (True, 248_364_480.0):
            with self.subTest(bad_value=bad_value):
                history = valid_history()
                history[0]["valid_rows"] = bad_value  # type: ignore[assignment]
                with self.assertRaisesRegex(ValueError, "valid_rows training counter"):
                    validate_source4_final_history(history)

    def test_rejects_wrong_epoch_update_or_real_row_count(self) -> None:
        for name, bad_value in (("epoch", 1), ("updates", 979), ("real_rows", 249_599_999)):
            with self.subTest(name=name):
                history = valid_history()
                history[0][name] = bad_value
                with self.assertRaises(ValueError):
                    validate_source4_final_history(history)

    def test_rejects_incomplete_epoch_and_history(self) -> None:
        history = valid_history()
        history[0]["complete"] = False
        with self.assertRaisesRegex(ValueError, "incomplete epoch"):
            validate_source4_final_history(history)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            validate_source4_final_history(valid_history()[:-1])


if __name__ == "__main__":
    unittest.main()
