"""Focused CPU tests for source-identity bridging and completed-fit budgets."""

from __future__ import annotations

import hashlib
import json
import unittest

from r01_recovery_common import producer_source_identity, validate_epoch_records


class ProducerIdentityTests(unittest.TestCase):
    """Require precisely the one README-only source-set difference."""

    def test_61_to_60_bridge_removes_only_training_readme(self) -> None:
        files = {f"src/wnn/training/module_{i}.py": f"{i:064x}" for i in range(60)}
        files["src/wnn/training/README.md"] = "a" * 64
        result = producer_source_identity(files)
        expected = dict(files)
        expected.pop("src/wnn/training/README.md")
        digest = hashlib.sha256(
            json.dumps(expected, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self.assertEqual(result["files"], expected)
        self.assertEqual(result["source_identity_sha256"], digest)
        self.assertEqual(result["portable_only_file"], "src/wnn/training/README.md")

    def test_other_file_counts_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            producer_source_identity({"src/wnn/training/README.md": "a" * 64})


def _complete_history(*, skipped_epoch: int | None = None) -> tuple[list[dict], dict]:
    """Build minimal valid six-epoch counter records for pure invariants."""
    history = []
    provenance_rows = []
    for epoch in range(6):
        skipped = int(epoch == skipped_epoch)
        counts = {
            "optimizer_attempts": 10,
            "optimizer_completed_updates": 10 - skipped,
            "amp_skipped_updates": skipped,
            "completed_batches": 1,
        }
        history.append(
            {
                "epoch": epoch,
                "evaluation": "withheld",
                "update_diagnostics": {"epoch": counts},
                "fit_summary": {"optimizer_windows": 10},
            }
        )
        provenance_rows.append({"epoch_index": epoch, "epoch": counts})
    provenance = {
        "args": {"batch": 8, "chunk": 2000},
        "fit_row_groups": {"train": list(range(7)), "valid": [0]},
        "update_diagnostics": provenance_rows,
        "cumulative_updates": {
            "optimizer_attempts": 60,
            "optimizer_completed_updates": sum(
                row["update_diagnostics"]["epoch"]["optimizer_completed_updates"] for row in history
            ),
            "amp_skipped_updates": int(skipped_epoch is not None),
            "completed_batches": 6,
        },
    }
    return history, provenance


class EpochBudgetTests(unittest.TestCase):
    """Reject incomplete, skipped or internally inconsistent fit histories."""

    def test_accepts_six_complete_fit_epochs(self) -> None:
        history, provenance = _complete_history()
        self.assertEqual(
            validate_epoch_records(history, provenance), {"per_epoch": 10, "total": 60, "epochs": 6}
        )

    def test_rejects_amp_skip_even_when_history_is_six_epochs(self) -> None:
        history, provenance = _complete_history(skipped_epoch=3)
        with self.assertRaises(ValueError):
            validate_epoch_records(history, provenance)

    def test_actual_fold0_counter_scope_has_ten_updates_per_completed_batch(self) -> None:
        history, provenance = _complete_history()
        provenance["fit_row_groups"] = {"train": list(range(10535)), "valid": list(range(1405))}
        for epoch, row in enumerate(history):
            counts = {
                "optimizer_attempts": 14920,
                "optimizer_completed_updates": 14920,
                "amp_skipped_updates": 0,
                "completed_batches": 1492,
            }
            row["update_diagnostics"]["epoch"] = counts
            row["fit_summary"]["optimizer_windows"] = 14920
            provenance["update_diagnostics"][epoch]["epoch"] = counts
        provenance["cumulative_updates"] = {
            "optimizer_attempts": 89520,
            "optimizer_completed_updates": 89520,
            "amp_skipped_updates": 0,
            "completed_batches": 8952,
        }
        self.assertEqual(validate_epoch_records(history, provenance)["total"], 89520)
        history[0]["update_diagnostics"]["epoch"]["completed_batches"] = 14920
        with self.assertRaises(ValueError):
            validate_epoch_records(history, provenance)

    def test_rejects_missing_or_evaluated_epoch(self) -> None:
        history, provenance = _complete_history()
        with self.assertRaises(ValueError):
            validate_epoch_records(history[:-1], provenance)
        history[2]["evaluation"] = {"wp": 0.01}
        with self.assertRaises(ValueError):
            validate_epoch_records(history, provenance)
        history[2]["fit_summary"]["loss"] = float("nan")
        with self.assertRaises(ValueError):
            validate_epoch_records(history, provenance)


if __name__ == "__main__":
    unittest.main()
