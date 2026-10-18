"""Exercise exact F3 schedule/membership gates without touching real data."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from f3_preflight import FOLD_SHA, VENDOR_SHA, checked_module, membership_record


class F3PreflightTests(unittest.TestCase):
    """Cover the changed fold gates and retained generic verifier binding."""

    def fixture(self) -> SimpleNamespace:
        """Make only small metadata lists, with the observed F3 counts."""
        return SimpleNamespace(
            fold=3,
            n_folds=4,
            sha256=FOLD_SHA,
            train_groups=list(range(10539)),
            valid_groups=list(range(1368)),
            evaluation_groups=list(range(505)),
            components=[("valid", group, group % 145) for group in range(505)],
        )

    def test_f3_membership_and_schedule(self) -> None:
        """Correct metadata produces exact batch/update expectations."""
        metadata = membership_record(self.fixture())
        helper = checked_module("frozen_f1_preflight", VENDOR_SHA)
        schedule = helper._schedule_evidence(metadata)
        self.assertEqual(schedule["batches_per_epoch"], 1488)
        self.assertEqual(schedule["stage1_updates"], 29760)
        self.assertEqual(schedule["terminal_updates"], 89280)
        self.assertEqual(schedule["terminal_batches"], 8928)

    def test_reject_wrong_fold(self) -> None:
        """F1 metadata cannot enter a new F3 receipt."""
        split = self.fixture()
        split.fold = 1
        with self.assertRaises(ValueError):
            membership_record(split)

    def test_reject_wrong_sha(self) -> None:
        """Matching counts cannot hide a different fold identity."""
        split = self.fixture()
        split.sha256 = "0" * 64
        with self.assertRaises(ValueError):
            membership_record(split)

    def test_reject_wrong_membership_count(self) -> None:
        """An evaluation omission cannot pass by total fit count."""
        split = self.fixture()
        split.evaluation_groups.pop()
        with self.assertRaises(ValueError):
            membership_record(split)

    def test_reject_wrong_component_count(self) -> None:
        """Component coverage remains a separate condition."""
        split = self.fixture()
        split.components = [("valid", group, group % 144) for group in range(505)]
        with self.assertRaises(ValueError):
            membership_record(split)

    def test_reject_modified_helper_pin(self) -> None:
        """Do not import a verifier under the wrong source identity."""
        with self.assertRaises(ValueError):
            checked_module("frozen_f1_preflight", "0" * 64)


if __name__ == "__main__":
    unittest.main()
