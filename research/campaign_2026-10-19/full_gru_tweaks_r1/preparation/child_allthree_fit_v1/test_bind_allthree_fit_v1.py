"""Focused local checks for the unbound AllThree preparation packet."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import bind_allthree_fit_v1 as binder  # noqa: E402


class AllThreeUnboundPacketTests(unittest.TestCase):
    def test_pinned_template_has_only_the_two_scientific_argv_changes(self) -> None:
        derived, nll_argv, allthree_argv = binder._derive_template()
        self.assertEqual(derived, binder.TEMPLATE.read_bytes())
        changed = [
            index
            for index, (before, after) in enumerate(zip(nll_argv, allthree_argv, strict=True))
            if before != after
        ]
        self.assertEqual(changed, [1, 48])
        self.assertEqual((allthree_argv[1], allthree_argv[48]), (binder.ALLTHREE_RUN, "ema_sparse"))
        for flag in ("--mean-variance-objective", "--mean-variance-head", "--loss", "--loss-norm"):
            index = nll_argv.index(flag)
            self.assertEqual(nll_argv[index : index + 2], allthree_argv[index : index + 2])
        self.assertEqual(nll_argv[nll_argv.index("--mean-variance-objective") + 1], "nll")

    def test_unbound_check_does_not_claim_actual_nll_terminal_receipts(self) -> None:
        result = binder._check_template()
        self.assertEqual(result["status"], "PASS_LOCAL_UNBOUND_ALLTHREE_TEMPLATE")
        self.assertFalse(result["actual_nll_terminal_receipts_read"])
        self.assertTrue(result["final_body_absent"])
        source = binder.TEMPLATE.read_text()
        for marker in (
            binder.NLL_ACCEPTANCE_UNBOUND,
            binder.NLL_STOP_UNBOUND,
            binder.NLL_ACCEPTANCE_B64_UNBOUND,
            binder.NLL_STOP_B64_UNBOUND,
        ):
            self.assertIn(marker, source)

    def test_launch_acceptance_fragment_rejects_missing_or_wrong_exact_pin(self) -> None:
        # These are isolated field-validator fragments, not fabricated receipts.
        with self.assertRaises(ValueError):
            binder._require_launch_acceptance_field({})
        with self.assertRaises(ValueError):
            binder._require_launch_acceptance_field({"launch_acceptance_sha256": "0" * 64})

    def test_receipt_validator_fails_closed_when_bytes_are_unavailable(self) -> None:
        with self.assertRaisesRegex(ValueError, "acceptance is absent"):
            binder._validate_nll_receipts(b"", b"")


if __name__ == "__main__":
    unittest.main()
