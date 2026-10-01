from __future__ import annotations

import unittest

from book_replay import DepthReplay, keyed_edit, price_key, probe_perturbations
from encoding_study import publication_signature


class ReconstructionChecks(unittest.TestCase):
    """Independent toy oracle checks for the reconstruction contract."""

    def setUp(self) -> None:
        self.replay = DepthReplay()
        self.snapshot = {
            "lastUpdateId": 10,
            "bids": [["100", "2"], ["99", "3"]],
            "asks": [["101", "4"], ["102", "5"]],
        }

    def test_missing_snapshot_is_not_reconstructed(self) -> None:
        update = {"U": 11, "u": 11, "b": [["100", "9"]], "a": []}
        self.assertEqual(self.replay.update(update, 90), "invalid")
        self.assertFalse(self.replay.bids)

    def test_continuity_delete_and_independent_oracle(self) -> None:
        self.replay.snapshot(self.snapshot, 100)
        update = {"U": 10, "u": 12, "b": [["100", "0"], ["99", "7"]], "a": [["101", "6"]]}
        self.assertEqual(self.replay.update(update, 110), "applied")
        self.assertEqual(self.replay.bids, {99 * 10**8: 7.0})
        self.assertEqual(self.replay.asks, {101 * 10**8: 6.0, 102 * 10**8: 5.0})
        self.assertEqual(self.replay.update_id, 12)

    def test_gap_remains_invalid_until_snapshot(self) -> None:
        self.replay.snapshot(self.snapshot, 100)
        self.assertEqual(self.replay.update({"U": 12, "u": 13, "b": [], "a": []}, 110), "gap")
        self.assertEqual(self.replay.update({"U": 14, "u": 14, "b": [], "a": []}, 120), "invalid")
        self.replay.snapshot(self.snapshot | {"lastUpdateId": 14}, 130)
        self.assertEqual(self.replay.update({"U": 15, "u": 15, "b": [], "a": []}, 140), "applied")

    def test_snapshot_cannot_backfill_receive_time(self) -> None:
        self.replay.snapshot(self.snapshot, 100)
        with self.assertRaises(ValueError):
            self.replay.update({"U": 11, "u": 11, "b": [], "a": []}, 99)

    def test_covered_updates_and_boundary_uncertainty(self) -> None:
        self.replay.snapshot(self.snapshot, 100)
        self.assertEqual(self.replay.update({"U": 9, "u": 10, "b": [], "a": []}, 110), "covered")
        self.assertFalse(self.replay.has_known_top(2))
        update = {"U": 11, "u": 11, "b": [["99", "0"], ["98", "8"]], "a": []}
        self.replay.update(update, 120)
        self.assertFalse(self.replay.has_known_top(2))

    def test_price_identity_and_edit_conservation(self) -> None:
        self.assertEqual(price_key("0.10000000"), price_key("0.1"))
        edits = keyed_edit({100: 2.0, 99: 4.0}, {100: 5.0, 98: 6.0})
        self.assertEqual(edits["interior_add"], 3.0)
        self.assertEqual(edits["interior_remove"], 0.0)
        self.assertEqual(
            edits["entered_mass"]
            - edits["exited_mass"]
            + edits["interior_add"]
            - edits["interior_remove"],
            5.0,
        )
        probes = probe_perturbations()
        self.assertLess(max(probes["geometry_max_errors"].values()), 1e-12)
        self.assertGreater(probes["rank_shift"]["rank_l1"], 0)
        self.assertEqual(probes["rank_shift"]["interior_add"], 0)
        self.assertEqual(probes["rank_shift"]["interior_remove"], 0)

    def test_publication_modes_retain_and_destroy_different_information(self) -> None:
        events = [
            (0, (100.0, 2.0, 101.0, 4.0)),
            (1, (101.0, 1.0, 1.0)),
            (2, ()),
            (0, (100.0, 3.0, 101.0, 4.0)),
            (2, ()),
        ]
        held = publication_signature(events, transient_trade=False)
        transient = publication_signature(events, transient_trade=True)
        coalesced = publication_signature(events, transient_trade=False, stride=5)
        delayed = publication_signature(events, transient_trade=False, delay_trade=5)
        self.assertEqual(held["signatures"]["001"], 1)
        self.assertEqual(transient["signatures"]["001"], 2)
        self.assertEqual(coalesced["emitted_rows"], 1)
        self.assertEqual(delayed["delayed_trades_pending_at_end"], 1)


if __name__ == "__main__":
    unittest.main()
