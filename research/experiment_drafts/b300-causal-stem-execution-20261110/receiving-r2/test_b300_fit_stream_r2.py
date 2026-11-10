"""CPU-only R2 scheduling and descendant-ownership fixtures."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

R1_RECEIVER = Path(__file__).resolve().parents[1] / "receiving"
sys.path.insert(0, str(R1_RECEIVER))

from b300_driver_core_r1 import FIT_STREAMS, ProcIdentity
from b300_fit_stream_r2 import _authorized_cell, _import_r1, _pair_stream
from b300_owned_runner_r2 import descendant_closure, live_owned


def _proc(
    pid: int, ppid: int, pgid: int, session: int, birth: int, rss: int
) -> ProcIdentity:
    """Create the minimal identity row used by Linux process ownership gates."""
    return ProcIdentity(pid, "S", ppid, pgid, session, birth, rss)


class FitStreamTests(unittest.TestCase):
    """Prove sequential per-pair allocation and exact descendant PID/birth closure."""

    def test_pair_stream_is_serial_and_stream_roster_is_disjoint(self) -> None:
        seen: list[str] = []

        def run_cell(cell: str) -> dict[str, str]:
            seen.append(cell)
            return {"cell": cell, "status": "fit_complete"}

        pair, cells = next(iter(FIT_STREAMS.items()))
        result = _pair_stream(pair, cells, run_cell)
        self.assertEqual(tuple(seen), cells)
        self.assertEqual(tuple(result), cells)
        flattened = [cell for stream in FIT_STREAMS.values() for cell in stream]
        self.assertEqual(len(flattened), 6)
        self.assertEqual(len(set(flattened)), 6)

    def test_each_cell_rechecks_admission_before_running(self) -> None:
        events: list[str] = []

        def validate(cell: str) -> dict[str, str]:
            events.append(f"admit:{cell}")
            return {"cell": cell}

        def fit(cell: str, admitted: dict[str, str]) -> dict[str, str]:
            self.assertEqual(admitted["cell"], cell)
            events.append(f"fit:{cell}")
            return {"cell": cell, "status": "fit_complete"}

        cells = FIT_STREAMS["0,1"]
        _pair_stream("0,1", cells, lambda cell: _authorized_cell(cell, validate, fit))
        self.assertEqual(
            events,
            [
                f"admit:{cells[0]}",
                f"fit:{cells[0]}",
                f"admit:{cells[1]}",
                f"fit:{cells[1]}",
            ],
        )

    def test_failed_admission_never_calls_fit(self) -> None:
        def reject(_cell: str) -> None:
            raise ValueError("admission SHA changed")

        def forbidden_fit(_cell: str, _admitted: object) -> dict[str, str]:
            self.fail("fit callback ran after failed admission")

        with self.assertRaisesRegex(ValueError, "admission SHA changed"):
            _authorized_cell("plain_s203201", reject, forbidden_fit)

    def test_r2_injects_descendant_runner_into_frozen_stage_helper(self) -> None:
        _core, _fits, _driver, support, guard_type, runner, _path = _import_r1(
            R1_RECEIVER
        )
        self.assertIs(support.run_owned, runner.run_owned)
        self.assertIs(guard_type, runner.ResourceGuard)

    def test_descendants_cross_sessions_but_foreign_process_is_excluded(self) -> None:
        leader = _proc(100, 1, 100, 100, 10, 100)
        rank = _proc(101, 100, 101, 101, 11, 200)
        rank_child = _proc(102, 101, 102, 102, 12, 300)
        foreign = _proc(200, 1, 200, 200, 20, 9_000)
        first_snapshot = {
            item.pid: item for item in (leader, rank, rank_child, foreign)
        }
        known = descendant_closure(first_snapshot, {leader.pid: leader.start_ticks})
        self.assertEqual(known, {100: 10, 101: 11, 102: 12})

        # The leader exited; ranks were reparented, but their recorded births remain owned.
        later_rank = _proc(101, 1, 101, 101, 11, 200)
        later_child = _proc(102, 101, 102, 102, 12, 300)
        later_snapshot = {101: later_rank, 102: later_child, 200: foreign}
        retained = descendant_closure(later_snapshot, known)
        self.assertEqual(retained, known)
        self.assertEqual(
            {
                (entry.pid, entry.start_ticks)
                for entry in live_owned(later_snapshot, retained)
            },
            {(101, 11), (102, 12)},
        )


if __name__ == "__main__":
    unittest.main()
