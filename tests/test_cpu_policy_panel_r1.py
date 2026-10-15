"""Focused tests for the portable xteam_h4 CPU-panel design."""

from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "benchmark_cpu_policy_panel_r1.py"
SPEC = importlib.util.spec_from_file_location("cpu_policy_panel_r1_tested", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot import the panel harness for unit checks")
PANEL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PANEL
SPEC.loader.exec_module(PANEL)


class PanelProtocolTests(unittest.TestCase):
    """Check block pairing, budget admission math, and reported variance levels."""

    def test_block_contains_two_bracketed_controls_and_each_candidate(self) -> None:
        candidate_orders: dict[int, set[tuple[str, ...]]] = {slot: set() for slot in range(8)}
        for block in range(12):
            schedule = PANEL.block_schedule(83, block)
            self.assertEqual(len(schedule), 8 * 5)
            for slot in range(8):
                samples = [item for item in schedule if item["slot"] == slot]
                self.assertEqual([item["label"] for item in samples].count("A"), 2)
                self.assertEqual(
                    {item["label"] for item in samples if item["label"] != "A"},
                    {"B", "C", "D"},
                )
                candidate_orders[slot].add(
                    tuple(item["label"] for item in samples if item["label"] != "A")
                )
                self.assertEqual(
                    [item["control"] for item in samples if item["label"] == "A"],
                    ["first", "last"],
                )
        self.assertEqual(PANEL.block_schedule(83, 3), PANEL.block_schedule(83, 3))
        self.assertEqual({len(orders) for orders in candidate_orders.values()}, {6})

    def test_pilot_never_admits_fewer_than_six_blocks(self) -> None:
        pilot = [2.0] * 5
        self.assertEqual(PANEL.planned_blocks(pilot, 100.0, 0.0), 0)
        self.assertEqual(PANEL.planned_blocks(pilot, 650.0, 0.0), 6)
        self.assertEqual(PANEL.planned_blocks(pilot, 1_000.0, 0.0), 9)

    def test_paired_log_interval_recovers_constant_ratio(self) -> None:
        ratio = 0.75
        receipt = PANEL.paired_log_ci([math.log(ratio)] * 6)
        self.assertAlmostEqual(receipt["geometric_mean_ratio"], ratio)
        self.assertAlmostEqual(receipt["ci95_halfwidth_pct"], 0.0)

    def test_summary_keeps_repeat_and_sequence_variance_separate(self) -> None:
        records: list[dict[str, object]] = []
        for block in range(6):
            for slot in range(8):
                base = 1.0 + slot * 0.01 + block * 0.001
                for label, factor in (("A", 1.0), ("B", 0.75), ("C", 0.70), ("D", 0.60)):
                    repeats = 2 if label == "A" else 1
                    for _ in range(repeats):
                        records.append(
                            {
                                "block": block,
                                "slot": slot,
                                "label": label,
                                "callback_wall_seconds": base * factor,
                                "callback_process_seconds": base * factor * 0.9,
                            }
                        )
        result = PANEL._compute_results(records, 6, 12)
        self.assertEqual(result["completed_blocks"], 6)
        self.assertEqual(set(result["within_sequence_repeat_variation"]), {"B", "C", "D"})
        self.assertEqual(set(result["between_sequence_mean_wall_variation"]), {"A", "B", "C", "D"})
        b_over_a = result["pairwise_block_ci95"]["B_over_A"]
        self.assertAlmostEqual(b_over_a["wall"]["geometric_mean_ratio"], 0.75)
        self.assertAlmostEqual(b_over_a["process"]["geometric_mean_ratio"], 0.75)


if __name__ == "__main__":
    unittest.main()
