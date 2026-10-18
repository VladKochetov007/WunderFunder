"""Tiny synthetic checks for bounded, fit-only mechanism screens."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from idea_screens_r1 import run
from idea_screens_r1.input_diagnostics import input_summary


def _sequence(role: str, component: str, n: int = 520) -> dict[str, object]:
    rows = np.arange(n, dtype=np.float32)
    x = np.zeros((n, 112), dtype=np.float32)
    x[:, :22] = rows[:, None] * 0.002 + np.arange(22, dtype=np.float32)[None, :] * 0.1
    x[:, 22:44] = 1.0
    x[:, 44:48] = rows[:, None] * 0.001 + np.arange(4, dtype=np.float32)[None, :] * 0.05
    x[:, 52:74] = rows[:, None] * 0.003 + np.arange(22, dtype=np.float32)[None, :] * 0.2
    x[:, 74:96] = 2.0
    y = np.column_stack((np.sin(rows / 11), np.cos(rows / 13))).astype(np.float32)
    return {
        "x": x,
        "y": y,
        "scored": np.ones(n, dtype=bool),
        "role": role,
        "source": "synthetic",
        "group": f"group-{component}",
        "component": component,
    }


class EdaMechanismsTest(unittest.TestCase):
    def test_warmup_summary_does_not_depend_on_future_suffix(self) -> None:
        x = _sequence("fit", "a")["x"]
        changed = x.copy()
        changed[99:] += 500
        before, after = input_summary(x), input_summary(changed)
        self.assertEqual(
            before["warmup_vs_later"]["i0_p"]["warmup"], after["warmup_vs_later"]["i0_p"]["warmup"]
        )

    def test_invalid_volume_rows_are_not_composition_eligible(self) -> None:
        sequence = _sequence("fit", "a", n=3)
        sequence["x"][1, 22] = 0
        sequence["x"][2, 74] = -1
        with tempfile.TemporaryDirectory() as directory:
            result = run([sequence], Path(directory) / "result.json")
        volume = result["per_sequence"][0]["input_only"]["volume_blocks"]
        self.assertEqual(volume["i0_v"]["strictly_positive_complete_rows"], 2)
        self.assertEqual(volume["i1_v"]["strictly_positive_complete_rows"], 2)

    def test_fit_labels_change_only_label_screen_not_input_transforms(self) -> None:
        first = _sequence("fit", "a")
        second = _sequence("fit", "a")
        second["y"] = np.zeros_like(second["y"])
        with tempfile.TemporaryDirectory() as directory:
            one = run([first], Path(directory) / "one.json")
            two = run([second], Path(directory) / "two.json")
        self.assertEqual(one["per_sequence"][0]["input_only"], two["per_sequence"][0]["input_only"])
        self.assertNotEqual(
            one["target_operator"]["pooled_fit_components"],
            two["target_operator"]["pooled_fit_components"],
        )

    def test_eval_labels_do_not_enter_fit_operator_screen(self) -> None:
        fit, evaluation = _sequence("fit", "a"), _sequence("eval", "eval")
        changed_eval = _sequence("eval", "eval")
        changed_eval["y"][:] = np.nan
        changed_eval["scored"][:] = False
        with tempfile.TemporaryDirectory() as directory:
            one = run([fit, evaluation], Path(directory) / "one.json")
            two = run([fit, changed_eval], Path(directory) / "two.json")
        self.assertEqual(one["target_operator"], two["target_operator"])
        self.assertIsNone(two["per_sequence"][1]["fit_label_rows"])

    def test_component_split_reports_exact_horizon_pair_counts_and_refuses_overwrite(self) -> None:
        first, second = _sequence("fit", "a"), _sequence("fit", "b")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            result = run([first, second], output)
            self.assertEqual(result["target_operator"]["candidate_count"], 10)
            self.assertEqual(len(result["target_operator"]["leave_one_component_out"]), 4)
            self.assertEqual(
                result["target_operator"]["pooled_fit_components"]["i0_p_mean_delta_h1"][
                    "target_0"
                ]["n_rows"],
                1038,
            )
            with self.assertRaises(FileExistsError):
                run([first], output)


if __name__ == "__main__":
    unittest.main()
