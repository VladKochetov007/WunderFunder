"""Synthetic selector, metadata, and canonical-import checks for R3 sampling."""

from __future__ import annotations

import importlib
import sys
import tempfile
import unittest
from pathlib import Path

R3_DIR = Path(__file__).resolve().parent
REPO_SRC = R3_DIR.parents[2] / "src"
sys.path.insert(0, str(REPO_SRC))
sys.path.insert(0, str(R3_DIR))

sample_panel = importlib.import_module("sample_panel")
panel_membership = importlib.import_module("panel_membership")
R1_SAMPLE_DATA = sample_panel.R1_SAMPLE_DATA
R3_DIR = panel_membership.R3_DIR
SPLIT_SHA = panel_membership.SPLIT_SHA
_r1_helpers = sample_panel._r1_helpers
select_membership = panel_membership.select_membership


def _split_fixture() -> dict[str, object]:
    train = [
        {"row_group": group, "seq_ix": 10_000 + group, "component": group // 2}
        for group in range(140)
    ]
    valid_fit = {"row_group": 1_000, "seq_ix": 11_000, "component": 0}
    evaluation = [
        {"row_group": group, "seq_ix": 20_000 + group, "component": 100 + (group - 2_000) // 2}
        for group in range(2_000, 2_080)
    ]
    return {
        "fit": {"train": list(range(140)), "valid": [1_000]},
        "evaluation": {"valid": list(range(2_000, 2_080))},
        "sequences": {"train": train, "valid": [valid_fit, *evaluation]},
    }


def _panel_metadata() -> dict[str, object]:
    fit_components = set(range(70))
    return {
        "split_sha256": SPLIT_SHA,
        "roles": {
            "fit": {
                "components": len(fit_components),
                "groups": 141,
                "train_components": 70,
                "valid_components": 1,
                "source_groups": {"train": 140, "valid": 1},
            },
            "eval": {
                "components": 40,
                "groups": 80,
                "train_components": 0,
                "valid_components": 40,
                "source_groups": {"valid": 80},
            },
        },
    }


class SamplePanelTest(unittest.TestCase):
    def test_selection_is_deterministic_nested_and_records_stage_probabilities(self) -> None:
        split = _split_fixture()
        panel = _panel_metadata()
        first = select_membership(split, panel)
        second = select_membership(split, panel)
        self.assertEqual(first, second)
        records = first["records"]
        fit = [item for item in records if item["role"] == "fit"]
        evaluation = [item for item in records if item["role"] == "eval"]
        self.assertEqual((len(fit), len(evaluation)), (64, 32))
        self.assertEqual(len({item["component"] for item in fit}), 64)
        self.assertEqual(len({item["component"] for item in evaluation}), 32)
        self.assertTrue(all(item["source"] == "train" for item in fit))
        self.assertTrue(all(item["source"] == "valid" for item in evaluation))
        self.assertEqual(
            [item["component"] for item in fit[:32]], first["selection"]["fit_low32_components"]
        )
        self.assertTrue(all(item["included_in_low32"] for item in fit[:32]))
        self.assertFalse(any(item["included_in_low32"] for item in fit[32:]))
        fit_probability = fit[0]["selection_probabilities"]
        self.assertAlmostEqual(fit_probability["fit_full64_component"], 64 / 70)
        self.assertAlmostEqual(fit_probability["fit_low32_component"], 32 / 70)
        self.assertAlmostEqual(fit_probability["group_given_component"], 0.5)
        self.assertAlmostEqual(fit_probability["fit_full64_sequence"], 64 / 70 / 2)
        self.assertAlmostEqual(fit_probability["fit_low32_sequence"], 32 / 70 / 2)
        self.assertAlmostEqual(
            evaluation[0]["selection_probabilities"]["eval32_component"], 32 / 40
        )

    def test_metadata_coverage_and_global_component_overlap_fail_before_selection(self) -> None:
        missing = _split_fixture()
        missing["sequences"]["train"].pop()
        with self.assertRaisesRegex(ValueError, "missing fit source/group metadata"):
            select_membership(missing)
        overlap = _split_fixture()
        overlap["sequences"]["valid"][1]["component"] = 0
        with self.assertRaisesRegex(ValueError, "full fitting/evaluation component overlap"):
            select_membership(overlap)

    def test_panel_preflight_metadata_must_match_membership(self) -> None:
        split = _split_fixture()
        panel = _panel_metadata()
        panel["roles"]["eval"]["source_groups"]["valid"] -= 1
        with self.assertRaisesRegex(
            ValueError, "panel metadata mismatch at evaluation.source_groups"
        ):
            select_membership(split, panel)

    def test_r1_import_closure_exposes_pinned_helpers_and_canonical_reader(self) -> None:
        helpers = _r1_helpers()
        self.assertEqual(Path(helpers.__file__).resolve(), R1_SAMPLE_DATA.resolve())
        self.assertEqual(helpers.SPLIT_SHA, SPLIT_SHA)
        self.assertEqual(helpers.SequenceFile.__module__, "wnn.data")
        reader_module = sys.modules[helpers.SequenceFile.__module__]
        self.assertEqual(
            Path(reader_module.__file__).resolve(),
            (R3_DIR.parents[2] / "src" / "wnn" / "data.py").resolve(),
        )
        self.assertTrue(
            all(
                callable(getattr(helpers, name))
                for name in ("digest", "write_json", "verify_sources")
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.json"
            helpers.write_json(path, {"seed": 19021})
            with self.assertRaises(FileExistsError):
                helpers.write_json(path, {"seed": 19021})


if __name__ == "__main__":
    unittest.main()
