"""Synthetic native-schema and isolated-import checks for the R3 v2 sampler."""

from __future__ import annotations

import copy
import importlib
import sys
import tempfile
import unittest
from pathlib import Path

R3_DIR = Path(__file__).resolve().parent


def _canonical_src() -> Path:
    """Locate canonical source in either the repo or staged support layout."""
    for ancestor in (R3_DIR, *R3_DIR.parents):
        candidates = [ancestor / "src", *ancestor.glob("source-*/src")]
        for candidate in candidates:
            if (candidate / "wnn" / "data.py").is_file():
                return candidate.resolve()
    raise FileNotFoundError("canonical src/wnn/data.py not found")


CANONICAL_SRC = _canonical_src()
sys.path.insert(0, str(CANONICAL_SRC))
sys.path.insert(0, str(R3_DIR))

sample_panel_v2 = importlib.import_module("sample_panel_v2")
panel_membership_v2 = importlib.import_module("panel_membership_v2")
R1_SAMPLE_DATA = sample_panel_v2.R1_SAMPLE_DATA
SPLIT_SHA = panel_membership_v2.SPLIT_SHA
_r1_helpers = sample_panel_v2._r1_helpers
select_membership = panel_membership_v2.select_membership


def _split_fixture() -> dict[str, object]:
    """Build canonical-style fit/eval plus explicit train and valid exclusions."""
    train = [
        {"row_group": group, "seq_ix": 10_000 + group, "component": group // 2}
        for group in range(140)
    ]
    train.extend(
        [
            {"row_group": 140, "seq_ix": 10_140, "component": 2_000},
            {"row_group": 141, "seq_ix": 10_141, "component": 2_001},
        ]
    )
    valid_fit = {"row_group": 1_000, "seq_ix": 11_000, "component": 0}
    evaluation = [
        {"row_group": group, "seq_ix": 20_000 + group, "component": 2_000 + (group - 2_000) // 2}
        for group in range(2_000, 2_080)
    ]
    eval_groups = list(range(2_000, 2_080))
    return {
        "fit": {"train": list(range(140)), "valid": [1_000]},
        "evaluation": {"valid": eval_groups},
        "excluded": {"train": [140, 141], "valid": eval_groups.copy(), "corrupt_train": [528]},
        "sequences": {"train": train, "valid": [valid_fit, *evaluation]},
    }


def _panel_metadata() -> dict[str, object]:
    """Return synthetic metadata-only panel counts matching the fixture."""
    return {
        "split_sha256": SPLIT_SHA,
        "roles": {
            "fit": {
                "components": 70,
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


class SamplePanelV2Test(unittest.TestCase):
    """Guard canonical exclusions, fixed selection, and source imports."""

    def test_builder_style_exclusions_and_valid_alias_are_accepted(self) -> None:
        split = _split_fixture()
        membership = select_membership(split, _panel_metadata())
        selected = membership["records"]
        fit = [item for item in selected if item["role"] == "fit"]
        evaluation = [item for item in selected if item["role"] == "eval"]
        coverage = membership["metadata_counts"]["coverage_checks"]
        self.assertEqual((len(fit), len(evaluation)), (64, 32))
        self.assertTrue({item["group"] for item in fit}.issubset(set(split["fit"]["train"])))
        self.assertTrue(
            {item["group"] for item in evaluation}.issubset(set(split["evaluation"]["valid"]))
        )
        self.assertFalse({item["group"] for item in fit} & set(split["excluded"]["train"]))
        self.assertTrue(coverage["train_partition_complete"])
        self.assertEqual(coverage["excluded_train_groups"], 2)
        self.assertTrue(coverage["excluded_valid_is_eval_alias"])
        self.assertEqual(coverage["corrupt_train_markers_absent_from_train_metadata"], [528])

    def test_membership_is_deterministic_nested_and_probabilities_unchanged(self) -> None:
        split = _split_fixture()
        one = select_membership(split)
        two = select_membership(split)
        self.assertEqual(one, two)
        self.assertEqual(one["schema"], "idea_screens_r3.membership.v2")
        fit = [item for item in one["records"] if item["role"] == "fit"]
        evaluation = [item for item in one["records"] if item["role"] == "eval"]
        self.assertEqual(
            [item["component"] for item in fit[:32]], one["selection"]["fit_low32_components"]
        )
        self.assertTrue(all(item["included_in_low32"] for item in fit[:32]))
        self.assertFalse(any(item["included_in_low32"] for item in fit[32:]))
        self.assertAlmostEqual(fit[0]["selection_probabilities"]["fit_full64_component"], 64 / 70)
        self.assertAlmostEqual(fit[0]["selection_probabilities"]["fit_low32_component"], 32 / 70)
        self.assertAlmostEqual(fit[0]["selection_probabilities"]["group_given_component"], 0.5)
        self.assertAlmostEqual(
            evaluation[0]["selection_probabilities"]["eval32_component"], 32 / 40
        )

    def test_corrupt_marker_absent_from_metadata_is_recorded_not_promoted_to_exclusion(
        self,
    ) -> None:
        split = _split_fixture()
        membership = select_membership(split)
        coverage = membership["metadata_counts"]["coverage_checks"]
        self.assertEqual(split["excluded"]["corrupt_train"], [528])
        self.assertEqual(coverage["corrupt_train_markers_fit_eligible"], [])
        self.assertEqual(coverage["corrupt_train_markers_absent_from_train_metadata"], [528])
        self.assertEqual(coverage["fit_train_groups"], 140)

    def test_corrupt_marker_cannot_appear_anywhere_in_fit_eligible_metadata(self) -> None:
        split = _split_fixture()
        baseline = select_membership(split)
        selected = {item["group"] for item in baseline["records"] if item["role"] == "fit"}
        unselected_fit_group = min(set(split["fit"]["train"]) - selected)
        marked = copy.deepcopy(split)
        marked["excluded"]["corrupt_train"] = [unselected_fit_group]
        with self.assertRaisesRegex(ValueError, "corrupt train markers are fit-eligible"):
            select_membership(marked)

    def test_inconsistent_missing_duplicate_and_unknown_relations_are_rejected(self) -> None:
        missing_train_exclusion = _split_fixture()
        missing_train_exclusion["excluded"]["train"].pop()
        with self.assertRaisesRegex(ValueError, "partition train metadata"):
            select_membership(missing_train_exclusion)

        duplicate_exclusion = _split_fixture()
        duplicate_exclusion["excluded"]["train"].append(140)
        with self.assertRaisesRegex(ValueError, "duplicate excluded.train"):
            select_membership(duplicate_exclusion)

        unknown_fit = _split_fixture()
        unknown_fit["fit"]["train"].append(9_999)
        with self.assertRaisesRegex(ValueError, "unknown fit.train"):
            select_membership(unknown_fit)

        fit_excluded_overlap = _split_fixture()
        fit_excluded_overlap["fit"]["train"].append(140)
        with self.assertRaisesRegex(ValueError, "partition train metadata"):
            select_membership(fit_excluded_overlap)

        missing_fit_metadata = _split_fixture()
        missing_fit_metadata["sequences"]["valid"].pop(0)
        with self.assertRaisesRegex(ValueError, "unknown fit.valid"):
            select_membership(missing_fit_metadata)

        duplicate_metadata = _split_fixture()
        duplicate_metadata["sequences"]["train"].append(
            duplicate_metadata["sequences"]["train"][0].copy()
        )
        with self.assertRaisesRegex(ValueError, "duplicate sequence metadata"):
            select_membership(duplicate_metadata)

        bad_alias = _split_fixture()
        bad_alias["excluded"]["valid"].pop()
        with self.assertRaisesRegex(
            ValueError, "excluded.valid must be the evaluation.valid alias"
        ):
            select_membership(bad_alias)

        wrong_excluded_component = _split_fixture()
        wrong_excluded_component["sequences"]["train"][-1]["component"] = 0
        with self.assertRaisesRegex(ValueError, "excluded.train does not match"):
            select_membership(wrong_excluded_component)

    def test_panel_preflight_count_mismatch_fails(self) -> None:
        split = _split_fixture()
        panel = _panel_metadata()
        panel["roles"]["fit"]["groups"] -= 1
        with self.assertRaisesRegex(ValueError, "panel metadata mismatch at fit.groups"):
            select_membership(split, panel)

    def test_r1_import_closure_pins_v2_and_canonical_reader(self) -> None:
        helpers = _r1_helpers()
        self.assertEqual(Path(helpers.__file__).resolve(), R1_SAMPLE_DATA.resolve())
        self.assertEqual(helpers.SPLIT_SHA, SPLIT_SHA)
        reader_module = sys.modules[helpers.SequenceFile.__module__]
        self.assertEqual(
            Path(reader_module.__file__).resolve(),
            (CANONICAL_SRC / "wnn" / "data.py").resolve(),
        )
        self.assertTrue(callable(helpers.verify_sources))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "create-only.json"
            helpers.write_json(path, {"schema": "v2-test"})
            with self.assertRaises(FileExistsError):
                helpers.write_json(path, {"schema": "v2-test"})


if __name__ == "__main__":
    unittest.main()
