"""CPU-only contract checks for the isolated native-feature smoke runner."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SMOKE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SMOKE_DIR))

from smoke_native import options, set_option  # noqa: E402
from smoke_native_subset import _validate_subset  # noqa: E402
from smoke_native_validation import grader_call  # noqa: E402


def test_recipe_overrides_are_unique_and_preserve_other_flags() -> None:
    argv = ["--hidden", "512", "--precision", "fp32", "--no-eval-during-fit"]
    set_option(argv, "--hidden", "16")
    assert options(argv) == {"hidden": "16", "precision": "fp32", "no-eval-during-fit": True}
    with pytest.raises(ValueError, match="exactly one"):
        set_option(["--hidden", "16", "--hidden", "32"], "--hidden", "8")


def test_subset_requires_exact_fit_counts_and_full_evaluation_roster() -> None:
    manifest = {
        "selected_groups": {"train": [2, 4], "valid": [7]},
        "selected_group_counts": {"train": 2, "valid": 1},
        "selected_components": [2, 3],
        "selected_members_sha256": "a" * 64,
        "evaluation_groups": [10, 11, 12],
        "evaluation_group_count": 3,
    }
    assert _validate_subset(manifest, [10, 12, 11])["evaluation_group_count"] == 3
    manifest["evaluation_groups"] = [10]
    with pytest.raises(ValueError, match="full original fold"):
        _validate_subset(manifest, [10, 11, 12])


def test_official_smoke_grade_is_epoch_zero_without_a_reference(tmp_path: Path) -> None:
    class FakeGrader:
        def grade_endpoint(self, *args, **kwargs):
            args[4].write_text("{}", encoding="utf-8")
            self.call = (args, kwargs)
            return {
                "endpoint": {
                    "score": {"metric_id": "global_wp_clip2_v1", "selected_rows": 19, "wp": 0.125}
                }
            }

    grader = FakeGrader()
    output = tmp_path / "grade.json"
    checked = grader_call(grader, tmp_path, "probe", tmp_path / "split.json", output)
    assert grader.call[0][2] == 0
    assert grader.call[1]["reference"] is None
    assert checked["score"]["selected_rows"] == 19
    assert output.is_file()
