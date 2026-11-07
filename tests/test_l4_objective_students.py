from __future__ import annotations

import pytest

from scripts.l4_objective_students_20261104 import _profile_args, _valid_endpoint, choose_arm


def test_choose_arm_prefers_clip5_when_candidate_means_are_tied() -> None:
    assert choose_arm({"clip5": 0.701, "aux_late": 0.7010005}, 0.699) == "clip5"


def test_choose_arm_requires_point_one_percentage_point_over_control() -> None:
    assert choose_arm({"clip5": 0.701, "aux_late": 0.702}, 0.701) == "aux_late"
    assert choose_arm({"clip5": 0.7009, "aux_late": 0.7019}, 0.701) is None


def test_choose_arm_rejects_incomplete_or_nonfinite_means() -> None:
    with pytest.raises(ValueError, match="exactly the two frozen teacher arms"):
        choose_arm({"clip5": 0.7}, 0.69)
    with pytest.raises(ValueError, match="finite"):
        choose_arm({"clip5": 0.7, "aux_late": float("nan")}, 0.69)


def test_profile_command_keeps_recipe_and_drops_fit_state() -> None:
    args = ["--name", "fit", "--epochs", "2", "--save-training-state"]
    profile = _profile_args(args, "profile_r1")

    assert "--save-training-state" not in profile
    assert profile[:2] == ["--name", "profile_r1"]
    assert profile[-4:] == ["--profile-warmup", "65", "--profile-steps", "3"]
    assert args[-1] == "--save-training-state"


def test_official_valid_history_uses_metric_coverage_not_display_alias() -> None:
    actual_row = {
        "epoch": 3,
        "split": "vfold1873",
        "weighted_pearson": 0.68,
        "metric_id": "global_wp_clip2_v1",
        "blocks": 1873,
        "selected_rows": 3_528_273,
    }
    assert _valid_endpoint([actual_row], 1873, 3_528_273) == actual_row
    with pytest.raises(ValueError, match="exact official-valid manifest"):
        _valid_endpoint([{**actual_row, "selected_rows": 3_528_272}], 1873, 3_528_273)
