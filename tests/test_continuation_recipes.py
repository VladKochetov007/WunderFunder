from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def recipes():
    path = Path(__file__).resolve().parents[1] / "scripts/continuation_recipes.py"
    spec = importlib.util.spec_from_file_location("continuation_recipes", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def flag_value(args: list[str], flag: str) -> str:
    index = len(args) - 1 - args[::-1].index(flag)
    return args[index + 1]


def test_head_is_single_parent_fixed_historical_finetune() -> None:
    item = recipes().recipe("combo_head", 2, 3, "local")
    args = item["argv"]
    assert flag_value(args, "--init") == "c20261003_combo_s3_f2@ep3"
    assert flag_value(args, "--optim") == "adamw"
    assert flag_value(args, "--chunk") == "2000"
    assert flag_value(args, "--freeze") == "inp,gru"
    assert flag_value(args, "--checkpoint-epoch") == "0"
    assert "--distill" not in args
    assert "--inherit-init-preprocessing" in args
    assert "--no-eval-during-fit" in args


def test_final_recipe_preserves_population_and_ancestors() -> None:
    item = recipes().recipe("combo", None, 3, "local")
    args = item["argv"]
    assert flag_value(args, "--distill") == "ab67_final_since_s3,ab67_final_since_s4"
    assert flag_value(args, "--propensity") == "prop_v1"
    assert flag_value(args, "--eval-source") == "holdout"
    assert "--include-heldout-valid" not in args
    assert "--no-eval-during-fit" not in args
    assert item["limits"]["gpu_stop_bytes"] < 5_000_000_000


def test_deployed_contrast_is_equal_single_trunk_head_mixture() -> None:
    screen = recipes().merged_screen()
    for members in screen["arms"].values():
        assert len(members) == 2
        assert list(members.values()) == [0.5, 0.5]
        assert all("since" not in member for member in members)
    assert screen["comparisons"] == [["COMBO", "CTRL"]]


def test_invalid_fold_cannot_generate_training_recipe() -> None:
    with pytest.raises(ValueError, match="fold"):
        recipes().recipe("combo", 4, 3, "local")
