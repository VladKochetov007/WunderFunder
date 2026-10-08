"""Adversarial mask and row-identity checks for the H233 scientific grader."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PATH = (
    ROOT
    / "research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/grade_h233_readouts.py"
)
SPEC = importlib.util.spec_from_file_location("h233_grade_under_test", PATH)
assert SPEC is not None and SPEC.loader is not None
GRADE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GRADE)


def mask_fixture() -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray]:
    """An excluded group between two fit groups, with one unscored tail row."""
    groups = [40, 7, 19]
    original = np.ones((3, 102), dtype=bool)
    original[1, 100] = False
    selected = np.zeros_like(original)
    selected[1, 99] = True
    selected[1, 101] = True
    return groups, original, selected, np.array([False, True, False])


def test_scientific_mask_excludes_fitting_groups_and_unscored_rows() -> None:
    groups, original, selected, group_mask = mask_fixture()
    GRADE.validate_masks(groups, [7], groups, original, selected, group_mask)


@pytest.mark.parametrize("row,column", [(0, 101), (1, 98), (1, 100), (2, 101)])
def test_mask_leakage_and_score_mask_changes_are_rejected(row: int, column: int) -> None:
    groups, original, selected, group_mask = mask_fixture()
    selected[row, column] = True
    with pytest.raises(ValueError, match="Scored mask differs"):
        GRADE.validate_masks(groups, [7], groups, original, selected, group_mask)


def test_omitted_evaluation_group_is_rejected() -> None:
    groups, original, selected, group_mask = mask_fixture()
    group_mask[1] = False
    with pytest.raises(ValueError, match="Evaluation-group mask"):
        GRADE.validate_masks(groups, [7], groups, original, selected, group_mask)


def test_full_export_alignment_uses_actual_positions() -> None:
    shared = GRADE.load_module("h233_shared_row_fixture", GRADE.SHARED)
    aligned = GRADE.full_alignment(shared, [7, 19], [40, 7, 19], [19, 40, 7], [7, 19, 40])
    assert aligned == [(("valid", 7), 1, 2, 0), (("valid", 19), 2, 0, 1)]


def test_duplicate_source_identity_is_rejected() -> None:
    shared = GRADE.load_module("h233_shared_duplicate_fixture", GRADE.SHARED)
    with pytest.raises(ValueError, match="duplicate"):
        GRADE.full_alignment(shared, [7], [7, 7], [7], [7])
