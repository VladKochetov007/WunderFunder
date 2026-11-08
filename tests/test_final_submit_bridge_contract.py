from __future__ import annotations

from copy import deepcopy

import pytest

from scripts.tl3_chronological_recipes import W18_COEFFICIENTS
from wnn.training.final_submit_bridge_contract import (
    FINAL_W18_HOMOGENEOUS_FAMILY,
    FINAL_W18_TEACHERS,
    validate_exact_final_w18_teachers,
)


def _metadata() -> dict[str, object]:
    records = [
        {"run": run, "family": family, "coefficient": coefficient, "epoch": epoch}
        for run, family, coefficient, epoch in FINAL_W18_TEACHERS
    ]
    return {
        "source_coefficients": deepcopy(records),
        "sources": deepcopy(records),
        "homogeneous_family": FINAL_W18_HOMOGENEOUS_FAMILY,
        "coefficient_sum": 0.8125,
    }


def test_final_w18_contract_matches_the_frozen_recipe_mapping() -> None:
    expected = tuple(
        (f"tl3final_{run}", family, coefficient, epoch)
        for run, (family, coefficient, epoch) in W18_COEFFICIENTS.items()
    )
    assert expected == FINAL_W18_TEACHERS


def test_final_w18_contract_accepts_exact_six_ordered_teachers() -> None:
    validate_exact_final_w18_teachers(_metadata())


def test_final_w18_contract_rejects_wrong_named_teacher_with_correct_sum() -> None:
    metadata = _metadata()
    for records_name in ("source_coefficients", "sources"):
        records = metadata[records_name]
        assert isinstance(records, list)
        records[0]["run"] = "tl3final_fmu128_s3"
    with pytest.raises(ValueError, match="fbc1k_s5"):
        validate_exact_final_w18_teachers(metadata)


def test_final_w18_contract_rejects_coefficient_corruption_even_when_sum_is_correct() -> None:
    metadata = _metadata()
    for records_name in ("source_coefficients", "sources"):
        records = metadata[records_name]
        assert isinstance(records, list)
        records[0]["coefficient"] = 0.31
        records[1]["coefficient"] = 0.09
    with pytest.raises(ValueError, match="differs from tl3final_fbc1k_s5"):
        validate_exact_final_w18_teachers(metadata)


def test_final_w18_contract_requires_the_frozen_homogeneous_family() -> None:
    metadata = _metadata()
    metadata["homogeneous_family"] = "full_finetuned"
    with pytest.raises(ValueError, match="frozen six-teacher recipe"):
        validate_exact_final_w18_teachers(metadata)
