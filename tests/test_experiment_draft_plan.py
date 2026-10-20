from __future__ import annotations

import pytest

from scripts.experiment_draft_plan import build_plan


def recipe() -> dict[str, object]:
    """Return a minimal source-unbound strict-FP32 scientific proposal."""
    return {
        "id": "screen",
        "arms": ["control", "candidate"],
        "precision": "fp32",
        "tf32": False,
        "autocast": False,
    }


def test_new_two_arm_screen_does_not_silently_double_the_fit_budget() -> None:
    plan = build_plan(recipe())
    assert plan["cells_per_arm"] == 2
    assert plan["total_fits"] == 4
    assert plan["launch_ready"] is False
    assert plan["comparison_stage"] == "screen"


def test_declared_historical_cp30_recipe_keeps_its_frozen_panel() -> None:
    plan = build_plan(
        {
            **recipe(),
            "schema": "wnn.experiment-draft.v1",
            "replications_per_arm": 30,
            "fold_seed_counts": [8, 8, 7, 7],
        },
        legacy_source_commit="a" * 40,
    )
    assert plan["total_fits"] == 60
    cells = plan["cells"]
    assert isinstance(cells, list)
    assert [sum(cell["fold"] == fold for cell in cells) for fold in range(4)] == [8, 8, 7, 7]
    assert plan["comparison_stage"] == "frozen_legacy_cp30"


def test_third_cell_is_six_fits_and_expansion_needs_a_followup_stage() -> None:
    assert build_plan({**recipe(), "replications_per_arm": 3})["total_fits"] == 6
    cells = [{"fold": fold, "seed": 1771} for fold in range(4)]
    with pytest.raises(ValueError, match="exceeds 6 total new fits"):
        build_plan({**recipe(), "cells": cells})
    plan = build_plan(
        {
            **recipe(),
            "cells": cells,
            "comparison_stage": "targeted_followup",
            "followup_allocation": {"reference": "dated-recipe-amendment", "new_fit_budget": 8},
        }
    )
    assert plan["total_fits"] == 8


def test_new_cp30_declaration_cannot_accidentally_bypass_the_screen_budget() -> None:
    with pytest.raises(ValueError, match="exceeds 6 total new fits"):
        build_plan({**recipe(), "schema": "wnn.experiment-draft.v2", "replications_per_arm": 30})


def test_reused_controls_need_provenance_and_reduce_new_fit_counts() -> None:
    reused = {"arm": "control", "fold": 0, "seed": 1001}
    with pytest.raises(ValueError, match="artifact_ref and ancestry_ref"):
        build_plan({**recipe(), "reused_controls": [reused]})
    reused.update(
        {"artifact_ref": "checkpoint-and-prediction-receipt", "ancestry_ref": "clean-fold"}
    )
    plan = build_plan({**recipe(), "reused_controls": [reused]})
    assert (plan["total_fits"], plan["new_fits"], plan["reused_fits"]) == (4, 3, 1)


def test_followup_requires_an_explicit_allocation_and_autocast_must_be_off() -> None:
    with pytest.raises(ValueError, match="dated allocation"):
        build_plan({**recipe(), "comparison_stage": "targeted_followup"})
    with pytest.raises(ValueError, match="autocast"):
        build_plan({**recipe(), "autocast": True})


@pytest.mark.parametrize(
    "cells",
    [
        [{"fold": 0, "seed": 7}, {"fold": 0, "seed": 7}],
        [{"fold": True, "seed": 7}],
        [{"fold": 4, "seed": 7}],
        [{"fold": 0, "seed": -1}],
    ],
)
def test_invalid_identities_cannot_count_as_new_repetitions(cells: list[dict[str, object]]) -> None:
    with pytest.raises(ValueError):
        build_plan({**recipe(), "cells": cells})
