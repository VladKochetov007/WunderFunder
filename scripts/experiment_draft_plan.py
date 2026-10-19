from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def paired_cells(recipe: dict[str, object]) -> list[dict[str, int]]:
    """Resolve an explicit screen panel, preserving declared historical CP30 recipes."""
    explicit = recipe.get("cells")
    count = recipe.get("replications_per_arm", 2)
    if type(count) is not int or count < 1:
        raise ValueError("replications_per_arm must be a positive integer")
    if explicit is None and count == 30:
        return [
            {"fold": fold, "seed": seed}
            for fold in range(4)
            for seed in range(1001, 1009 if fold < 2 else 1008)
        ]
    if explicit is None:
        if type(count) is not int or count not in (2, 3):
            raise ValueError("screens require 2 or 3 cells per arm, or an explicit cells list")
        return [
            {"fold": fold, "seed": seed} for fold, seed in ((0, 1001), (2, 1002), (3, 1001))[:count]
        ]
    if not isinstance(explicit, list) or not explicit:
        raise ValueError("cells must be a nonempty list")
    cells: list[dict[str, int]] = []
    for cell in explicit:
        if not isinstance(cell, dict) or set(cell) != {"fold", "seed"}:
            raise ValueError("each cell requires exactly fold and seed")
        fold, seed = cell["fold"], cell["seed"]
        if type(fold) is not int or fold not in range(4):
            raise ValueError("cell fold must be an integer in 0..3")
        if type(seed) is not int or seed < 0:
            raise ValueError("cell seed must be a nonnegative integer")
        cells.append({"fold": fold, "seed": seed})
    if len({(cell["fold"], cell["seed"]) for cell in cells}) != len(cells):
        raise ValueError("duplicate fold/seed cells are not replications")
    if "replications_per_arm" in recipe and count != len(cells):
        raise ValueError("replications_per_arm conflicts with explicit cells")
    return cells


def fit_counts(
    recipe: dict[str, object], arms: list[str], cells: list[dict[str, int]]
) -> dict[str, int]:
    """Count declared reused endpoints; references still require verification before launch."""
    reused = recipe.get("reused_controls", [])
    if not isinstance(reused, list):
        raise ValueError("reused_controls must be a list of source-qualified endpoint records")
    expected = {(arm, cell["fold"], cell["seed"]) for arm in arms for cell in cells}
    seen: set[tuple[str, int, int]] = set()
    for endpoint in reused:
        if not isinstance(endpoint, dict):
            raise ValueError(
                "each reused endpoint requires arm, fold, seed and evidence references"
            )
        arm, fold, seed = endpoint.get("arm"), endpoint.get("fold"), endpoint.get("seed")
        if not isinstance(arm, str) or type(fold) is not int or type(seed) is not int:
            raise ValueError("reused endpoint identities require string arm and integer fold/seed")
        identity = (arm, fold, seed)
        if identity not in expected or identity in seen:
            raise ValueError("reused endpoint must uniquely match a declared arm/cell")
        for key in ("artifact_ref", "ancestry_ref"):
            if not isinstance(endpoint.get(key), str) or not endpoint[key].strip():
                raise ValueError("reused endpoints require artifact_ref and ancestry_ref")
        seen.add(identity)
    return {"total_fits": len(expected), "reused_fits": len(seen), "new_fits": len(expected - seen)}


def build_plan(
    recipe: dict[str, object], *, legacy_source_commit: str | None = None
) -> dict[str, object]:
    """Expand a proposal without touching data, allocating devices or launching fits."""
    arms = recipe.get("arms")
    if (
        not isinstance(arms, list)
        or not arms
        or not all(isinstance(arm, str) and arm for arm in arms)
    ):
        raise ValueError("recipe requires a nonempty list of arm names")
    if len(set(arms)) != len(arms):
        raise ValueError("arm names must be unique")
    if (
        recipe.get("precision") != "fp32"
        or recipe.get("tf32") is not False
        or recipe.get("autocast") is not False
    ):
        raise ValueError("this scientific campaign requires strict FP32, autocast and TF32 off")
    engineering = recipe.get("kind") == "engineering"
    cells = [] if engineering else paired_cells(recipe)
    legacy = (
        not engineering
        and recipe.get("schema") == "wnn.experiment-draft.v1"
        and recipe.get("replications_per_arm") == 30
        and recipe.get("fold_seed_counts") == [8, 8, 7, 7]
        and "cells" not in recipe
    )
    if legacy_source_commit is not None and (
        not legacy or re.fullmatch(r"[0-9a-f]{40}", legacy_source_commit) is None
    ):
        raise ValueError("legacy override requires a v1 CP30 recipe and full source commit")
    legacy = legacy and legacy_source_commit is not None
    stage = recipe.get("comparison_stage", "screen")
    if stage not in ("screen", "targeted_followup"):
        raise ValueError("comparison_stage must be screen or targeted_followup")
    counts = fit_counts(recipe, arms, cells)
    if not legacy and stage == "screen" and counts["new_fits"] > 6:
        raise ValueError("initial screen exceeds 6 total new fits; declare a targeted_followup")
    if stage == "targeted_followup":
        allocation = recipe.get("followup_allocation")
        if not isinstance(allocation, dict) or not isinstance(allocation.get("reference"), str):
            raise ValueError(
                "targeted_followup requires a dated allocation reference and fit budget"
            )
        budget = allocation.get("new_fit_budget")
        if (
            not allocation["reference"].strip()
            or type(budget) is not int
            or budget < counts["new_fits"]
        ):
            raise ValueError("followup allocation must cover the planned new fits")
    return {
        "experiment_id": recipe["id"],
        "status": "plan_only_not_training",
        "launch_ready": recipe.get("launch_ready", False),
        "arms": arms,
        "cells_per_arm": len(cells),
        **counts,
        "comparison_stage": "frozen_legacy_cp30" if legacy else stage,
        "legacy_source_commit": legacy_source_commit,
        "provenance_refs_status": "caller-declared; verify artifacts, ancestry and allocation at admission",
        "reused_controls": recipe.get("reused_controls", []),
        "followup_allocation": recipe.get("followup_allocation"),
        "fit_count_scope": "listed arms only; budget separate teacher/student stages explicitly",
        "evidence_scope": "directional screen, not significance; all existing folds exposed",
        "cells": cells,
        "required_bindings_per_cell": [
            "source-qualified split hash and fitting membership",
            "fold-clean teacher, initializer, normalizer and propensity ancestry",
            "exact command and config hashes, checkpoint and resume policy",
            "GPU profile, environment, dataset/cache paths and output directory",
        ],
        "implementation_sources": recipe.get("implementation_sources", []),
        "remaining_work": recipe.get("remaining_work", []),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Expand a draft's paired run plan; no fits")
    parser.add_argument("recipe", type=Path)
    parser.add_argument(
        "--legacy-source-commit", help="explicit source pin for a historical v1 CP30 recipe"
    )
    args = parser.parse_args()
    if args.recipe.stat().st_size > 1_000_000:
        parser.error("recipe exceeds 1MB metadata limit")
    recipe = json.loads(args.recipe.read_text())
    print(
        json.dumps(
            build_plan(recipe, legacy_source_commit=args.legacy_source_commit),
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
