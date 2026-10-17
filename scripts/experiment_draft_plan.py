from __future__ import annotations

import argparse
import json
from pathlib import Path


def paired_cells() -> list[dict[str, int]]:
    """Return the frozen 30 cells, shared by every arm (not 30 IID holdouts)."""
    return [
        {"fold": fold, "seed": seed}
        for fold in range(4)
        for seed in range(1001, 1009 if fold < 2 else 1008)
    ]


def build_plan(recipe: dict[str, object]) -> dict[str, object]:
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
    if recipe.get("precision") != "fp32" or recipe.get("tf32") is not False:
        raise ValueError("this scientific campaign requires strict FP32, TF32 off")
    engineering = recipe.get("kind") == "engineering"
    cells = [] if engineering else paired_cells()
    return {
        "experiment_id": recipe["id"],
        "status": "plan_only_not_training",
        "launch_ready": recipe.get("launch_ready", False),
        "arms": arms,
        "cells_per_arm": len(cells),
        "total_fits": len(cells) * len(arms),
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
    args = parser.parse_args()
    if args.recipe.stat().st_size > 1_000_000:
        parser.error("recipe exceeds 1MB metadata limit")
    recipe = json.loads(args.recipe.read_text())
    print(json.dumps(build_plan(recipe), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
