"""Small lookups over the immutable six-cell age-screen recipe."""

from __future__ import annotations

from typing import Any


def find_cell(recipe: dict[str, Any], fold: int, arm: str) -> dict[str, Any]:
    """Return the unique cell for one frozen fold and arm."""
    matches = [cell for cell in recipe["cells"] if int(cell["fold"]) == fold and cell["arm"] == arm]
    if len(matches) != 1:
        raise ValueError(f"recipe must contain exactly one F{fold} arm {arm} cell")
    return matches[0]
