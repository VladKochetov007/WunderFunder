"""Validate PR81 profile objective summaries before canonical admission."""

from __future__ import annotations

import math
from collections.abc import Mapping


def _mapping(value: object, name: str) -> Mapping[str, object]:
    """Require an object at a metadata boundary."""
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _finite_number(value: object, name: str) -> float:
    """Return finite numeric evidence, excluding booleans."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite numeric evidence")
    return float(value)


def validate_profile_objective(value: dict[str, object], config: dict[str, object]) -> None:
    """Admit the signed NLL profile fields without rewriting its measured objective.

    Args:
        value: Parsed profile receipt that will be passed to canonical admission.
        config: Normalized training config bound by the receipt identity.

    Returns:
        None.

    Raises:
        ValueError: If the PR81 NLL objective summary is incomplete or inconsistent.
    """
    extension = _mapping(config.get("impact81"), "config.impact81")
    cell = _mapping(extension.get("cell"), "config.impact81.cell")
    variant = cell.get("variant")
    scope = value.get("losses_total_scope")

    if variant != "nll_transfer":
        if "losses_total_scope" in value:
            raise ValueError("signed-objective loss scope is reserved for nll_transfer")
        return

    if cell.get("card") != "09" or cell.get("arm") != "treatment":
        raise ValueError("nll_transfer profile must identify the card09 treatment")
    if scope != "primary_only_signed_objective_in_terms":
        raise ValueError("nll_transfer profile must declare its primary-only canonical loss scope")

    losses = _mapping(value.get("losses"), "profile.losses")
    objective_terms = _mapping(value.get("objective_terms"), "profile.objective_terms")
    primary = _finite_number(objective_terms.get("primary_loss"), "primary_loss")
    auxiliary = _finite_number(
        objective_terms.get("weighted_auxiliary_loss"), "weighted_auxiliary_loss"
    )
    actual_total = _finite_number(objective_terms.get("total_loss"), "signed total_loss")
    canonical_total = _finite_number(losses.get("total"), "losses.total")
    if primary < 0:
        raise ValueError("nll_transfer primary loss must remain nonnegative")
    if canonical_total != primary:
        raise ValueError("canonical losses.total must equal nll_transfer primary_loss")
    if not math.isclose(actual_total, primary + auxiliary, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("signed total_loss must equal primary_loss plus weighted auxiliary loss")
