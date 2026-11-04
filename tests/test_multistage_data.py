from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from wnn.multistage_data import _selected_membership
from wnn.splits import CampaignSplit


def _primary_split(*, linked_valid_component: bool = False) -> CampaignSplit:
    """Build a small source-qualified primary split with unequal components."""
    components = (
        ("train", 0, 10),
        ("train", 1, 10),
        ("train", 2, 11),
        ("train", 3, 11),
        ("train", 4, 12),
        ("train", 5, 13),
        ("valid", 0, 14),
    )
    evaluation_groups = (0,)
    if linked_valid_component:
        components += (("valid", 2, 10),)
        evaluation_groups = (0, 2)
    return CampaignSplit(
        Path("primary.json"),
        "a" * 64,
        0,
        1,
        (0, 1, 2, 3, 4, 5),
        (),
        evaluation_groups,
        components,
        "wnn.chronological-primary.v1",
    )


def test_auxiliary_membership_is_deterministic_and_component_complete() -> None:
    split = _primary_split()
    fitting, control = _selected_membership(split, 42, 0.25)

    assert (fitting, control) == _selected_membership(split, 42, 0.25)
    assert set(fitting).isdisjoint(control)
    assert set(fitting).union(control) == set(split.train_groups)
    component_of = {(source, group): component for source, group, component in split.components}
    held_components = {component_of[("train", group)] for group in control}
    fit_components = {component_of[("train", group)] for group in fitting}
    assert held_components.isdisjoint(fit_components)


def test_auxiliary_membership_rejects_components_linked_to_official_valid() -> None:
    split = _primary_split(linked_valid_component=True)
    with pytest.raises(ValueError, match="validation-linked component"):
        _selected_membership(split, 42, 0.25)


@pytest.mark.parametrize("fraction", [0.0, 1.0, True, float("nan")])
def test_auxiliary_membership_rejects_invalid_holdout_fraction(fraction: object) -> None:
    with pytest.raises(ValueError, match="evaluation fraction"):
        _selected_membership(_primary_split(), 42, cast(float, fraction))
