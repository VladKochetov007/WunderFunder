"""Smoke-test the score validator with the native compact CampaignSplit API."""

from __future__ import annotations

from pathlib import Path

from grade_tl3_roles_r2 import PRIMARY_SPLIT_SHA256, _validate_split

from wnn.splits import CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA, CampaignSplit


def test_native_campaign_split_primary_membership() -> None:
    """Build the exact B-source shape, which has no evaluation-source/mask fields."""
    train = tuple(range(10_325))
    evaluation = tuple(range(1_873))
    components = tuple(
        [("train", group, group) for group in train]
        + [("valid", group, 10_325 + group) for group in evaluation]
    )
    split = CampaignSplit(
        Path("chronological-primary.json"),
        PRIMARY_SPLIT_SHA256,
        0,
        1,
        train,
        (),
        evaluation,
        components,
        CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA,
    )

    assert not hasattr(split, "evaluation_source")
    assert not hasattr(split, "evaluation_mask")
    assert _validate_split(split) == list(evaluation)
