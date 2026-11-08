"""Regression tests for the R4 role scorer's native split membership checks."""

from __future__ import annotations

from pathlib import Path

import pytest
from grade_tl3_roles_r4 import (
    PRIMARY_SPLIT_SHA256,
    ROLE_PINS,
    _canonical_sha256,
    _validate_prediction_row_order,
    _validate_run_provenance,
    _validate_split,
)

from wnn.splits import CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA, CampaignSplit


def _primary_split(train_groups: tuple[int, ...] | None = None) -> CampaignSplit:
    """Construct a source-qualified split with the real cardinalities and noncontiguity."""
    if train_groups is None:
        train_groups = tuple(group for group in range(10_607) if not 1_000 <= group < 1_282)
    evaluation_groups = tuple(range(1_873))
    components = tuple(
        [*(('train', group, group) for group in train_groups)]
        + [*(('valid', group, 20_000 + group) for group in evaluation_groups)]
    )
    return CampaignSplit(
        Path("chronological-primary.json"),
        PRIMARY_SPLIT_SHA256,
        0,
        1,
        train_groups,
        (),
        evaluation_groups,
        components,
        CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA,
    )


def _run_provenance(split: CampaignSplit) -> dict[str, object]:
    """Return the fixed train-only role metadata with exact manifest membership."""
    pin = ROLE_PINS[0]
    checkpoint_file = f"model_ep{pin.epoch}.pt"
    return {
        "split_manifest_sha256": split.sha256,
        "fit_row_groups": {"train": list(split.train_groups), "valid": []},
        "evaluation_row_groups": list(split.evaluation_groups),
        "args": {
            "name": pin.run,
            "checkpoint_policy": "fixed",
            "checkpoint_epoch": pin.epoch,
            "no_eval_during_fit": False,
            "train_source": "train",
            "split_manifest_sha256": split.sha256,
        },
        "checkpoint_sha256": {checkpoint_file: pin.checkpoint_sha256},
    }


def test_noncontiguous_train_membership_is_accepted_at_exact_count() -> None:
    """Accept the pinned count and actual manifest membership shape without inventing ranges."""
    split = _primary_split()

    assert len(split.train_groups) == 10_325
    assert split.train_groups[999] == 999
    assert split.train_groups[1_000] == 1_282
    assert split.train_groups[-1] == 10_606
    assert _validate_split(split) == list(range(1_873))


def test_duplicate_or_incomplete_source_membership_is_rejected() -> None:
    """Fail closed for duplicate fitting groups or incomplete valid components."""
    split = _primary_split()
    duplicate_membership = (*split.train_groups[:-1], split.train_groups[0])
    with pytest.raises(ValueError, match="frozen 10,325/1,873"):
        _validate_split(_primary_split(duplicate_membership))

    incomplete_components = tuple(
        component for component in split.components if component[:2] != ("valid", 1_872)
    )
    with pytest.raises(ValueError, match="frozen 10,325/1,873"):
        _validate_split(split._replace(components=incomplete_components))


def test_run_provenance_must_match_noncontiguous_manifest_exactly() -> None:
    """Keep exact role fit membership binding after relaxing the range assumption."""
    split = _primary_split()
    pin = ROLE_PINS[0]
    provenance = _run_provenance(split)
    _validate_run_provenance(provenance, split, pin, f"model_ep{pin.epoch}.pt")

    incorrect_membership = (*split.train_groups[:-1], 10_607)
    provenance = {
        **provenance,
        "fit_row_groups": {"train": list(incorrect_membership), "valid": []},
    }
    with pytest.raises(ValueError, match="exact split membership"):
        _validate_run_provenance(provenance, split, pin, f"model_ep{pin.epoch}.pt")


def test_seed_shuffled_provenance_order_is_recorded_but_prediction_order_is_fixed() -> None:
    """Permit shuffled provenance lists while keeping cache rows aligned to source order."""
    split = _primary_split()
    pin = ROLE_PINS[0]
    train_order = list(reversed(split.train_groups))
    evaluation_order = list(reversed(split.evaluation_groups))
    provenance = _run_provenance(split)
    provenance["fit_row_groups"] = {"train": train_order, "valid": []}
    provenance["evaluation_row_groups"] = evaluation_order

    order_hashes = _validate_run_provenance(
        provenance, split, pin, f"model_ep{pin.epoch}.pt"
    )

    assert order_hashes["fit_train"] == {
        "observed_order_sha256": _canonical_sha256(train_order),
        "manifest_order_sha256": _canonical_sha256(list(split.train_groups)),
    }
    assert order_hashes["evaluation"] == {
        "observed_order_sha256": _canonical_sha256(evaluation_order),
        "manifest_order_sha256": _canonical_sha256(list(split.evaluation_groups)),
    }
    assert (
        _validate_prediction_row_order(list(split.evaluation_groups), list(split.evaluation_groups))
        == _canonical_sha256(list(split.evaluation_groups))
    )
    with pytest.raises(ValueError, match="official source order"):
        _validate_prediction_row_order(evaluation_order, list(split.evaluation_groups))


@pytest.mark.parametrize("field", ["fit_train", "evaluation"])
@pytest.mark.parametrize("defect", ["missing", "extra", "duplicate"])
def test_provenance_member_set_errors_are_rejected(field: str, defect: str) -> None:
    """Reject incomplete, substituted, or duplicate provenance membership."""
    split = _primary_split()
    pin = ROLE_PINS[0]
    provenance = _run_provenance(split)
    expected = list(split.train_groups if field == "fit_train" else split.evaluation_groups)
    members = list(expected)
    if defect == "missing":
        members.pop()
    elif defect == "extra":
        members[-1] = 99_999
    else:
        members[-1] = members[0]
    if field == "fit_train":
        provenance["fit_row_groups"] = {"train": members, "valid": []}
        field_name = "fit_row_groups.train"
    else:
        provenance["evaluation_row_groups"] = members
        field_name = "evaluation_row_groups"

    with pytest.raises(ValueError, match=field_name):
        _validate_run_provenance(provenance, split, pin, f"model_ep{pin.epoch}.pt")
