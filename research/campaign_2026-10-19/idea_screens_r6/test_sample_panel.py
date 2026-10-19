from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
from sample_panel import select_eight_fit_records


def _fixture() -> tuple[dict[str, object], dict[str, object]]:
    """Make source-qualified metadata only; no NPY or real sequence data is touched."""
    fit_order = list(range(100, 164))
    eval_order = list(range(900, 932))
    records: list[dict[str, object]] = []
    for role, source, components in (("fit", "train", fit_order), ("eval", "valid", eval_order)):
        for order, component in enumerate(components):
            records.append(
                {
                    "role": role,
                    "source": source,
                    "group": order,
                    "row_group": order,
                    "seq_ix": order + 10_000,
                    "component": component,
                    "included_in_low32": role == "fit" and order < 32,
                    "fit_low32_order": order if role == "fit" and order < 32 else None,
                    "eligible_groups_in_component": 1,
                    "selection_probabilities": {"stage": 1.0},
                    "files": {"x": {}, "y": {}, "scored": {}},
                }
            )
    selection = {
        "fit_full64_component_order": fit_order,
        "eval32_component_order": eval_order,
    }
    counts = {"roles": {"fit": 64, "eval": 32}}
    sample = {
        "schema": "idea_screens_r3.sample.v2",
        "split_sha256": "c09f7240396a33ca82977115af8d6067e571c0ec25c58a3c99b438ca2d2294b7",
        "selection": selection,
        "metadata_counts": counts,
        "records": records,
    }
    identity = (
        "role",
        "source",
        "group",
        "row_group",
        "seq_ix",
        "component",
        "included_in_low32",
        "fit_low32_order",
        "eligible_groups_in_component",
        "selection_probabilities",
    )
    member_records = [{key: record[key] for key in identity} for record in records]
    membership = {
        "schema": "idea_screens_r3.membership.v2",
        "split_sha256": sample["split_sha256"],
        "selection": selection,
        "metadata_counts": counts,
        "records": member_records,
    }
    return sample, membership


def test_selector_returns_first_eight_fit_components_only() -> None:
    """Metadata selection preserves fixed order and never asks for evaluation arrays."""
    sample, membership = _fixture()
    selected = select_eight_fit_records(sample, membership)
    assert [record["component"] for record in selected] == list(range(100, 108))
    assert all(record["role"] == "fit" and record["source"] == "train" for record in selected)


def test_selector_rejects_component_overlap_and_duplicate_metadata() -> None:
    """The complete component disjointness check happens before any array load."""
    sample, membership = _fixture()
    bad_sample, bad_membership = copy.deepcopy(sample), copy.deepcopy(membership)
    bad_sample["records"][64]["component"] = 100
    bad_membership["records"][64]["component"] = 100
    bad_sample["selection"]["eval32_component_order"][0] = 100
    bad_membership["selection"]["eval32_component_order"][0] = 100
    with pytest.raises(ValueError, match="overlap"):
        select_eight_fit_records(bad_sample, bad_membership)

    bad_sample, bad_membership = copy.deepcopy(sample), copy.deepcopy(membership)
    bad_sample["records"][1]["component"] = bad_sample["records"][0]["component"]
    bad_membership["records"][1]["component"] = bad_membership["records"][0]["component"]
    bad_sample["selection"]["fit_full64_component_order"][1] = 100
    bad_membership["selection"]["fit_full64_component_order"][1] = 100
    with pytest.raises(ValueError, match="duplicates"):
        select_eight_fit_records(bad_sample, bad_membership)


def test_selector_rejects_membership_identity_mismatch() -> None:
    """A changed selected group cannot pass through matching component metadata."""
    sample, membership = _fixture()
    membership["records"][0]["group"] = 999
    with pytest.raises(ValueError, match="fixed membership"):
        select_eight_fit_records(sample, membership)


def test_actual_r3_sample_and_membership_metadata_acceptance() -> None:
    """The frozen producer-layout metadata selects its fixed fit order without reading arrays."""
    fixture_dir = Path(__file__).resolve().parent
    sample_bytes = (fixture_dir / "r3_sample_metadata_fixture.json").read_bytes()
    membership_bytes = (fixture_dir / "r3_membership_metadata_fixture.json").read_bytes()
    assert (
        hashlib.sha256(sample_bytes).hexdigest()
        == "b89455ba449f31bb9ebe815976d4da23bb1a8e499e421b6bb5f8d44566f76e64"
    )
    assert (
        hashlib.sha256(membership_bytes).hexdigest()
        == "e809d4810eeea7d2065ea8f2153b53ece3bde285f7900ce3d49e468e18045217"
    )
    selected = select_eight_fit_records(json.loads(sample_bytes), json.loads(membership_bytes))
    assert [record["component"] for record in selected] == [
        1655,
        861,
        1987,
        127,
        2008,
        2455,
        1876,
        2440,
    ]
    assert all(record["source"] == "train" and record["role"] == "fit" for record in selected)
    assert all(set(record["files"]) == {"x", "y", "scored"} for record in selected)
