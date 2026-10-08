"""Source-qualified fold-membership checks for the H232 fixed-checkpoint export."""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Mapping, Sequence
from typing import Any


class MembershipError(ValueError):
    """Raised when realized source-qualified fit membership differs from fold 0."""


def _unique_ints(values: Sequence[Any], label: str) -> set[int]:
    """Validate an ordered row-group sequence and return its unique members."""
    if not isinstance(values, list):
        raise MembershipError(f"{label} must be a JSON array")
    if any(type(value) is not int or value < 0 for value in values):
        raise MembershipError(f"{label} contains an invalid row-group ID")
    if len(values) != len(set(values)):
        raise MembershipError(f"{label} contains duplicate row-group IDs")
    return set(values)


def _ordered_digest(values: Sequence[int]) -> str:
    """Return the V7 provenance's canonical signed-int64 little-endian digest."""
    digest = hashlib.sha256()
    for value in values:
        digest.update(struct.pack("<q", value))
    return digest.hexdigest()


def audit_fit_membership(
    provenance: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
    *,
    expected_split_sha256: str,
) -> dict[str, Any]:
    """Check exact fit/evaluation membership by ``(source, row_group)`` identity.

    Numeric row-group IDs are local to each parquet source. Equal integers from
    ``train`` and ``valid`` are therefore aliases, not shared sequences.
    """
    if provenance.get("split_manifest_sha256") != expected_split_sha256:
        raise MembershipError("provenance split fingerprint differs from fold manifest")
    fit = provenance.get("fit_row_groups")
    evaluation = provenance.get("evaluation_row_groups")
    manifest_fit = split_manifest.get("fit")
    manifest_eval = split_manifest.get("evaluation")
    if not all(isinstance(value, Mapping) for value in (fit, manifest_fit, manifest_eval)):
        raise MembershipError("fit/evaluation membership maps are missing")

    fitted: dict[str, list[int]] = {}
    expected: dict[str, set[int]] = {}
    for source in ("train", "valid"):
        actual_ids = fit.get(source)
        expected_ids = manifest_fit.get(source)
        if not isinstance(actual_ids, list) or not isinstance(expected_ids, list):
            raise MembershipError(f"{source} fit membership is missing")
        fitted[source] = actual_ids
        actual = _unique_ints(actual_ids, f"provenance.fit_row_groups.{source}")
        expected[source] = _unique_ints(expected_ids, f"fold.fit.{source}")
        if actual != expected[source]:
            raise MembershipError(f"{source} fit membership differs from frozen fold 0")

    if not isinstance(evaluation, list):
        raise MembershipError("provenance evaluation row groups must be a JSON array")
    actual_eval = _unique_ints(evaluation, "provenance.evaluation_row_groups")
    expected_eval_value = manifest_eval.get("valid")
    if not isinstance(expected_eval_value, list):
        raise MembershipError("fold evaluation.valid membership is missing")
    expected_eval = _unique_ints(expected_eval_value, "fold.evaluation.valid")
    if expected["valid"] & actual_eval:
        raise MembershipError("same-source valid fit/evaluation row-group overlap")
    if actual_eval != expected_eval:
        raise MembershipError("valid evaluation membership differs from frozen fold 0")

    membership_identity = provenance.get("fit_membership_identity")
    if not isinstance(membership_identity, Mapping):
        raise MembershipError("provenance fit-membership identity is missing")
    for source in ("train", "valid"):
        identity = membership_identity.get(source)
        if not isinstance(identity, Mapping):
            raise MembershipError(f"{source} membership identity is missing")
        if identity.get("count") != len(fitted[source]):
            raise MembershipError(f"{source} membership identity count differs")
        if identity.get("ordered_int64_le_sha256") != _ordered_digest(fitted[source]):
            raise MembershipError(f"{source} membership identity digest differs")

    nodes = split_manifest.get("sequences")
    if not isinstance(nodes, Mapping):
        raise MembershipError("fold manifest component identities are missing")
    component_by_key: dict[tuple[str, int], int] = {}
    for source in ("train", "valid"):
        source_nodes = nodes.get(source)
        if not isinstance(source_nodes, list):
            raise MembershipError(f"fold component records for {source} are missing")
        for node in source_nodes:
            if not isinstance(node, Mapping):
                raise MembershipError("fold component record is malformed")
            row_group = node.get("row_group")
            component = node.get("component")
            if (
                type(row_group) is not int
                or row_group < 0
                or type(component) is not int
                or component < 0
            ):
                raise MembershipError("fold component record has invalid IDs")
            key = (source, row_group)
            if key in component_by_key:
                raise MembershipError("fold has duplicate source-qualified component record")
            component_by_key[key] = component

    fit_nodes = {("train", group) for group in expected["train"]}
    fit_nodes.update(("valid", group) for group in expected["valid"])
    eval_nodes = {("valid", group) for group in expected_eval}
    missing = (fit_nodes | eval_nodes) - component_by_key.keys()
    if missing:
        raise MembershipError("fold component map omits a fitted or evaluated group")
    fit_components = {component_by_key[key] for key in fit_nodes}
    eval_components = {component_by_key[key] for key in eval_nodes}
    component_overlap = fit_components & eval_components
    if component_overlap:
        raise MembershipError("fit/evaluation feature-component overlap")

    # Report the legacy integer-only alias count for diagnostics, but never use
    # that cross-source intersection as a contamination test.
    fit_numeric = expected["train"] | expected["valid"]
    aliases = sorted(fit_numeric & expected_eval)
    return {
        "status": "passed",
        "split_manifest_sha256": expected_split_sha256,
        "fit_counts": {source: len(expected[source]) for source in ("train", "valid")},
        "evaluation_count": len(expected_eval),
        "source_qualified_fit_evaluation_group_overlap_count": 0,
        "fit_component_count": len(fit_components),
        "evaluation_component_count": len(eval_components),
        "fit_evaluation_component_overlap_count": 0,
        "cross_source_integer_alias_count": len(aliases),
        "cross_source_integer_aliases_are_not_overlap": True,
        "source_qualified_example": {
            "train_row_group": 4,
            "train_component": component_by_key.get(("train", 4)),
            "valid_row_group": 4,
            "valid_component": component_by_key.get(("valid", 4)),
        },
    }
