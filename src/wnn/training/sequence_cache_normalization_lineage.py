"""Validation of fitting-only normalization membership and moment identities."""

from __future__ import annotations

from typing import Any

import numpy as np

from wnn.training.sequence_normalization import (
    build_normalization_identity,
    moments_sha256,
    normalization_identity_sha256,
)


def validate_normalization_ancestry(
    config: dict[str, Any],
    data: dict[str, Any],
    provenance_normalization: dict[str, Any],
    members: list[list[object]],
) -> None:
    """Recompute the declared normalizer membership and raw FP32 moments hash."""
    normalization = data.get("normalization")
    if not isinstance(normalization, dict):
        raise ValueError("teacher normalization membership is missing")
    scope = config.get("normalization_scope", "prefix")
    if scope == "all_fit":
        selected = members
    elif scope == "prefix":
        sequence_limit = config.get("normalization_sequences", 32)
        if type(sequence_limit) is not int or sequence_limit < 1:
            raise ValueError("teacher normalization prefix length is malformed")
        selected = members[:sequence_limit]
    else:
        raise ValueError("teacher normalization scope is unsupported")
    selected_members = [(str(member[0]), int(member[1])) for member in selected]
    source_fingerprints = normalization.get("source_fingerprints")
    if (
        normalization.get("scope") != scope
        or type(normalization.get("sequence_count")) is not int
        or normalization.get("sequence_count") != len(selected_members)
        or (
            scope == "prefix"
            and (
                type(normalization.get("fit_prefix_sequences")) is not int
                or normalization.get("fit_prefix_sequences") != len(selected_members)
            )
        )
        or not isinstance(source_fingerprints, dict)
    ):
        raise ValueError("teacher normalization scope or sequence count differs from its recipe")
    input_schema = normalization.get("input_schema", "raw112-v1")
    if input_schema != data.get("input_schema"):
        raise ValueError("teacher normalization input schema differs from its fit data")
    identity = build_normalization_identity(
        source_fingerprints,
        data["split_sha256"],
        selected_members,
        scope,
        112,
        input_schema=input_schema,
    )
    if normalization.get("identity_sha256") != normalization_identity_sha256(identity):
        raise ValueError("teacher normalization membership hash differs from fitting members")
    mean, scale = provenance_normalization.get("mean"), provenance_normalization.get("std")
    if not isinstance(mean, list) or not isinstance(scale, list):
        raise ValueError("teacher normalization moments are malformed")
    if normalization.get("moments_sha256") != moments_sha256(
        np.asarray(mean, dtype=np.float32), np.asarray(scale, dtype=np.float32)
    ):
        raise ValueError("teacher normalization moments hash differs from checkpoint values")
