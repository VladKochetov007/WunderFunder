"""Validate the frozen H390 predictor's full-valid sidecar schema."""

from __future__ import annotations

import hashlib
from typing import cast

import numpy as np
from export_contract import _mapping
from pins import (
    CHECKPOINT_SHA256,
    CONTROL_RUN,
    OLD_SPLIT_SHA256,
)
from source_alignment import prediction_row_map


def validate_h390_sidecar(sidecar: dict[str, object], receipt: dict[str, object]) -> dict[int, int]:
    """Validate actual H390 fields, null evaluation claims and full-valid source order."""
    expected_fields = {
        "run",
        "split",
        "epoch",
        "checkpoint_sha256",
        "prediction_sha256",
        "split_manifest_sha256",
        "quote_cache",
        "gate_mode",
        "inference_precision",
        "quantization_preset",
        "recurrent_rank",
        "sentinel_preprocessing",
        "numeric_preprocessing",
        "row_groups",
        "evaluation_membership_sha256",
        "fit_subset_manifest_sha256",
        "campaign_split_manifest_sha256",
        "campaign_fit_scope",
        "campaign_evaluation_membership_sha256",
    }
    if set(sidecar) != expected_fields:
        raise ValueError("prediction sidecar field inventory differs from frozen H390 schema")
    expected = {
        "run": CONTROL_RUN,
        "split": "valid",
        "epoch": 5,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "split_manifest_sha256": OLD_SPLIT_SHA256,
        "campaign_split_manifest_sha256": OLD_SPLIT_SHA256,
        "inference_precision": "fp32",
        "campaign_fit_scope": "full-valid-feature-only",
        "evaluation_membership_sha256": None,
        "fit_subset_manifest_sha256": None,
        "campaign_evaluation_membership_sha256": None,
    }
    for key, value in expected.items():
        if sidecar.get(key) != value:
            raise ValueError(f"H390 sidecar binding mismatch: {key}")
    raw_groups = sidecar.get("row_groups")
    if not isinstance(raw_groups, list) or any(type(value) is not int for value in raw_groups):
        raise ValueError("H390 sidecar row_groups must be an integer list")
    row_groups = cast(list[int], raw_groups)
    row_map = prediction_row_map(row_groups)
    expected_ordered_digest = hashlib.sha256(
        np.asarray(row_groups, dtype="<i8").tobytes()
    ).hexdigest()
    ordered = _mapping(receipt.get("ordered_rows"), "receipt ordered rows")
    if (
        ordered.get("ordered_int64_le_sha256") != expected_ordered_digest
        or ordered.get("first") != row_groups[0]
        or ordered.get("last") != row_groups[-1]
    ):
        raise ValueError("sidecar row order differs from the captured full-valid order")
    sidecar_meta = _mapping(receipt.get("sidecar"), "receipt sidecar")
    if sidecar_meta.get("campaign_fit_scope") != "full-valid-feature-only" or sidecar_meta.get(
        "schema_fields"
    ) != sorted(expected_fields):
        raise ValueError("receipt sidecar schema record differs from actual H390 sidecar")
    return row_map
