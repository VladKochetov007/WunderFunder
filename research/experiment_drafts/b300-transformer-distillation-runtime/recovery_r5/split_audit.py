"""Expose actual component membership before the repeated 1.54B teacher fit."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from campaign_io import read_object, sha256_file
from split_inputs import validate_outer_inner_pair

# Split manifests carry source-qualified IDs and source fingerprints as JSON mappings.


def audit_splits(campaign: dict[str, Any]) -> dict[str, Any]:
    """Validate and count the exact train/development/outer memberships for both folds."""
    records = []
    for fold in (0, 2):
        outer_path = Path(campaign["outer_splits"][f"F{fold}"]["source_path"])
        inner_path = Path(campaign["inner_splits"][f"F{fold}"]["source_path"])
        outer, inner = read_object(outer_path), read_object(inner_path)
        validate_outer_inner_pair(outer_path, outer, inner_path, inner, fold)
        components = {
            (source, row["row_group"]): row["component"]
            for source, rows in inner["sequences"].items()
            for row in rows
        }
        train_members = [
            (source, group) for source in ("train", "valid") for group in inner["fit"][source]
        ]
        development = [("valid", group) for group in inner["evaluation"]["valid"]]
        held_out = [("valid", group) for group in outer["evaluation"]["valid"]]
        training_components = {components[member] for member in train_members}
        development_components = {components[member] for member in development}
        outer_components = {components[member] for member in held_out}
        overlaps = {
            "train_development": len(training_components & development_components),
            "train_outer": len(training_components & outer_components),
            "development_outer": len(development_components & outer_components),
        }
        if any(overlaps.values()):
            raise ValueError("a connected component crosses the train/validation boundary")
        records.append(
            {
                "fold": fold,
                "inner_split_sha256": sha256_file(inner_path),
                "outer_split_sha256": sha256_file(outer_path),
                "train_sequences_by_source": {
                    source: len(inner["fit"][source]) for source in ("train", "valid")
                },
                "train_components": len(training_components),
                "development_sequences": len(development),
                "development_components": len(development_components),
                "development_fraction_basis": "outer fitting valid-source sequences",
                "requested_development_fraction": inner["selection"]["evaluation_fraction"],
                "target_development_sequences": math.ceil(
                    inner["selection"]["evaluation_fraction"] * len(outer["fit"]["valid"])
                ),
                "actual_development_fraction": len(development) / len(outer["fit"]["valid"]),
                "excluded_training_source_sequences": len(outer["fit"]["train"])
                - len(inner["fit"]["train"]),
                "outer_sequences": len(held_out),
                "outer_components": len(outer_components),
                "overlapping_components": overlaps,
                "source_fingerprints": inner["sources"],
            }
        )
    return {
        "schema": "wnn.recovery-split-audit.v1",
        "folds": records,
        "normalization": "all fitting members only; excludes development and outer components",
        "selection": "inner development only; previously exposed outer folds are reporting only",
        "data_rows_recomputed": False,
    }
