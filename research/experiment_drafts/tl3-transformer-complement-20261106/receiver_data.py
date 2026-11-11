"""Audit the exact reused train-only source split on the receiving host."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from wnn.splits import CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA, load_campaign_split
from wnn.training.sequence_reference_pins import PRIMARY_SPLIT_SHA256


def main() -> None:
    """Validate full source identities and component exclusion without a model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    split = load_campaign_split(args.split_manifest, args.data_root)
    if split.sha256 != PRIMARY_SPLIT_SHA256:
        raise ValueError("the PR49 manifest byte identity changed; do not rebuild the split")
    if split.schema != CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA or split.valid_groups:
        raise ValueError("this panel requires the original PR49 train-only source split")
    if len(split.train_groups) != 10325 or len(split.evaluation_groups) != 1873:
        raise ValueError("source membership differs from the pinned clean TL3 comparison")
    # The pinned legacy exporter reads only this index member, not cached labels.
    with np.load(args.data_root / "cache/train_targets.npz", allow_pickle=False) as targets:
        groups = np.asarray(targets["row_group"])
    with np.load(args.data_root / "cache/train_index.npz", allow_pickle=False) as index:
        expected = np.asarray(index["row_group"])
    if groups.ndim != 1 or groups.dtype.kind not in "iu" or not np.array_equal(groups, expected):
        raise ValueError("legacy train target index must retain the verified full source order")
    result = {
        "schema": "wnn.tl3-complement-data-check.v1",
        "split_sha256": split.sha256,
        "fit_train": len(split.train_groups),
        "fit_valid": 0,
        "evaluation_valid": len(split.evaluation_groups),
        "component_disjoint": True,
        "legacy_train_export_index_checked": True,
        "evaluation_exposure": "development_exposed",
        "chronology": "official source boundary; calendar timestamps unavailable",
        "new_model_executions": 0,
    }
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
