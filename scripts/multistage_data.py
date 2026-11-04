"""Prepare the frozen component-disjoint auxiliary control manifest."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from wnn.multistage_data import prepare_multistage_auxiliary_split
from wnn.splits import CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA, file_sha256, load_campaign_split


def main() -> None:
    """Create and summarize a train-only auxiliary control membership."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--primary-split", type=Path, required=True)
    parser.add_argument("--portable-registry", type=Path, required=True)
    parser.add_argument("--expected-primary-sha256", required=True)
    parser.add_argument("--expected-registry-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261031)
    parser.add_argument("--evaluation-fraction", type=float, default=0.2)
    args = parser.parse_args()
    primary_sha256 = file_sha256(args.primary_split)
    registry_sha256 = file_sha256(args.portable_registry)
    if primary_sha256 != args.expected_primary_sha256:
        parser.error("primary split SHA256 differs from the frozen receiver identity")
    if registry_sha256 != args.expected_registry_sha256:
        parser.error("portable registry SHA256 differs from the frozen receiver identity")
    registry = json.loads(args.portable_registry.read_text())
    if registry.get("split_manifest_sha256") != primary_sha256:
        parser.error("portable registry is not bound to the frozen primary split")
    os.environ["WNN_PORTABLE_DATA_MANIFEST"] = str(args.portable_registry.resolve())
    primary = load_campaign_split(args.primary_split, args.data_root)
    if (
        primary.schema != CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA
        or len(primary.train_groups) != 10325
        or primary.valid_groups
        or len(primary.evaluation_groups) != 1873
    ):
        parser.error("receiver primary manifest has unexpected train/valid membership")
    split = prepare_multistage_auxiliary_split(
        args.primary_split,
        args.data_root,
        args.output,
        seed=args.seed,
        evaluation_fraction=args.evaluation_fraction,
    )
    print(
        json.dumps(
            {
                "manifest": str(split.path),
                "manifest_sha256": split.sha256,
                "primary_split_sha256": primary_sha256,
                "portable_registry_sha256": registry_sha256,
                "primary_fit_sequences": len(split.train_groups) + len(split.evaluation_groups),
                "auxiliary_fit_sequences": len(split.train_groups),
                "auxiliary_control_sequences": len(split.evaluation_groups),
                "evaluation_role": split.evaluation_role,
                "evaluation_source": split.evaluation_source,
                "evaluation_mask": split.evaluation_mask,
                "chronology_claim": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
