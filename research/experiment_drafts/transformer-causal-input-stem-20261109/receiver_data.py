"""Check the exact original source split before receiver model execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wnn.splits import CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA, load_campaign_split

SPLIT_SHA = "0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c"


def main() -> None:
    """Bind all canonical source/index identities and excluded fitting components."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    split = load_campaign_split(args.split_manifest, args.data_root)
    if (
        split.sha256 != SPLIT_SHA
        or split.schema != CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA
        or len(split.train_groups) != 10325
        or split.valid_groups
        or len(split.evaluation_groups) != 1873
    ):
        raise ValueError(
            "requires the unchanged PR49 train-only source split and source identities"
        )
    result = {
        "schema": "wnn.input-stem-data-check.v1",
        "split_sha256": split.sha256,
        "fit_train": 10325,
        "fit_valid": 0,
        "evaluation_valid": 1873,
        "component_disjoint": True,
        "evaluation_exposure": "development_exposed",
        "new_model_executions": 0,
    }
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
