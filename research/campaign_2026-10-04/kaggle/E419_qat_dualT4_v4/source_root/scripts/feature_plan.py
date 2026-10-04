from __future__ import annotations

import argparse
import json
from pathlib import Path

from feature_recipes import descriptor, head_partition

from wnn.splits import load_campaign_split


def prepare(
    root: Path, split_path: Path, checkpoint: str, ancestors: list[str], output: Path
) -> None:
    """Freeze reciprocal head splits inside one backbone's canonical held-out components."""
    if output.exists():
        raise FileExistsError(output)
    split = load_campaign_split(split_path, root)
    components = {(source, group): component for source, group, component in split.components}
    partitions = [
        [
            group
            for group in split.evaluation_groups
            if head_partition(components[("valid", group)]) == side
        ]
        for side in (0, 1)
    ]
    if any(len({components[("valid", group)] for group in groups}) < 4 for groups in partitions):
        raise ValueError("Each head partition needs at least four held-out components")
    common = {
        "version": 1,
        "source": "valid",
        "campaign_split": descriptor(root, str(split_path.resolve().relative_to(root.resolve()))),
        "checkpoint": descriptor(root, checkpoint),
        "ancestors": [descriptor(root, name) for name in ancestors],
        "mask": descriptor(root, "datasets/valid_mask.parquet"),
    }
    output.mkdir(parents=True)
    for direction in (0, 1):
        for role, side in (("fit", direction), ("evaluation", 1 - direction)):
            value = {**common, "role": role, "row_groups": sorted(partitions[side])}
            (output / f"direction{direction}_{role}.json").write_text(
                json.dumps(value, indent=2) + "\n"
            )


def main() -> None:
    """Prepare immutable, input-only cache-selection manifests without extracting or fitting."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ancestor", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.root, args.split, args.checkpoint, args.ancestor, args.output)


if __name__ == "__main__":
    main()
