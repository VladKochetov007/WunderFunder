#!/usr/bin/env python3
"""Freeze one nested, component-complete E540 fitting subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from wnn.splits import load_campaign_split

ROOT = Path(__file__).resolve().parents[1]


def build_manifest(
    split_manifest: Path, fraction: float, seed: int, root: Path
) -> dict[str, object]:
    """Select whole eligible components without reading targets or evaluation labels."""
    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("fraction must be finite and in (0,1]")
    campaign = load_campaign_split(split_manifest, root)
    allowed = {
        "train": set(campaign.train_groups),
        "valid": set(campaign.valid_groups),
    }
    component_members: dict[int, list[tuple[str, int]]] = {}
    component_of: dict[tuple[str, int], int] = {}
    for split, group, component in campaign.components:
        member = (split, group)
        component_of[member] = component
        if group in allowed[split]:
            component_members.setdefault(component, []).append(member)
    expected_members = {(split, group) for split, groups in allowed.items() for group in groups}
    if set(component_of).intersection(expected_members) != expected_members:
        raise ValueError("campaign manifest omits a fitting group component identity")
    evaluation_components = {component_of[("valid", group)] for group in campaign.evaluation_groups}
    eligible_components = sorted(component_members)
    if set(eligible_components).intersection(evaluation_components):
        raise ValueError("campaign fitting and evaluation components overlap")
    count = max(1, math.ceil(len(eligible_components) * fraction))
    order = np.random.default_rng(seed).permutation(eligible_components)
    selected_components = sorted(int(value) for value in order[:count])
    selected = {
        split: sorted(
            group
            for component in selected_components
            for member_split, group in component_members[component]
            if member_split == split
        )
        for split in ("train", "valid")
    }
    if not selected["train"] or not selected["valid"]:
        raise ValueError("selected component subset must include train and valid fitting groups")
    selected_members = sorted(
        (split, group) for split, groups in selected.items() for group in groups
    )
    member_bytes = json.dumps(selected_members, separators=(",", ":"), ensure_ascii=True).encode(
        "ascii"
    )
    component_bytes = np.asarray(selected_components, dtype="<i8").tobytes()
    if not selected_members:
        raise ValueError("component selection produced no fitting groups")
    return {
        "version": 1,
        "experiment_id": "E540",
        "split_manifest_sha256": campaign.sha256,
        "selection_seed": seed,
        "target_component_fraction": fraction,
        "selection_rule": "seeded permutation of sorted eligible components; ceil fraction; nested by seed",
        "eligible_component_count": len(eligible_components),
        "selected_component_count": len(selected_components),
        "selected_components": selected_components,
        "selected_component_sha256": hashlib.sha256(component_bytes).hexdigest(),
        "eligible_group_counts": {key: len(value) for key, value in allowed.items()},
        "selected_groups": selected,
        "selected_group_counts": {key: len(value) for key, value in selected.items()},
        "selected_members_sha256": hashlib.sha256(member_bytes).hexdigest(),
        "evaluation_membership_sha256": hashlib.sha256(
            np.asarray(sorted(campaign.evaluation_groups), dtype="<i8").tobytes()
        ).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=540)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_manifest(args.split_manifest, args.fraction, args.seed, args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    with temporary.open("x", encoding="utf-8") as destination:
        json.dump(result, destination, sort_keys=True, indent=2, allow_nan=False)
        destination.write("\n")
        destination.flush()
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "selected_component_count",
                    "selected_group_counts",
                    "selected_members_sha256",
                )
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
