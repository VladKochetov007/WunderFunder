#!/usr/bin/env python3
"""Freeze a seeded, complete-component initializer fit subset from fold-0 JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import struct
from collections import defaultdict
from pathlib import Path
from typing import Any

DEFAULT_SPLIT = Path("research/campaign_2026-09-29/artifacts/splits/fold0.json")
DEFAULT_OUTPUT = Path(
    "research/campaign_2026-10-15/initializer_family_f0_r1/recipes/FIT_SUBSET_R1.json"
)
SELECTION_SEED = 20261015
TARGET_FIT_SEQUENCES = 100


def digest_ints(values: list[int]) -> str:
    """Hash canonical little-endian int64 IDs."""
    digest = hashlib.sha256()
    for value in values:
        digest.update(struct.pack("<q", value))
    return digest.hexdigest()


def build_subset(split_path: Path, output_path: Path) -> dict[str, Any]:
    """Select whole fit-only components without reading sequence payloads."""
    raw = split_path.read_bytes()
    split = json.loads(raw)
    if split.get("version") != 1 or split.get("fold") != 0:
        raise ValueError("expected the frozen version-1 fold-0 split")

    fit_members = {
        (source, int(row_group))
        for source in ("train", "valid")
        for row_group in split["fit"][source]
    }
    evaluation_members = {("valid", int(row_group)) for row_group in split["evaluation"]["valid"]}
    excluded_members = {
        (source, int(row_group))
        for source in ("train", "valid")
        for row_group in split["excluded"].get(source, [])
    }
    rows_by_component: dict[int, list[dict[str, int | str]]] = defaultdict(list)
    category_by_component: dict[int, set[str]] = defaultdict(set)
    for source in ("train", "valid"):
        for row in split["sequences"][source]:
            row_group = int(row["row_group"])
            component = int(row["component"])
            member = (source, row_group)
            category = (
                "fit"
                if member in fit_members
                else "evaluation"
                if member in evaluation_members
                else "excluded"
                if member in excluded_members
                else "unassigned"
            )
            rows_by_component[component].append(
                {
                    "source": source,
                    "row_group": row_group,
                    "seq_ix": int(row["seq_ix"]),
                    "component": component,
                }
            )
            category_by_component[component].add(category)

    fit_components = sorted(
        component for component, categories in category_by_component.items() if "fit" in categories
    )
    rng = random.Random(SELECTION_SEED)
    selection_order = fit_components.copy()
    rng.shuffle(selection_order)

    selected_order: list[int] = []
    selected_count = 0
    for component in selection_order:
        categories = category_by_component[component]
        if categories != {"fit"}:
            continue
        selected_order.append(component)
        selected_count += len(rows_by_component[component])
        if selected_count >= TARGET_FIT_SEQUENCES:
            break
    if selected_count < TARGET_FIT_SEQUENCES:
        raise ValueError("not enough complete fit-only components")

    selected_set = set(selected_order)
    selected_rows = [
        row
        for component in selected_order
        for row in sorted(
            rows_by_component[component],
            key=lambda member: (0 if member["source"] == "train" else 1, member["row_group"]),
        )
    ]
    selected_members = sorted((str(row["source"]), int(row["row_group"])) for row in selected_rows)
    if len(selected_rows) != len(selected_members) or any(
        (str(row["source"]), int(row["row_group"])) not in fit_members for row in selected_rows
    ):
        raise ValueError("selected complete-component rows are not unique fitting members")
    for component in selected_set:
        component_rows = rows_by_component[component]
        if len(component_rows) != sum(
            1 for row in selected_rows if int(row["component"]) == component
        ):
            raise ValueError("selection omitted a train/valid node from a selected component")

    selected_groups = {
        source: sorted(int(row["row_group"]) for row in selected_rows if row["source"] == source)
        for source in ("train", "valid")
    }
    evaluation_groups = sorted(int(group) for group in split["evaluation"]["valid"])
    selected_components = sorted(selected_set)
    component_digest = hashlib.sha256(
        b"".join(struct.pack("<q", value) for value in selected_components)
    ).hexdigest()
    manifest: dict[str, Any] = {
        "format": "wnn.initializer-family-f0-fit-subset",
        "version": 1,
        "experiment_id": "initializer-family-f0-r1",
        "split_manifest_path": str(split_path),
        "split_manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "selection": {
            "algorithm": "shuffle sorted fit-only component IDs, include whole components until threshold",
            "seed": SELECTION_SEED,
            "target_fit_sequence_count": TARGET_FIT_SEQUENCES,
            "selected_fit_sequence_count": len(selected_rows),
            "selection_order": selected_order,
            "component_completeness": "all train and valid source nodes in each selected component",
            "labels_or_targets_used": False,
        },
        "selected_components": selected_components,
        "selected_component_sha256": component_digest,
        "selected_groups": selected_groups,
        "selected_source_group_counts": {
            source: len(selected_groups[source]) for source in ("train", "valid")
        },
        "selected_members_sha256": hashlib.sha256(
            json.dumps(selected_members, separators=(",", ":"), ensure_ascii=True).encode("ascii")
        ).hexdigest(),
        "selected_members": selected_rows,
        "evaluation_groups": evaluation_groups,
        "evaluation_group_count": len(evaluation_groups),
        "evaluation_membership_sha256": digest_ints(evaluation_groups),
        "component_count": len(selected_components),
        "fit_component_count_in_split": len(fit_components),
        "source_category_audit": {
            "selected_component_categories": {
                str(component): sorted(category_by_component[component])
                for component in selected_components
            },
            "selected_nonfit_members": 0,
            "selected_omitted_component_members": 0,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build_subset(args.split, args.out)
    print(
        json.dumps(
            {
                "subset": str(args.out),
                "components": result["component_count"],
                "members": result["selection"]["selected_fit_sequence_count"],
                "source_counts": result["selected_source_group_counts"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
