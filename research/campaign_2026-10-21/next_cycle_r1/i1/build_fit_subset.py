"""Build this panel's 10% whole-component membership from a frozen split JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import cast

import numpy as np


def _int_list(value: object, *, label: str) -> list[int]:
    if not isinstance(value, list) or any(type(item) is not int for item in value):
        raise ValueError(f"{label} must be an integer list")
    return cast(list[int], value)


def build_manifest(split_path: Path, *, seed: int, experiment_id: str) -> dict[str, object]:
    """Sample sorted fitting components without opening any feature/target cache."""
    split_bytes = split_path.read_bytes()
    split = json.loads(split_bytes)
    if not isinstance(split, dict) or split.get("fold") != 2:
        raise ValueError("the i1 F2 membership builder requires the frozen fold-2 split")
    fit = split.get("fit")
    sequences = split.get("sequences")
    evaluation = split.get("evaluation")
    if not all(isinstance(value, dict) for value in (fit, sequences, evaluation)):
        raise ValueError("split JSON is missing fit, sequences or evaluation objects")

    fit_groups = {
        split_name: set(_int_list(fit.get(split_name), label=f"fit.{split_name}"))
        for split_name in ("train", "valid")
    }
    component_members: dict[int, dict[str, list[int]]] = {}
    component_for_valid: dict[int, int] = {}
    for split_name in ("train", "valid"):
        records = sequences.get(split_name)
        if not isinstance(records, list):
            raise ValueError(f"sequences.{split_name} must be a list")
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("sequence records must be objects")
            group = record.get("row_group")
            component = record.get("component")
            if type(group) is not int or type(component) is not int:
                raise ValueError("sequence records require integer row_group and component")
            if split_name == "valid":
                component_for_valid[group] = component
            if group in fit_groups[split_name]:
                component_members.setdefault(component, {"train": [], "valid": []})[
                    split_name
                ].append(group)

    eligible = sorted(component_members)
    if len(eligible) != 3145:
        raise ValueError(f"expected 3145 eligible fold-2 components, found {len(eligible)}")
    evaluation_groups = sorted(_int_list(evaluation.get("valid"), label="evaluation.valid"))
    evaluation_components = {component_for_valid[group] for group in evaluation_groups}
    if set(eligible).intersection(evaluation_components):
        raise ValueError("fitting and evaluation components overlap")
    selected_count = math.ceil(len(eligible) * 0.1)
    order = np.random.default_rng(seed).permutation(np.asarray(eligible, dtype=np.int64))
    selected_components = sorted(int(component) for component in order[:selected_count])
    selected_groups = {
        split_name: sorted(
            group
            for component in selected_components
            for group in component_members[component][split_name]
        )
        for split_name in ("train", "valid")
    }
    members = sorted(
        (split_name, group) for split_name, groups in selected_groups.items() for group in groups
    )
    member_bytes = json.dumps(members, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    component_bytes = np.asarray(selected_components, dtype="<i8").tobytes()
    evaluation_bytes = np.asarray(evaluation_groups, dtype="<i8").tobytes()
    return {
        "version": 1,
        "experiment_id": experiment_id,
        "split_manifest_sha256": hashlib.sha256(split_bytes).hexdigest(),
        "selection_seed": seed,
        "target_component_fraction": selected_count / len(eligible),
        "selection_rule": "seeded permutation of sorted eligible components; ceil fraction; nested by seed",
        "eligible_component_count": len(eligible),
        "selected_component_count": len(selected_components),
        "selected_components": selected_components,
        "selected_component_sha256": hashlib.sha256(component_bytes).hexdigest(),
        "eligible_group_counts": {name: len(groups) for name, groups in fit_groups.items()},
        "selected_groups": selected_groups,
        "selected_group_counts": {name: len(groups) for name, groups in selected_groups.items()},
        "selected_group_sha256": {
            name: hashlib.sha256(np.asarray(groups, dtype="<i8").tobytes()).hexdigest()
            for name, groups in selected_groups.items()
        },
        "selected_members_sha256": hashlib.sha256(member_bytes).hexdigest(),
        "evaluation_groups": evaluation_groups,
        "evaluation_group_count": len(evaluation_groups),
        "evaluation_membership_sha256": hashlib.sha256(evaluation_bytes).hexdigest(),
    }


def main() -> None:
    """Write one immutable F2 subset manifest and print its membership summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1002)
    parser.add_argument("--experiment-id", default="E-PLATEAU-I1-20261021-R1")
    args = parser.parse_args()
    manifest = build_manifest(args.split, seed=args.seed, experiment_id=args.experiment_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as output:
        json.dump(manifest, output, sort_keys=True, indent=2, allow_nan=False)
        output.write("\n")
        output.flush()
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "selected_component_count": manifest["selected_component_count"],
                "selected_group_counts": manifest["selected_group_counts"],
                "selected_members_sha256": manifest["selected_members_sha256"],
                "split_manifest_sha256": manifest["split_manifest_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
