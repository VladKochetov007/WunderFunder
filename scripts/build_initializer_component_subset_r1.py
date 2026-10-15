"""Freeze a seed-3 subset of complete clean-fold fitting components."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any  # JSON has a heterogeneous nested schema.

import numpy as np

MAX_SPLIT_BYTES = 5_000_000
EXPECTED_SPLIT_SHA256 = "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d"
SUBSET_MIN_GROUPS = 256
SUBSET_MAX_GROUPS = 512
SUBSET_TARGET_GROUPS = 256
DEFAULT_EXPERIMENT_ID = "E-STATE-INIT-SUBSET-20261015-R1"


def sha256_bytes(value: bytes) -> str:
    """Hash an already bounded byte string."""
    return hashlib.sha256(value).hexdigest()


def int64_le_sha256(values: list[int]) -> str:
    """Hash signed little-endian group or component identifiers."""
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def canonical_pairs_sha256(values: list[list[str | int]]) -> str:
    """Hash canonical source-qualified membership pairs."""
    raw = json.dumps(values, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return sha256_bytes(raw)


def load_split_manifest(path: Path) -> tuple[dict[str, Any], str]:
    """Read and identify the pinned, size-bounded F0 split manifest."""
    size = path.stat().st_size
    if size <= 0 or size > MAX_SPLIT_BYTES:
        raise ValueError("campaign split manifest is outside the 5 MB read bound")
    raw = path.read_bytes()
    if len(raw) != size:
        raise ValueError("campaign split manifest changed while being read")
    digest = sha256_bytes(raw)
    if digest != EXPECTED_SPLIT_SHA256:
        raise ValueError("campaign split is not the frozen source-qualified F0 manifest")
    manifest = json.loads(raw)
    if not isinstance(manifest, dict) or manifest.get("fold") != 0:
        raise ValueError("expected the frozen F0 campaign split")
    if set(manifest.get("fit", {})) != {"train", "valid"}:
        raise ValueError("F0 split lacks its exact source-qualified fit groups")
    return manifest, digest


def build_subset(manifest: dict[str, Any], split_sha256: str, seed: int) -> dict[str, Any]:
    """Select complete fitting components until the subset reaches 256 sequences."""
    fit_groups = {source: set(map(int, manifest["fit"][source])) for source in ("train", "valid")}
    evaluation_groups = sorted(set(map(int, manifest["evaluation"]["valid"])))
    evaluation_group_set = set(evaluation_groups)
    evaluation_components = {
        int(item["component"])
        for item in manifest["sequences"]["valid"]
        if int(item["row_group"]) in evaluation_group_set
    }
    component_members: dict[int, dict[str, list[dict[str, int]]]] = {}
    seen: set[tuple[str, int]] = set()
    for source in ("train", "valid"):
        for item in manifest["sequences"][source]:
            group = int(item["row_group"])
            key = (source, group)
            if key in seen:
                raise ValueError(f"duplicate source-qualified sequence in split: {key}")
            seen.add(key)
            if group not in fit_groups[source]:
                continue
            component = int(item["component"])
            component_members.setdefault(component, {"train": [], "valid": []})[source].append(
                {"row_group": group, "seq_ix": int(item["seq_ix"])}
            )

    eligible_components = sorted(component_members)
    if len(eligible_components) != 3148:
        raise ValueError("eligible F0 fitting component count differs from the frozen split")
    component_order = np.random.default_rng(seed).permutation(eligible_components).tolist()
    selected_order: list[int] = []
    selected_count = 0
    selected_by_source = {"train": [], "valid": []}
    for component in component_order:
        selected_order.append(int(component))
        for source in ("train", "valid"):
            selected_by_source[source].extend(component_members[component][source])
        selected_count = len(selected_by_source["train"]) + len(selected_by_source["valid"])
        if selected_count >= SUBSET_TARGET_GROUPS and all(
            selected_by_source[source] for source in ("train", "valid")
        ):
            break
        if selected_count > SUBSET_MAX_GROUPS:
            raise ValueError("seeded component prefix overshot the 512-sequence subset ceiling")

    if not SUBSET_MIN_GROUPS <= selected_count <= SUBSET_MAX_GROUPS:
        raise ValueError("seeded whole-component selection did not reach the 256-sequence floor")

    selected_components = sorted(selected_order)
    if set(selected_components) & evaluation_components:
        raise ValueError("selected fit components overlap the fixed evaluation components")
    selected_groups = {
        source: sorted(item["row_group"] for item in selected_by_source[source])
        for source in ("train", "valid")
    }
    selected_members = sorted(
        [[source, group] for source, groups in selected_groups.items() for group in groups]
    )
    fit_order = [
        selected_members[index]
        for index in np.random.default_rng(seed).permutation(len(selected_members))
    ]
    evaluation_group_sha256 = int64_le_sha256(evaluation_groups)
    selected_identities = sorted(
        [
            [source, item["row_group"], item["seq_ix"], component]
            for component in selected_order
            for source in ("train", "valid")
            for item in component_members[component][source]
        ]
    )
    selected_identity_raw = json.dumps(
        selected_identities, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return {
        "format": "wnn.complete-component-fit-subset",
        "version": 1,
        "experiment_id": DEFAULT_EXPERIMENT_ID,
        "split_manifest_sha256": split_sha256,
        "selection": {
            "seed": seed,
            "rng": "numpy.random.default_rng(seed)",
            "algorithm": "permutation of sorted eligible component IDs; take the shortest prefix reaching 256 source-qualified fitting sequences, keeping both fit sources represented",
            "target_sequence_count": SUBSET_TARGET_GROUPS,
            "allowed_sequence_count": [SUBSET_MIN_GROUPS, SUBSET_MAX_GROUPS],
            "eligible_component_count": len(eligible_components),
            "selected_component_count": len(selected_components),
            "selected_components_in_draw_order": selected_order,
            "selected_components_sorted": selected_components,
            "selected_component_sha256": int64_le_sha256(selected_components),
            "selected_sequence_count": selected_count,
            "realized_sequence_fraction": selected_count
            / (len(fit_groups["train"]) + len(fit_groups["valid"])),
            "component_complete": True,
            "label_independent_selection": True,
        },
        "selected_groups": selected_groups,
        "selected_components": selected_components,
        "selected_component_count": len(selected_components),
        "selected_component_sha256": int64_le_sha256(selected_components),
        "selected_group_counts": {
            source: len(groups) for source, groups in selected_groups.items()
        },
        "selected_group_sha256": {
            source: int64_le_sha256(groups) for source, groups in selected_groups.items()
        },
        "selected_members_sha256": canonical_pairs_sha256(selected_members),
        "selected_source_group_seq_ix_component_sha256": sha256_bytes(selected_identity_raw),
        "selected_source_group_seq_ix_component": selected_identities,
        "fit_order": fit_order,
        "fit_order_sha256": canonical_pairs_sha256(fit_order),
        "evaluation_groups": evaluation_groups,
        "evaluation_group_count": len(evaluation_groups),
        "evaluation_membership_sha256": evaluation_group_sha256,
        "fit_evaluation_components_disjoint": not bool(
            set(selected_components) & evaluation_components
        ),
        "sequence_rows": 20_000,
        "purpose": "exploratory initializer-only subset screen; not a full-data or clean-holdout claim",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--experiment-id", default=DEFAULT_EXPERIMENT_ID)
    args = parser.parse_args()
    if args.seed != 3 or args.experiment_id != DEFAULT_EXPERIMENT_ID:
        parser.error("R1 freezes selection seed 3 and its declared experiment ID")
    if args.output.exists():
        raise FileExistsError(f"preserving existing subset manifest: {args.output}")
    split, split_sha256 = load_split_manifest(args.split_manifest)
    subset = build_subset(split, split_sha256, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(subset, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    with args.output.open("xb") as output_file:
        output_file.write(payload)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": sha256_bytes(payload),
                "selected_sequences": subset["selection"]["selected_sequence_count"],
                "selected_components": subset["selection"]["selected_component_count"],
                "fit_order_sha256": subset["fit_order_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
