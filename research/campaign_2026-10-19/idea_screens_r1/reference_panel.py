from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sample_data import BASE, digest, load_sample, write_json

from wnn.gmetric import GlobalAccumulator


def compare(sample: Path, output: Path) -> None:
    """Score existing FP32 F0 predictions on exactly the selected evaluation rows.

    This reads a compact already-produced prediction bundle. It does not restore
    weights, rerun inference or compare this subset with an unmatched full fold.
    """
    bundle = BASE / "submission-f0-r1/evidence/scored.npz"
    expected = "6277cca8d92f103da556929b0e88fc1c63c9eada61c5ae36d98af4a153d62d38"
    if digest(bundle) != expected:
        raise ValueError("existing prediction bundle identity differs")
    sequences, _ = load_sample(sample)
    accumulators = {name: GlobalAccumulator() for name in ("wide", "deep")}
    with np.load(bundle, allow_pickle=False) as archive:
        targets, groups, components = archive["y"], archive["row_group"], archive["component"]
        predictions = {name: archive[name] for name in accumulators}
    selected_groups = []
    for sequence in sequences:
        if sequence["role"] != "eval":
            continue
        rows = groups == sequence["group"]
        original = sequence["y"][sequence["scored"]]
        if not np.array_equal(original, targets[rows]):
            raise ValueError("reference and sample target order/mask differ")
        if not np.all(components[rows] == sequence["component"]):
            raise ValueError("reference component identity differs")
        for name, accumulator in accumulators.items():
            accumulator.add(
                targets[rows],
                predictions[name][rows],
                np.ones(int(rows.sum()), dtype=bool),
                group=sequence["component"],
            )
        selected_groups.append(sequence["group"])
    write_json(
        output,
        {
            "schema": "nf19.existing_reference_same_panel.v1",
            "bundle_sha256": expected,
            "groups": selected_groups,
            "models": {name: accumulator.result() for name, accumulator in accumulators.items()},
            "target_alignment": "bitwise equal original selected labels/order for every group",
            "limits": "descriptive exposed subset; full-data-fitted GRUs and small-panel ridge have different fit budgets; no full-fold ranking or public inference",
        },
    )


def main() -> None:
    """Produce one create-only matched-panel context record."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compare(args.sample, args.output)


if __name__ == "__main__":
    main()
