"""Compare fixed-endpoint Signal Routes scores without selecting a winner."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from signal_routes.bindings import (
    RUNTIME_COMMIT,
    SEEDS,
    SPLIT_SHA256,
    VARIANTS,
    canonical,
    write_new,
)
from signal_routes.comparison_inputs import (
    METRIC_ID,
    SCORED_ROWS,
    VALIDATION_SEQUENCES,
    _admit_score,
)


def compare(paths: list[Path]) -> dict[str, Any]:
    """Validate the complete panel and report all same-seed treatment contrasts."""
    rows = [_admit_score(path) for path in paths]
    by_cell = {row["cell_id"]: row for row in rows}
    expected_cells = {f"sr_{variant}_s{seed}" for variant in VARIANTS for seed in SEEDS}
    if len(rows) != len(expected_cells) or set(by_cell) != expected_cells:
        raise ValueError("comparison requires exactly one score for each of the ten frozen cells")
    preprocessing_hashes = {row["preprocessing_sha256"] for row in rows}
    if len(preprocessing_hashes) != 1:
        raise ValueError("all cells must share the same train-only preprocessing artifact")
    data_identities = {canonical(row["identity"]["data"]) for row in rows}
    source_identities = {row["identity"]["source_sha256"] for row in rows}
    runtime_compatibilities = {canonical(row["runtime_compatibility"]) for row in rows}
    if (
        len(data_identities) != 1
        or len(source_identities) != 1
        or len(runtime_compatibilities) != 1
    ):
        raise ValueError("panel cells differ in source or fitting data identity")

    contrasts: dict[str, Any] = {}
    pairs = {
        **{f"{variant}_vs_plain": ("plain", variant) for variant in VARIANTS if variant != "plain"},
        "attention_differential_vs_attention_additive": (
            "attention_additive",
            "attention_differential",
        ),
        "ffn_conditional_vs_ffn_static": ("ffn_static", "ffn_conditional"),
    }
    for contrast_name, (control_variant, treatment_variant) in pairs.items():
        per_seed: dict[str, Any] = {}
        for seed in SEEDS:
            control = by_cell[f"sr_{control_variant}_s{seed}"]
            treatment = by_cell[f"sr_{treatment_variant}_s{seed}"]
            control_identity, treatment_identity = control["identity"], treatment["identity"]
            control_config = {
                key: value for key, value in control["config"].items() if key != "signal_routes"
            }
            treatment_config = {
                key: value for key, value in treatment["config"].items() if key != "signal_routes"
            }
            if canonical(control_config) != canonical(treatment_config):
                raise ValueError(f"matched pair changes native config: {contrast_name}/{seed}")
            if canonical(control_identity["data"]) != canonical(
                treatment_identity["data"]
            ) or canonical(control["runtime_compatibility"]) != canonical(
                treatment["runtime_compatibility"]
            ):
                raise ValueError(
                    f"matched pair differs in data or runtime class: {contrast_name}/{seed}"
                )
            deltas = {
                target: treatment["score"][target] - control["score"][target]
                for target in ("t0", "t1", "wp")
            }
            per_seed[str(seed)] = {
                "control_cell": control["cell_id"],
                "treatment_cell": treatment["cell_id"],
                "control_score": control["score"],
                "treatment_score": treatment["score"],
                "treatment_minus_control": deltas,
            }
        contrasts[contrast_name] = {
            "per_seed": per_seed,
            "mean_paired_wp_delta": sum(
                per_seed[str(seed)]["treatment_minus_control"]["wp"] for seed in SEEDS
            )
            / len(SEEDS),
        }

    return {
        "schema": "signal-routes.comparison.v1",
        "status": "complete_panel",
        "runtime_commit": RUNTIME_COMMIT,
        "split_sha256": SPLIT_SHA256,
        "metric_id": METRIC_ID,
        "selected_rows": SCORED_ROWS,
        "validation_sequences": VALIDATION_SEQUENCES,
        "preprocessing_sha256": next(iter(preprocessing_hashes)),
        "source_sha256": next(iter(source_identities)),
        "data_sha256": canonical(by_cell[next(iter(sorted(by_cell)))]["identity"]["data"]),
        "endpoint_admission": (
            "Predictions are admitted by signal_routes.endpoint.load_endpoint, which strict-loads "
            "the checkpoint and compares the full normalized native recipe; this comparator then "
            "checks the emitted E4 producer identity and source map."
        ),
        "cells": {
            cell_id: {
                "variant": row["variant"],
                "seed": row["seed"],
                "score": row["score"],
                "checkpoint_sha256": row["checkpoint_sha256"],
                "score_sha256": row["score_sha256"],
                "runtime_identity": row["identity"]["runtime"],
            }
            for cell_id, row in sorted(by_cell.items())
        },
        "paired_contrasts": contrasts,
        "interpretation": (
            "Descriptive same-seed contrasts on the repeatedly exposed official development fold; "
            "two seeds do not establish significance or public transfer. No winner is selected."
        ),
    }


def main() -> None:
    """Write one immutable comparison receipt for the complete frozen panel."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scores", type=Path, nargs=10, required=True)
    args = parser.parse_args()
    result = compare(args.scores)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_new(args.output, result)


if __name__ == "__main__":
    main()
