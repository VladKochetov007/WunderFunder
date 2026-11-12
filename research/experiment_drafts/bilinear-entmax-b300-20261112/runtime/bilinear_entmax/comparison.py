"""Compare one frozen Bilinear Entmax study without selecting a winner."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from bilinear_entmax.bindings import (
    RUNTIME_COMMIT,
    SEEDS,
    SPLIT_SHA256,
    STUDIES,
    canonical,
    write_new,
)
from bilinear_entmax.comparison_inputs import (
    METRIC_ID,
    SCORED_ROWS,
    VALIDATION_SEQUENCES,
    _admit_score,
)


def compare(study_id: str, paths: list[Path]) -> dict[str, Any]:
    """Validate four fresh study scores and report the two same-seed contrasts."""
    if study_id not in STUDIES:
        raise ValueError(f"unknown study: {study_id}")
    study = STUDIES[study_id]
    control_variant = study["control_variant"]
    treatment_variant = study["treatment_variant"]
    expected_cells = {
        f"be_{variant}_s{seed}"
        for variant in (control_variant, treatment_variant)
        for seed in SEEDS
    }
    rows = [_admit_score(path) for path in paths]
    by_cell = {row["cell_id"]: row for row in rows}
    if len(rows) != 4 or set(by_cell) != expected_cells:
        raise ValueError(f"{study_id} comparison requires its four unique new fits")
    preprocessing_hashes = {row["preprocessing_sha256"] for row in rows}
    if len(preprocessing_hashes) != 1:
        raise ValueError("study cells must share the same train-only preprocessing artifact")
    data_identities = {canonical(row["identity"]["data"]) for row in rows}
    source_identities = {row["identity"]["source_sha256"] for row in rows}
    runtime_compatibilities = {canonical(row["runtime_compatibility"]) for row in rows}
    if (
        len(data_identities) != 1
        or len(source_identities) != 1
        or len(runtime_compatibilities) != 1
    ):
        raise ValueError("study cells differ in source or fitting data identity")
    checkpoint_hashes = {row["checkpoint_sha256"] for row in rows}
    profile_hashes = {row["identity"]["bilinear_entmax_profile_receipt_sha256"] for row in rows}
    if len(checkpoint_hashes) != 4 or len(profile_hashes) != 4:
        raise ValueError("study scores must bind four distinct endpoint and profile receipts")

    per_seed: dict[str, Any] = {}
    for seed in SEEDS:
        control = by_cell[f"be_{control_variant}_s{seed}"]
        treatment = by_cell[f"be_{treatment_variant}_s{seed}"]
        control_identity, treatment_identity = control["identity"], treatment["identity"]
        control_config = {
            key: value for key, value in control["config"].items() if key != "bilinear_entmax"
        }
        treatment_config = {
            key: value for key, value in treatment["config"].items() if key != "bilinear_entmax"
        }
        if canonical(control_config) != canonical(treatment_config):
            raise ValueError(f"matched pair changes native config: {study_id}/{seed}")
        if canonical(control_identity["data"]) != canonical(
            treatment_identity["data"]
        ) or canonical(control["runtime_compatibility"]) != canonical(
            treatment["runtime_compatibility"]
        ):
            raise ValueError(f"matched pair differs in data or runtime class: {study_id}/{seed}")
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

    mean_delta = sum(per_seed[str(seed)]["treatment_minus_control"]["wp"] for seed in SEEDS) / len(
        SEEDS
    )
    cells = {
        cell_id: {
            "variant": row["variant"],
            "seed": row["seed"],
            "score": row["score"],
            "checkpoint_sha256": row["checkpoint_sha256"],
            "score_sha256": row["score_sha256"],
            "profile_receipt_sha256": row["identity"]["bilinear_entmax_profile_receipt_sha256"],
            "full_shape_acceptance_sha256": row["identity"][
                "bilinear_entmax_full_shape_acceptance_sha256"
            ],
            "runtime_identity": row["identity"]["runtime"],
        }
        for cell_id, row in sorted(by_cell.items())
    }
    return {
        "schema": "bilinear-entmax.comparison.v1",
        "status": "complete_study",
        "runtime_commit": RUNTIME_COMMIT,
        "split_sha256": SPLIT_SHA256,
        "metric_id": METRIC_ID,
        "selected_rows": SCORED_ROWS,
        "validation_sequences": VALIDATION_SEQUENCES,
        "preprocessing_sha256": next(iter(preprocessing_hashes)),
        "source_sha256": next(iter(source_identities)),
        "data_sha256": canonical(by_cell[next(iter(sorted(by_cell)))]["identity"]["data"]),
        "study": {
            "id": study["id"],
            "family": study["family"],
            "control_variant": control_variant,
            "treatment_variant": treatment_variant,
            "contrast": study["contrast"],
            "fresh_fit_count": 4,
            "reused_historical_controls": 0,
        },
        "endpoint_admission": (
            "Predictions are admitted by bilinear_entmax.endpoint.load_endpoint, which "
            "strict-loads each fixed-final checkpoint and compares the full normalized native "
            "recipe; this comparator checks the E4 producer identity, per-cell profile receipt "
            "and full-shape acceptance digest, and source map."
        ),
        "cells": cells,
        "paired_contrast": {
            "per_seed": per_seed,
            "mean_paired_wp_delta": mean_delta,
        },
        "interpretation": (
            "Descriptive same-seed contrasts on the repeatedly exposed official development "
            "fold; two seeds do not establish significance or public transfer. No winner is "
            "selected."
        ),
    }


def main() -> None:
    """Write one immutable comparison receipt for an explicitly selected study."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=tuple(STUDIES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scores", type=Path, nargs=4, required=True)
    args = parser.parse_args()
    result = compare(args.study, args.scores)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_new(args.output, result)


if __name__ == "__main__":
    main()
