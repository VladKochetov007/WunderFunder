"""Grade the preregistered six-fit numerical-input panel from canonical receipts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from wnn.evaluation import paired_component_bootstrap
from wnn.training.sequence_checkpoint import atomic_write_json
from wnn.training.sequence_config import load_config
from wnn.training.sequence_provenance import file_sha256

ARMS = ("raw_events", "uniform16", "quantile16")
CELLS = ((0, 1001), (2, 1002))


def _read_cell(root: Path, configs: Path, fold: int, seed: int) -> dict[str, Any]:
    receipts: dict[str, Any] = {}
    shared: dict[str, Any] | None = None
    for arm in ARMS:
        path = root / f"f{fold}_{arm}" / "score.json"
        grade = json.loads(path.read_text())
        if grade.get("score", {}).get("metric_id") != "global_wp_clip2_v1":
            raise ValueError(f"invalid canonical grade receipt: {path}")
        producer = grade["producer"]
        config = producer["config"]
        if config["model"]["input_encoding"] != arm:
            raise ValueError("arm label does not match its numerical input encoding")
        if config != load_config(configs / f"f{fold}_{arm}.json") or config["seed"] != seed:
            raise ValueError(f"config differs from the frozen panel: {path}")
        provenance = producer["provenance"]
        if (
            provenance["completed_epochs"] != 12
            or provenance["fixed_checkpoint_policy"] != "configured-final-epoch"
        ):
            raise ValueError("panel accepts only completed fixed-final endpoints")
        identity = provenance["identity"]
        data = identity["data"]
        preparation = data["input_preparation"]
        if (
            preparation["calibration_seed"] != 221022
            or preparation["reservoir_rows_limit"] != 131072
        ):
            raise ValueError("preprocessing differs from the frozen calibration recipe")
        if data["fold"] != fold or data["split_sha256"] != grade["split_sha256"]:
            raise ValueError("cell fold/split mismatch")
        basis = {
            "data": data,
            "source": identity["source_sha256"],
            "runtime": identity["runtime"],
            "driver": identity["nvidia_driver_version"],
            "world_size": identity["world_size"],
            "initial_common_parameters_sha256": identity["initial_common_parameters_sha256"],
            "components": sorted(grade["component_raw_sums"]),
            "scored_rows": grade["score"]["selected_rows"],
            "sequence_count": grade["sequences"],
        }
        basis["data"] = {key: value for key, value in data.items() if key != "feature_encoding"}
        if not data.get("input_artifact_sha256") or not basis["initial_common_parameters_sha256"]:
            raise ValueError("missing common preprocessing/initialization identity")
        if shared is not None and basis != shared:
            raise ValueError(
                "arms do not share data, preprocessing, initialization, source/runtime or coverage"
            )
        shared = basis
        receipts[arm] = grade | {"receipt_sha256": file_sha256(path)}
    return receipts


def grade_numeric_panel(root: Path, configs: Path, output: Path) -> None:
    """Compare all declared arms with paired whole-component bootstrap on both cells."""
    if output.exists():
        raise FileExistsError(output)
    cells = [_read_cell(root, configs, fold, seed) for fold, seed in CELLS]
    contrasts: dict[str, Any] = {}
    for candidate, reference in (
        ("quantile16", "uniform16"),
        ("uniform16", "raw_events"),
        ("quantile16", "raw_events"),
    ):
        reports, draws = [], []
        for index, (cell, (fold, seed)) in enumerate(zip(cells, CELLS, strict=True)):
            left, right = cell[candidate], cell[reference]
            components = sorted(left["component_raw_sums"])
            a = np.asarray([left["component_raw_sums"][key] for key in components])
            b = np.asarray([right["component_raw_sums"][key] for key in components])
            report = paired_component_bootstrap(
                a,
                b,
                np.arange(len(components)),
                n_boot=10_000,
                seed=221022 + index,
                return_draws=True,
            )
            draws.append(np.asarray(report.pop("bootstrap_deltas")))
            report.update(
                {
                    "fold": fold,
                    "fit_seed": seed,
                    "candidate_score": left["score"],
                    "reference_score": right["score"],
                    "candidate_receipt_sha256": left["receipt_sha256"],
                    "reference_receipt_sha256": right["receipt_sha256"],
                    "n_sequences": left["sequences"],
                }
            )
            if not np.isclose(
                report["mean_diff"], left["score"]["wp"] - right["score"]["wp"], atol=1e-10, rtol=0
            ):
                raise ValueError("bootstrap statistic disagrees with canonical pooled score")
            reports.append(report)
        mean_draws = np.mean(draws, axis=0)
        contrasts[f"{candidate}-minus-{reference}"] = {
            "cells": reports,
            "mean_diff": float(np.mean([report["mean_diff"] for report in reports])),
            "equal_cell_ci95": np.percentile(mean_draws, [2.5, 97.5]).tolist(),
            "bootstrap_fraction_le_zero": float(np.mean(mean_draws <= 0)),
            "per_target_mean_diff": {
                target: float(
                    np.mean(
                        [
                            cell[candidate]["score"][target] - cell[reference]["score"][target]
                            for cell in cells
                        ]
                    )
                )
                for target in ("t0", "t1")
            },
        }
    atomic_write_json(
        {
            "schema": "wnn.numeric-panel.v1",
            "metric_id": "global_wp_clip2_v1",
            "primary": "quantile16-minus-uniform16",
            "contrasts": contrasts,
            "practical_delta": 0.0003,
            "all_folds_exposed": True,
            "interpretation": "Exploratory two-cell screen; component CI excludes seed and unseen-fold uncertainty. Raw contrasts change capacity. No public-score or significance claim.",
        },
        output,
    )


def main() -> None:
    """Read six canonical score receipts and write the complete comparison panel."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--configs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    grade_numeric_panel(args.runs, args.configs, args.output)


if __name__ == "__main__":
    main()
