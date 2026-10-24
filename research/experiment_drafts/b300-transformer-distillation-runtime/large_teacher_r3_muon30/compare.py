"""Compare the four fixed large-teacher endpoints on the receiving host."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from admission import digest, file_hash, read, require, source_hash, write_new

from wnn.evaluation import paired_component_bootstrap

PACKET = Path(__file__).resolve().parent


def endpoint(root: Path, cell: str) -> dict[str, Any]:
    """Bind an official grade to the frozen config, final weights and prediction manifest."""
    directory = root / cell
    status = read(directory / "fit/terminal_status.json")
    config = read(PACKET / "configs" / f"{cell}.json")
    require(
        status["status"] == "completed" and status["completed_epochs"] == config["epochs"],
        f"Incomplete fixed endpoint: {cell}",
    )
    require(status["identity"]["config_sha256"] == digest(config), f"Wrong config: {cell}")
    require(status["identity"]["source_sha256"] == source_hash(), f"Wrong source: {cell}")
    checkpoint_hash = file_hash(directory / "fit/model.pt")
    require(checkpoint_hash == status["model_sha256"], f"Weights changed: {cell}")
    score = read(directory / "fit/score.json")
    require(score["producer"]["checkpoint_sha256"] == checkpoint_hash, f"Wrong grade model: {cell}")
    require(
        score["predictions_manifest_sha256"] == file_hash(directory / "evaluation/manifest.json"),
        f"Prediction manifest changed: {cell}",
    )
    require(
        score["split_sha256"] == status["identity"]["data"]["split_sha256"], f"Wrong split: {cell}"
    )
    require(score["score"]["metric_id"] == "global_wp_clip2_v1", f"Wrong metric: {cell}")
    return score


def main() -> None:
    """Compute paired component intervals without selecting a different endpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    cells = {}
    deltas = []
    receipts = {}
    for fold in (0, 2):
        small_name, large_name = f"t1280_f{fold}", f"t1792_f{fold}"
        small, large = endpoint(args.run_root, small_name), endpoint(args.run_root, large_name)
        require(small["split_sha256"] == large["split_sha256"], "Pair has different splits")
        require(
            small["score"]["selected_rows"] == large["score"]["selected_rows"], "Row count drift"
        )
        groups = sorted(small["component_centered_moments"], key=int)
        require(
            groups == sorted(large["component_centered_moments"], key=int),
            "Component membership drift",
        )
        reference = np.asarray(
            [small["component_centered_moments"][g] for g in groups], dtype=np.float64
        )
        candidate = np.asarray(
            [large["component_centered_moments"][g] for g in groups], dtype=np.float64
        )
        require(
            np.allclose(
                candidate[:, :, [0, 1, 3]], reference[:, :, [0, 1, 3]], rtol=1e-10, atol=1e-8
            ),
            "Different labels/weights inside pair",
        )
        interval = paired_component_bootstrap(
            candidate, reference, np.array(groups), n_boot=2000, seed=1729 + fold, centered=True
        )
        delta = large["score"]["wp"] - small["score"]["wp"]
        require(
            abs(interval["mean_diff"] - delta) < 2e-7,
            "Centered-moment and official grades disagree",
        )
        deltas.append(delta)
        cells[f"F{fold}"] = {
            "small": small["score"],
            "large": large["score"],
            "large_minus_small": delta,
            "component_bootstrap": interval,
        }
        for name in (small_name, large_name):
            path = args.run_root / name / "fit/score.json"
            receipts[name] = file_hash(path)
    mean = float(np.mean(deltas))
    write_new(
        args.run_root / "comparison.json",
        {
            "schema": "wnn.large-teacher-comparison.v2",
            "cells": cells,
            "mean_paired_delta": mean,
            "useful_followup_signal": mean >= 0.001 and min(deltas) >= 0,
            "score_sha256": receipts,
            "source_sha256": source_hash(),
            "limits": "Exposed folds; intervals condition on fixed fits. No seed/public significance or student-transfer claim.",
            "moment_representation": "Stable centered moments merged within each component, then paired bootstrap.",
        },
    )


if __name__ == "__main__":
    main()
