"""Publish the prespecified GRU since32 × distillation factorial comparison at E6."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from readout_evidence import ARMS, BASE, CACHE, METRICS, SEEDS, SPLIT, endpoint, qualification

# All signs use candidate minus comparator. Column order follows ARMS.
CONTRASTS = {
    "since32_minus_labels": [-1, 1, 0, 0],
    "kd_minus_labels": [-1, 0, 1, 0],
    "both_minus_labels": [-1, 0, 0, 1],
    "both_minus_since32": [0, -1, 0, 1],
    "both_minus_kd": [0, 0, -1, 1],
    "interaction": [1, -1, -1, 1],
}
BOOTSTRAP_SEED = 203303
REPLICATES = 2000
MIN_GAIN = 0.0005


def bootstrap(root: Path, panels: list[list[dict]], points: np.ndarray) -> dict:
    """Draw shared components across arms AND seeds, conditioning on the eight fitted models."""
    # qualification() has just re-hashed the training source and all modules under src.
    sys.path.insert(0, str(root / "runtime/src"))
    from wnn.evaluation import _score_component_moments

    keys = sorted(panels[0][0], key=int)
    if not keys or any(set(mapping) != set(keys) for panel in panels for mapping in panel):
        raise ValueError("centered component coverage differs across the factorial panel")
    moments = np.asarray(
        [[[mapping[k] for k in keys] for mapping in panel] for panel in panels], dtype=np.float64
    )
    if moments.shape != (len(SEEDS), len(ARMS), len(keys), 2, 6) or not np.isfinite(moments).all():
        raise ValueError("invalid canonical centered WP moments")
    for seed_index in range(len(SEEDS)):
        for arm_index in range(len(ARMS)):
            verified = _score_component_moments(
                moments[seed_index, arm_index], np.ones(len(keys), dtype=int)
            )
            if not np.isclose(verified, points[seed_index, arm_index], atol=1e-10, rtol=0):
                raise ValueError("component moments disagree with the canonical point estimate")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sampled = np.empty((REPLICATES, len(SEEDS), len(ARMS)))
    for draw in range(REPLICATES):
        counts = np.bincount(rng.integers(0, len(keys), size=len(keys)), minlength=len(keys))
        for seed_index in range(len(SEEDS)):
            for arm_index in range(len(ARMS)):
                sampled[draw, seed_index, arm_index] = _score_component_moments(
                    moments[seed_index, arm_index], counts
                )
    return {
        name: {
            "per_seed_ci95": {
                str(seed): np.percentile((sampled @ weights)[:, i], [2.5, 97.5]).tolist()
                for i, seed in enumerate(SEEDS)
            },
            "paired_mean_ci95": np.percentile(
                (sampled @ weights).mean(axis=1), [2.5, 97.5]
            ).tolist(),
        }
        for name, weights in CONTRASTS.items()
    }


def collect(root: Path, split_path: Path) -> dict:
    """Require all eight independent grades before computing contrasts or allocation decisions."""
    q, fitting, valid = qualification(root, split_path)
    packet_root = Path(__file__).resolve().parent
    commit = subprocess.check_output(
        ["git", "-C", str(packet_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if q["packet_commit"] != commit:
        raise ValueError("readout must use the packet commit recorded by qualification")
    cells, common, checkpoints = {}, set(), set()
    panels = []
    scores = np.empty((len(METRICS), len(SEEDS), len(ARMS)))
    for i, seed in enumerate(SEEDS):
        panel = []
        for j, arm in enumerate(ARMS):
            cell = f"{arm}_s{seed}"
            row, moments, shared = endpoint(root, arm, seed, q, fitting, valid)
            cells[cell] = row
            common.add(shared)
            checkpoints.add(row["model_sha256"])
            scores[:, i, j] = [row["grade"][key] for key in METRICS]
            panel.append(moments)
        panels.append(panel)
    if len(common) != 1 or len(checkpoints) != len(cells):
        raise ValueError(
            "scores must have matched common ancestry/settings and eight distinct fixed E6 checkpoints"
        )
    intervals = bootstrap(root, panels, scores[0])
    findings = {}
    for name, weights in CONTRASTS.items():
        delta = scores @ weights
        findings[name] = {
            "per_seed": {
                str(seed): dict(zip(METRICS, delta[:, i].tolist(), strict=True))
                for i, seed in enumerate(SEEDS)
            },
            "equal_seed_mean": dict(zip(METRICS, delta.mean(axis=1).tolist(), strict=True)),
            "conditional_wp": intervals[name],
        }

    def useful(name: str) -> bool:
        return findings[name]["equal_seed_mean"]["wp"] >= MIN_GAIN and all(
            row["wp"] > 0 for row in findings[name]["per_seed"].values()
        )

    signs = [x["wp"] for x in findings["interaction"]["per_seed"].values()]
    return {
        "schema": "wnn.b03.fixed-e6-factorial.v1",
        "packet_commit": commit,
        "base_commit": BASE,
        "fit_source_sha256": q["source_sha256"],
        "teacher_manifest_sha256": CACHE,
        "split_sha256": SPLIT,
        "cells": cells,
        "paired_contrasts": findings,
        "followup_for_combination": useful("both_minus_since32") and useful("both_minus_kd"),
        "kd_followup_on_since32": useful("both_minus_since32"),
        "interaction_direction": "subadditive_both_seeds"
        if max(signs) < 0
        else "superadditive_both_seeds"
        if min(signs) > 0
        else "mixed_or_zero",
        "uncertainty": {
            "unit": "held-out evaluation component",
            "same_draw_all_cells": True,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "draws": REPLICATES,
            "scope": "conditional on eight fixed E6 checkpoints and this development-exposed partition",
        },
        "decision_rule": "Follow up the combination only for positive gain on each seed and mean gain >=0.0005 WP over each individual intervention. The interaction estimates substitution independently of this decision.",
    }


def main() -> None:
    """Write one immutable comparison using receiver artifacts and the original chronological split."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root.resolve(strict=True)
    output = root / "comparison.json"
    if output.exists():
        raise FileExistsError(output)
    report = collect(root, args.split_manifest)
    with tempfile.NamedTemporaryFile(mode="w", dir=root, encoding="utf-8") as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        os.link(stream.name, output)


if __name__ == "__main__":
    main()
