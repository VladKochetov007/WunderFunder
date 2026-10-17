#!/usr/bin/env python3
"""Pair canonical point deltas with conditional chain-bootstrap percentiles."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    raw = args.results.read_bytes()
    results = json.loads(raw)
    summary: dict[str, object] = {
        "schema": "wunder.prediction-audit.bootstrap-summary.v1",
        "results_sha256": hashlib.sha256(raw).hexdigest(),
        "point_estimate_source": "exact canonical clipped pooled WP in RESULTS.json scenario metrics minus that model's baseline",
        "interval_source": "paired_chain_bootstrap in RESULTS.json, percentile interval conditional on fixed cached arrays",
        "models": {},
    }
    model_summary: dict[str, object] = summary["models"]  # type: ignore[assignment]
    bootstrap = results["paired_chain_bootstrap"]
    for model_index, model in enumerate(results["models"]):
        model_name = model["cache_name"]
        baseline = model["scenarios"]["baseline"]["global"]
        scenarios: dict[str, object] = {}
        for scenario_name, scenario in model["scenarios"].items():
            if scenario_name == "baseline":
                continue
            observed = scenario["global"]
            interval = bootstrap["intervals_by_model_index"][str(model_index)][scenario_name]
            scenarios[scenario_name] = {
                target_name: {
                    "observed_delta": float(observed[target_name] - baseline[target_name]),
                    "bootstrap_mean_delta": float(interval[target_name]["mean_delta"]),
                    "q025": float(interval[target_name]["q025"]),
                    "q975": float(interval[target_name]["q975"]),
                }
                for target_name in ("t0", "t1", "wp")
            }
        model_summary[model_name] = scenarios
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
