#!/usr/bin/env python3
"""Create a clearly labeled view correcting two reporting-field ambiguities."""

from __future__ import annotations

import argparse
import copy
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
    source_bytes = args.results.read_bytes()
    raw = json.loads(source_bytes)
    corrected = copy.deepcopy(raw)
    corrected["schema"] = "wunder.prediction-audit.results-corrected-view.v1"
    corrected["source_results_sha256"] = hashlib.sha256(source_bytes).hexdigest()
    corrected["presentation_corrections"] = [
        "Exposure fields are renamed to show that the original selected_rows value counted rows where a transform changed output and the official mask selected them; for baseline this count is zero.",
        "Bootstrap mean_delta is renamed bootstrap_mean_delta; observed_delta is computed directly from canonical scenario WP minus canonical baseline WP.",
        "No predictions, masks, moments, score values, bootstrap replicates, or intervals are changed.",
    ]

    for model_index, model in enumerate(corrected["models"]):
        baseline = model["scenarios"]["baseline"]["global"]
        for scenario_name, scenario in model["scenarios"].items():
            exposure = scenario["exposure"]
            old_selected = int(exposure.pop("selected_rows"))
            all_rows = int(exposure.pop("rows"))
            changed_all = int(exposure.pop("different_from_baseline_rows"))
            exposure.clear()
            exposure.update(
                {
                    "total_rows": all_rows,
                    "changed_selected_rows": 0 if scenario_name == "baseline" else old_selected,
                    "changed_rows_all": changed_all,
                }
            )
            if scenario_name == "baseline":
                continue
            bootstrap_interval = corrected["paired_chain_bootstrap"]["intervals_by_model_index"][
                str(model_index)
            ][scenario_name]
            observed = scenario["global"]
            for target_name in ("t0", "t1", "wp"):
                interval = bootstrap_interval[target_name]
                interval["bootstrap_mean_delta"] = float(interval.pop("mean_delta"))
                interval["observed_delta"] = float(observed[target_name] - baseline[target_name])

    args.output.write_text(
        json.dumps(corrected, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
