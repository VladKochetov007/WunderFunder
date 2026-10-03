from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from continuation_probe import ROOT, dfa, save_json, scaling_slope

OUT = Path(__file__).resolve().parent


def main() -> None:
    """Distinguish ordered increments from a shuffled stationary-level null."""
    manifest = json.loads((OUT / "sample_manifest.json").read_text())
    initial = json.loads((OUT / "continuation_results.json").read_text())
    rng = np.random.default_rng(20261004)
    parquet = pq.ParquetFile(ROOT / "datasets/valid.parquet")
    rows = []
    for record in manifest["new_sample"]["valid"][:12]:
        table = parquet.read_row_group(
            record["row_group"], columns=[f"i0_p{k}" for k in range(22)], use_threads=False
        )
        prices = np.column_stack([table[f"i0_p{k}"].to_numpy() for k in range(22)])
        events = np.r_[True, np.any(np.diff(prices, axis=0) != 0, axis=1)]
        levels = prices[events, 0].astype(float)
        slopes = [scaling_slope(dfa(np.diff(rng.permutation(levels)))) for _ in range(16)]
        original = scaling_slope(dfa(np.diff(levels)))
        rows.append(
            {
                **record,
                "prior_exposure": "scored in initial continuation probe",
                "original_slope": original,
                "shuffled_level_slopes": slopes,
                "original_minus_level_null": original - float(np.median(slopes)),
            }
        )
    # Synthetic controls test the estimator against known constructions, not labels.
    noise = rng.normal(size=20000)
    synthetic = {
        "iid_increment_slope": scaling_slope(dfa(noise)),
        "differenced_iid_level_slope": scaling_slope(dfa(np.diff(noise))),
    }
    difference = np.array([row["original_minus_level_null"] for row in rows])
    result = {
        "rows": rows,
        "synthetic": synthetic,
        "median_original_minus_level_null": float(np.median(difference)),
        "stationary_level_explanation_survives": bool(abs(np.median(difference)) < 0.05),
        "scope": "post-initial exploratory mechanistic follow-up frozen before scoring; same exposed 12 sequences; no targets",
    }
    save_json("stationarity_results.json", result)
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot([row["original_slope"] for row in rows], "o-", label="original event increments")
    axis.plot(
        [np.median(row["shuffled_level_slopes"]) for row in rows],
        "o-",
        label="differenced shuffled levels",
    )
    axis.plot(
        [np.median(row["shuffled_slopes"]) for row in initial["scaling"]],
        "o-",
        label="shuffled increments",
    )
    axis.set_xlabel("same exposed validation sequence index")
    axis.set_ylabel("DFA1 slope, event scales 16..256")
    axis.set_title("Stationary encoded levels can create near-zero increment slopes")
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUT / "fig10_stationary_level_control.png", dpi=150)
    plt.close(figure)
    state = json.loads((OUT / "STATE.json").read_text())
    state["phase"] = "stationarity_control_complete"
    state["commands"].append(
        "research/campaign_2026-10-03/continuation/run_cpu.sh eda-fractal .venv/bin/python research/campaign_2026-10-03/eda/fractal/stationarity_control.py"
    )
    state["artifacts"].extend(["stationarity_results.json", "fig10_stationary_level_control.png"])
    save_json("STATE.json", state)


if __name__ == "__main__":
    main()
