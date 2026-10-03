from __future__ import annotations

import marimo

__generated_with = "0.16.0"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    from pathlib import Path

    import marimo as mo
    import pandas as pd

    directory = Path(__file__).resolve().parent
    clock = json.loads((directory / "cache/continuation_results.json").read_text())
    hazard = json.loads((directory / "cache/continuation_hazard_results.json").read_text())
    manifest = json.loads((directory / "cache/continuation_manifest.json").read_text())
    return clock, directory, hazard, manifest, mo, pd


@app.cell
def _(clock, hazard, mo):
    mo.md(f"""
    # Econometrics: causal history, event time, and falsification

    Primary screen passed: **{clock["frozen_gate_passed"]}**.
    Event minus row-clock pooled WP: **{clock["event_minus_row"]:+.6f}**.
    Adaptive followup passed: **{hazard["frozen_gate_passed"]}**.
    Event minus hazard-clock pooled WP: **{hazard["event_minus_hazard_mean"]:+.6f}**.

    These small train-only ridge probes use 24 validation sequences previously
    exposed in my summary. They establish neither causal effects nor gains over the
    strong baseline. Prefix parity is exact. No targets, masks, or future features
    enter the history state. Sequence IDs are provenance only.
    """)
    return


@app.cell
def _(hazard, mo, pd):
    score_table = pd.DataFrame(
        [
            {
                "arm": name,
                "t0 scored WP": score["scored_wp"][0],
                "t1 scored WP": score["scored_wp"][1],
                "pooled mean WP": score["mean_scored_wp"],
                "all required mean WP": score["mean_all_required_wp"],
            }
            for name, score in hazard["scores"].items()
        ]
    )
    mo.ui.table(score_table)
    return (score_table,)


@app.cell
def _(directory, mo):
    mo.vstack(
        [
            mo.image(directory / "fig/continuation_clock_scores.png"),
            mo.image(directory / "fig/continuation_clock_window.png"),
            mo.image(directory / "fig/continuation_hazard_scores.png"),
        ]
    )
    return


@app.cell
def _(hazard, manifest, mo):
    mo.md(f"""
    ## Sample and provenance

    Training IDs: `{manifest["train_ids"]}`

    Validation IDs: `{manifest["valid_ids"]}`

    Training excludes my earlier lattice300, inferred ACF120, and train group0.
    The old event-delta table lacks its script and sample IDs, so novelty is only
    relative to reconstructable samples. ALL validation was summary-exposed.

    Full selected-row fingerprint screen: {hazard["train_rows_screened"]:,} train
    and {hazard["valid_rows_screened"]:,} validation rows; cross-split matching
    sequence pairs: **{len(hazard["exact_raw_row_cross_split_matches"])}**.
    This checks unchanged 112-feature float32 row identity only. Unsampled or
    transformed linked chains remain unknown. No independence interval is claimed.

    Scored rows: **{hazard["valid_scored_rows"]:,}**. Scores pool rows globally with
    abs(clipped target) weights. All-required scores diagnose mask sensitivity.
    """)
    return


@app.cell
def _(directory, mo):
    mo.md((directory / "REPORT.txt").read_text())
    return


if __name__ == "__main__":
    app.run()
