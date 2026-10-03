from __future__ import annotations

import marimo

app = marimo.App(width="full")


@app.cell
def _():
    import json
    from pathlib import Path

    import marimo as mo
    import pandas as pd

    root = Path(__file__).resolve().parent
    results = json.loads((root / "continuation_results.json").read_text())
    followup = json.loads((root / "continuation_hazard_control.json").read_text())
    provenance = json.loads((root / "continuation_samples.json").read_text())
    return followup, mo, pd, provenance, results, root


@app.cell
def _(mo, results):
    mo.md(f"""
    # Statistical physics continuation: causal event memory

    Frozen H1/H2 followed by stricter H3 controls. This notebook reads saved
    measurements and figures; it does not retrain models or scan raw datasets.
    **Verdict: {results["h1_verdict"]}** on a local ridge proxy.
    Event versus row-clock smoothing remains unresolved.

    48 train / 32 validation sequences, {results["valid_scored_rows"]:,} scored
    validation rows after common step-611 burn-in. All sequence targets were
    previously census-exposed; no campaign-independent confirmation is claimed.
    """)
    return


@app.cell
def _(mo, pd, results):
    table = pd.DataFrame(
        [
            {
                "model": name,
                "sampled WP": score,
                "t0 WP": results["target_correlations"][name][0],
                "t1 WP": results["target_correlations"][name][1],
            }
            for name, score in results["scores"].items()
        ]
    )
    mo.ui.table(table)
    return


@app.cell
def _(mo, root):
    mo.image(str(root / "f24_causal_memory_controls.png"))
    return


@app.cell
def _(followup, mo, pd):
    hazard_table = pd.DataFrame.from_dict(followup["results"], orient="index")
    mo.vstack([mo.md(f"Follow-up verdict: **{followup['verdict']}**"), mo.ui.table(hazard_table)])
    return


@app.cell
def _(mo, root):
    mo.image(str(root / "f25_conditional_age_controls.png"))
    return


@app.cell
def _(mo, pd, provenance):
    samples = pd.DataFrame(
        [
            {"split": split, **record}
            for split, records in provenance["samples"].items()
            for record in records
        ]
    )
    mo.vstack([mo.md("## Exact sample IDs and exposure"), mo.ui.table(samples)])
    return


if __name__ == "__main__":
    app.run()
