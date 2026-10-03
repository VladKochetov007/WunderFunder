from __future__ import annotations

import marimo

app = marimo.App(width="full")


@app.cell
def _():
    import json
    from pathlib import Path

    import marimo as mo
    import numpy as np

    return Path, json, mo, np


@app.cell
def _(Path, json):
    own_dir = Path(__file__).resolve().parent
    clock = json.loads((own_dir / "continuation_results.json").read_text())
    stationary = json.loads((own_dir / "stationarity_results.json").read_text())
    provenance = json.loads((own_dir / "sample_manifest.json").read_text())
    return clock, own_dir, provenance, stationary


@app.cell
def _(clock, mo, np, stationary):
    wp_values = np.array(clock["pooled_wp"])
    score_lines = ["| Arm | t0 WP | t1 WP | Mean WP |", "|---|---:|---:|---:|"]
    for arm_name, score in zip(clock["arms"], wp_values, strict=True):
        score_lines.append(f"| {arm_name} | {score[0]:.6f} | {score[1]:.6f} | {score.mean():.6f} |")
    mo.md(
        "# Fractal continuation: measured causal falsifiers\n\n"
        + "\n".join(score_lines)
        + f"\n\nEvent minus row: {clock['event_minus_row']:+.6f}. "
        + f"Frozen event-clock gate: **{clock['clock_survives']}**. "
        + f"Prefix/future-perturbation error: {clock['causality_max_abs_error']:.1e}.\n\n"
        + "DFA original-minus-shuffled increments median: "
        + f"{clock['scaling_median_delta']:+.6f}; "
        + f"positive sequences: {clock['scaling_positive_sequences']}/12.\n\n"
        + "Original-minus-differenced-shuffled-levels median: "
        + f"{stationary['median_original_minus_level_null']:+.6f}. "
        + "The stationary feature encoding is a plausible source of low DFA slopes. "
        + "These are retrospective diagnostics, not prediction features.\n\n"
        + "Local ridge comparisons do not establish improvement over the campaign baseline. "
        + "Connected components and campaign exposure remain unknown."
    )
    return


@app.cell
def _(mo, own_dir):
    mo.vstack(
        [
            mo.image(str(own_dir / "fig8_causal_clock.png")),
            mo.image(str(own_dir / "fig9_shuffle_falsifier.png")),
            mo.image(str(own_dir / "fig10_stationary_level_control.png")),
        ]
    )
    return


@app.cell
def _(json, mo, provenance):
    mo.md(
        "## Sample provenance\n\n"
        "Uniform seed 20261003; 32 train and 24 validation row groups exclude the "
        "conservative own prior exposure union. Actual seq_ix IDs follow. "
        "The stationary-level follow-up reuses the first 12 validation sequences "
        "after the initial screen; it is explicitly exploratory.\n\n"
        + "```json\n"
        + json.dumps(provenance["new_sample"], indent=2)
        + "\n```\n\n"
        + "Methods: [Kantelhardt et al., shuffled DFA controls]"
        + "(https://arxiv.org/abs/physics/0202070). "
        + "Our ordinary DFA1 diagnostic does not establish multifractality or asymptotic long memory."
    )
    return


if __name__ == "__main__":
    app.run()
