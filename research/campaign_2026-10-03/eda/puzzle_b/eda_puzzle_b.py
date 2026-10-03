from __future__ import annotations

import marimo

__generated_with = "0.16.0"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    from pathlib import Path

    import marimo as mo
    import matplotlib
    import numpy as np
    import pandas as pd

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path(__file__).resolve().parent
    scores = pd.read_csv(directory / "continuation_scores.csv")
    sequence_scores = pd.read_csv(directory / "continuation_sequence_scores.csv")
    recurrence = pd.read_csv(directory / "continuation_recurrence.csv")
    manifest = json.loads((directory / "continuation_manifest.json").read_text())
    checks = json.loads((directory / "continuation_checks.json").read_text())
    ablation = pd.read_csv(directory / "ablation_scores.csv")
    audit = json.loads((directory / "ablation_audit.json").read_text())
    return (
        ablation,
        audit,
        checks,
        directory,
        manifest,
        mo,
        np,
        plt,
        recurrence,
        scores,
        sequence_scores,
    )


@app.cell
def _(checks, manifest, mo):
    mo.md(f"""
    # Puzzle B: causal clock falsifier
    This notebook reads only this analyst's bounded continuation outputs.
    Old notebook is preserved as `eda_puzzle_b_prior.py`; its target-derived
    scale routing and full-sequence regime selection are unsuitable for inference.
    Main fit uses stride 7, sensitivity fit stride 6, and all eligible scored
    validation rows. No target-based class routing. Clock arms are compared
    to the same raw + causal age control. Sham is a fixed randomized row clock.
    Checks: `{checks}`. CPU time: {manifest["wall_seconds"]:.1f} seconds;
    peak RSS: {manifest["maxrss_kib"] / 1024:.1f} MiB.
    Sequence dependence and strong-baseline improvement remain untested.
    """)
    return


@app.cell
def _(mo, scores):
    mo.ui.table(scores, selection=None)
    return


@app.cell
def _(directory, mo, np, plt, scores, sequence_scores):
    _arms = ["raw", "age", "period4", "period6", "sham6"]
    _fig, _axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for _stride, _offset in [(7, -0.17), (6, 0.17)]:
        _subset = scores[scores.stride == _stride].set_index("arm").loc[_arms]
        _axes[0].bar(np.arange(5) + _offset, _subset["mean"], 0.32, label=f"train stride {_stride}")
        _axes[1].plot(_arms, _subset.delta_age, "o-", label=f"stride {_stride}")
    _axes[0].set_xticks(np.arange(5), _arms)
    _axes[0].set_title("Pooled clipped weighted correlation")
    _axes[0].legend()
    _axes[1].axhline(0, color="black", lw=0.8)
    _axes[1].set_title("Matched delta vs causal age control")
    _axes[1].legend()
    _paired = sequence_scores[sequence_scores.stride == 7].pivot(
        index="rg", columns="arm", values="mean"
    )
    for _j, _arm in enumerate(["period4", "period6", "sham6"]):
        _deltas = (_paired[_arm] - _paired.age).to_numpy()
        _axes[2].scatter(np.full(len(_deltas), _j), _deltas, s=18, alpha=0.6)
        _axes[2].plot(_j, np.median(_deltas), "kx", markersize=10)
    _axes[2].set_xticks(range(3), ["period4", "period6", "sham6"])
    _axes[2].axhline(0, color="black", lw=0.8)
    _axes[2].set_title("Paired per-sequence deltas (no CI)")
    _fig.tight_layout()
    _fig.savefig(directory / "fig10_causal_clock_falsifier.png", dpi=120)
    mo.mpl.interactive(_fig)
    return


@app.cell
def _(directory, mo, np, plt, recurrence):
    _fig, _ax = plt.subplots(figsize=(9, 4))
    for _rg, _frame in recurrence.groupby("rg"):
        _variance = _frame.event_rate * (1 - _frame.event_rate)
        _contrast = (_frame.repeat_rate - _frame.event_rate**2) / np.maximum(_variance, 1e-9)
        _ax.plot(_frame.lag, _contrast, "o-", alpha=0.3, lw=0.8)
    _ax.axhline(0, color="black", lw=0.8)
    _ax.set_xticks([3, 4, 5, 6, 7])
    _ax.set_xlabel("lag in rows")
    _ax.set_ylabel("normalized event coincidence excess")
    _ax.set_title("24 preselected valid sequences: clocks vary by sequence")
    _fig.tight_layout()
    _fig.savefig(directory / "fig11_event_clock_corners.png", dpi=120)
    mo.mpl.interactive(_fig)
    return


@app.cell
def _(ablation, audit, directory, mo, plt):
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4))
    _axes[0].bar(ablation.arm, ablation.delta_cap4)
    _axes[0].axhline(0, color="black", lw=0.8)
    _axes[0].set_title("Ablation: pooled delta vs capped age")
    _axes[0].tick_params(axis="x", rotation=20)
    _sensitivity = audit["posthoc_concentration_sensitivity"]
    _axes[1].bar(
        [str(_s["excluded_rg"]) for _s in _sensitivity], [_s["delta_mean"] for _s in _sensitivity]
    )
    _axes[1].axhline(0, color="black", lw=0.8)
    _axes[1].set_xlabel("excluded row groups (post hoc)")
    _axes[1].set_title("Full4 vs cap4: concentration sensitivity")
    _fig.tight_layout()
    _fig.savefig(directory / "fig12_clock_ablation.png", dpi=120)
    mo.mpl.interactive(_fig)
    return


@app.cell
def _(manifest, mo):
    mo.md(f"""
    Selected train row groups: `{manifest["sample"]["train"]}`.
    Selected valid row groups: `{manifest["sample"]["valid"]}`.
    Full seq_ix, source footer hashes, input hashes, scored-step hashes and prior
    exposure are in `continuation_manifest.json`. Sample disjointness is only
    relative to recovered own exposures; it is not campaign-wide freshness.
    Direct seq_ix overlap: `{manifest["direct_seq_ix_overlap"]}`.
    Prior own exposure: `{manifest["exposure_counts"]}`.
    Correction: the old train scan sampled 2,000 random row groups, and the
    old valid scan covered all 1,873 sequences. The initially recovered range
    assumption was incorrect. This is an exploratory test on exposed validation.
    Component independence: {manifest["component_independence"]}.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    Verdict: period-6 augmentation was negative. Phase-4 alone contributed
    essentially zero. Lag-4 recurrence carried an exploratory gain, mainly
    in two sequences; full4 versus capped-age wins only 12 of 24 sequences.
    Excluding those two post hoc leaves a mean delta of 0.000613.
    The frozen consistency criterion fails. No improvement over the controller's
    strong baseline or physical causal mechanism has been established.
    """)
    return


if __name__ == "__main__":
    app.run()
