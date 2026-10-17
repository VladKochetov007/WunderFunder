import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path
    import json
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    ROOT = Path(mo.notebook_dir()).parent
    RUN = ROOT / "research/campaign_2026-10-17/moment_geometry_r1"

    return RUN, json, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Sequence mean/std geometry

    Raw train/valid distributions, bounded-support identities, and what can be estimated causally.

    **Important:** full-sequence moments are descriptive diagnostics. Prediction features may use only rows observed so far. The screenshot's feature is not yet identified; the scan covers all 112 inputs, with 128 complete sequences from each file and population standard deviation (`ddof=0`). Validation is exposed, and a visual arc is not evidence of model improvement.

    For a two-level sequence with values $a,b$: $\sigma^2=(\mu-a)(b-\mu)$, equivalently $(\mu-(a+b)/2)^2+\sigma^2=((b-a)/2)^2$. Any values inside $[a,b]$ satisfy the corresponding inequality; equality diagnoses endpoint-only support.
    """)
    return


@app.cell(hide_code=True)
def _(RUN, json, mo, np):
    with np.load(RUN / "scan_r1/moments.npz", allow_pickle=False) as _source:
        moments = {key: _source[key] for key in _source.files}
    scan_metadata = json.loads((RUN / "scan_r1/scan.json").read_text())
    arc_rows = json.loads((RUN / "scan_r1/robust_arcs.json").read_text())
    support_probe = json.loads((RUN / "scan_r1/support_probe.json").read_text())
    column_picker = mo.ui.dropdown(options=list(moments["columns"]), value="i0_p0", label="Raw feature")
    column_picker
    return arc_rows, column_picker, moments, support_probe


@app.cell(hide_code=True)
def _(column_picker, moments, np, plt):
    _j = list(moments["columns"]).index(column_picker.value)
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for _split, _color in (("train", "tab:blue"), ("valid", "tab:orange")):
        _mu = moments[f"{_split}_mean"][:, -1, _j]
        _sd = np.sqrt(moments[f"{_split}_var"][:, -1, _j])
        _axes[0].scatter(_mu, _sd, s=14, alpha=0.65, label=_split, color=_color)
        _axes[1].hist(_mu, bins=24, density=True, histtype="step", label=_split, color=_color)
    for _axis in _axes:
        _axis.grid(alpha=0.2)
        _axis.legend()
    _axes[0].set(xlabel="Full-sequence raw mean", ylabel="Population std", title=column_picker.value)
    _axes[1].set(xlabel="Full-sequence raw mean", ylabel="Density", title="Train / validation distribution")
    _fig
    return


@app.cell(hide_code=True)
def _(RUN, arc_rows, mo, support_probe):
    mo.vstack([
        mo.md("""## Reconstructed arcs

    `i0_p0` closely resembles the supplied screenshot. Curves are **train-fitted exploratory consensus fits**, not exact support bounds. All112 columns were scanned; this selection is adaptive."""),
        mo.image(RUN / "scan_r1/candidate_arcs.png"),
        mo.ui.table(arc_rows),
        mo.md("""## Raw support of selected sequences

    Near-arc examples have multiple distinct values; the exact two-point explanation fails for these examples. A frequent value can dominate occupancy, suggesting a point-mass plus broad-distribution mixture. No targets were used to select these examples."""),
        mo.ui.table(support_probe["rows"]),
    ])
    return


@app.cell(hide_code=True)
def _(plt, support_probe):
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for _split in ("train", "valid"):
        _rows = [_r for _r in support_probe["prefix_errors"] if _r["split"] == _split and _r["length"] < 20000]
        for _axis, _metric in zip(_axes, ("mean_mae", "std_mae")):
            _axis.plot([_r["length"] for _r in _rows], [_r[_metric] for _r in _rows], marker="o", label=_split)
            _axis.set(xscale="log", xlabel="Observed prefix rows", ylabel=_metric)
            _axis.grid(alpha=.2)
            _axis.legend()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Research implications

    1. **Retain raw levels.** Whole-sequence scaling would leak unseen rows; forcing every point onto one arc would erase the separate cloud.
    2. **Add causal evidence.** Expanding mean/std, signed arc residual and occupancy of a train-derived frequent value describe the observed prefix.
    3. **Test scale as a separate addition.** Keep raw + moment features and add a scale-floored, clipped normalized current value.
    4. **Measure model quality.** Geometry and prefix accuracy are not target-prediction gains. Matched recurrent pilots use disjoint components and the official metric. Exposed validation cannot establish fresh generalization.

    Initial Parquet metadata parsing exceeded its1.4GB child cap. The corrected one-reader scan completed. The task slice remains5GB; no Kaggle jobs are touched.
    """)
    return


@app.cell(hide_code=True)
def _(RUN, mo):
    mo.vstack([
        mo.md("""## Visual overview — all112 raw inputs

    Most price-like fields show narrow branches or arcs alongside a second cloud; corresponding i0/i1 shapes often resemble each other. Volume-like fields show broader multimodal clusters. Trade-volume channels show strong mean/std coupling with column-specific direction. Additional a0–a3 are mostly clouds (a1 mean nearly zero); a4/a6 have a decreasing relation and a7 has two visible branches. Blue=train, orange=valid. These are visual observations on128complete sequences per source, not identified mechanisms or a distribution-shift test."""),
        *[mo.image(RUN / f"scan_r1/all_columns_{_page}.png") for _page in range(1, 5)],
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A circle does not require exactly two values

    If $X$ is a mixture of a frequent value $a$ and a fixed broad distribution $Q$, changing only the mixture weight makes $E[X^2]$ linear in $E[X]$. Thus $\sigma^2=B\mu+C-\mu^2$ is again a circular arc. Its second intersection with $\sigma=0$ may be a **virtual endpoint**, not an observed bound.

    That is compatible with our near-arc examples: the most common level can occupy most rows, yet the sequence contains 8, 23 or many more unique values. The mixture explanation is a hypothesis; verifying stable conditional $Q$ moments is a separate check. Multiple branches can reflect different components, transformations or regimes. We have not identified physical market semantics from this geometry.
    """)
    return


@app.cell(hide_code=True)
def _(RUN, mo):
    mo.vstack([
        mo.md("""## Actual raw traces and prefix paths

    The examples are selected on geometry, so they illustrate structure rather than estimate prevalence."""),
        mo.image(RUN / "scan_r1/support_traces.png"),
        mo.image(RUN / "scan_r1/prefix_paths.png"),
    ])
    return


@app.cell(hide_code=True)
def _(RUN, json, mo):
    _results = json.loads((RUN / "pilot_results_r1.json").read_text())
    _pilot_table = [{"arm": _r["arm"], "seed": _r["seed"], "pooled_wp": _r["evaluation"]["wp"], "fold0_wp": _r["evaluation_by_fold"]["0"]["wp"], "fold1_wp": _r["evaluation_by_fold"]["1"]["wp"], "updates": _r["training"]["updates"], "cuda_peak_MB": _r["training"]["cuda_peak_allocated_bytes"] / 1e6} for _r in _results]
    mo.vstack([
        mo.md("""## First causal RNN pilot — all paired seeds

    Three matched-capacity arms, H32×1,43fitting sequences, fixed2epochs/880updates. Official scores were independently recomputed with the frozen scorer. Mean pooled moments-control delta is **+0.000883**, but fold0 worsens and fold1 improves. There are only8components per fold, validation is exposed and this is not a production-gain claim. All conditional component intervals include zero; that does not close the mechanism.

    The discrete/heuristic direction remains active. The follow-up separates dictionary/event/dwell channels and their interaction with moments at two capacities; student distillation itself has not run.
    """),
        mo.ui.table(_pilot_table),
    ])
    return


if __name__ == "__main__":
    app.run()
