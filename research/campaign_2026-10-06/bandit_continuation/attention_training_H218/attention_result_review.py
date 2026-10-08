import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import hashlib
    import json
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    h218_dir = (
        Path.cwd() / "research/campaign_2026-10-06/bandit_continuation/attention_training_H218"
    )
    h218_report_path = h218_dir / "H218_FOLD0_REFERENCE_BLEND_20261008_R1.json"
    h218_report_bytes = h218_report_path.read_bytes()
    assert (
        hashlib.sha256(h218_report_bytes).hexdigest()
        == "a6e487c1d91459a523c2bca956ae42b9ab679ae0b46833ea11aaf63123040c62"
    )
    h218_report = json.loads(h218_report_bytes)
    h218_plots = h218_dir / "plots"
    h218_plots.mkdir(exist_ok=True)

    return h218_plots, h218_report, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Block-memory attention: fixed-checkpoint result

    This notebook reads only the frozen independent grade report. It launches no fits and does not open labels or predictions.

    Own scratch block-memory model, fixed epoch 3, versus the independently trained GRU reference on 468 previously exposed evaluation groups. The recipes and training ancestry differ. The intervals describe component resampling of fixed predictions; they do not forecast public score.

    All five blend weights were registered before candidate predictions. This exact recipe and the tested blends showed no gain; other attention or teacher interactions remain untested.
    """)

    return


@app.cell
def _(h218_plots, h218_report, mo, np, plt):
    _rows = h218_report["predeclared_blends_all_reported"]
    _alpha = np.array([_r["alpha_on_h218"] for _r in _rows])
    _delta = np.array([_r["delta_vs_reference"]["wp"] for _r in _rows])
    _ci = np.array([_r["paired_component_bootstrap_vs_reference"]["ci95"] for _r in _rows])
    h218_curve, _ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    _ax.axhline(0, color="0.4", linewidth=1)
    _ax.errorbar(
        _alpha,
        _delta,
        yerr=np.maximum(0, np.vstack((_delta - _ci[:, 0], _ci[:, 1] - _delta))),
        fmt="o-",
        capsize=4,
        color="#386cb0",
    )
    _ax.set(
        xlabel="Fixed weight on block-memory attention",
        ylabel="Official local WP difference vs GRU",
        title="Preregistered blends on 468 exposed groups",
    )
    _ax.grid(alpha=0.2)
    for _extension in ("png", "svg"):
        h218_curve.savefig(h218_plots / f"fixed_blend_curve_20261008_r1.{_extension}", dpi=160)
    mo.as_html(h218_curve)

    return


@app.cell
def _(h218_plots, h218_report, mo, plt):
    _relations = h218_report["pair_prediction_covariance_correlation"]["per_target"]
    h218_correlations, _ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    _targets = list(_relations)
    _values = [_relations[_target]["weighted_pearson_correlation"] for _target in _targets]
    _ax.bar(_targets, _values, color=["#386cb0", "#7fc97f"])
    _ax.set(
        ylim=(0, 1),
        ylabel="Weighted prediction correlation",
        title="Attention versus own GRU reference",
    )
    for _target, _value in zip(_targets, _values, strict=True):
        _ax.text(_target, _value - 0.07, f"{_value:.5f}", ha="center")
    for _extension in ("png", "svg"):
        h218_correlations.savefig(
            h218_plots / f"prediction_correlation_20261008_r1.{_extension}", dpi=160
        )
    mo.as_html(h218_correlations)

    return


if __name__ == "__main__":
    app.run()
