import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    from pathlib import Path
    import marimo as mo
    import matplotlib.pyplot as plt
    project_root = Path.cwd()
    campaign_dir = project_root / "research/campaign_2026-10-04"
    return campaign_dir, json, mo, plt, project_root


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Connectome continuation

    Public incumbent: **0.6641**. New implementation is underway; no new forecasting gain is claimed.

    This notebook reads durable campaign records. It never starts training through reactive cells. Historical folds are exposed; public-mixture estimates are diagnostics.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    refresh = mo.ui.run_button(label="Refresh saved evidence")
    refresh
    return (refresh,)


@app.cell(hide_code=True)
def _(campaign_dir, json, mo, refresh):
    refresh
    implementation = json.loads((campaign_dir / "implementation_state.json").read_text())
    mo.json(implementation)
    return


@app.cell(hide_code=True)
def _(json, mo, project_root):
    registry = json.loads((project_root / "research/lb/registry.json").read_text())["subs"]
    combo_evidence = json.loads((project_root / "research/campaign_2026-10-03/continuation/FOLD23_DECISION.json").read_text())
    mo.ui.table([{ "candidate": row["name"], "date": row["date"], "public_score": row["public"]} for row in registry], selection=None)
    return (registry,)


@app.cell(hide_code=True)
def _(plt, registry):
    public_fig, public_ax = plt.subplots(figsize=(10, 4))
    public_ax.plot(range(len(registry)), [row["public"] for row in registry], marker="o", linewidth=1.5)
    public_ax.axhline(0.6660, color="tab:orange", linestyle="--", label="Milestone (not achieved)")
    public_ax.set_xticks(range(len(registry)), [row["name"] for row in registry], rotation=55, ha="right", fontsize=7)
    public_ax.set_ylabel("Recorded public Global WP")
    public_ax.set_title("Historical uploads — before the continuation")
    public_ax.grid(axis="y", alpha=0.25)
    public_ax.legend()
    public_fig.tight_layout()
    public_fig
    return (public_fig,)


@app.cell(hide_code=True)
def _(mo):
    save_history = mo.ui.run_button(label="Save historical evidence figure")
    save_history
    return (save_history,)


@app.cell(hide_code=True)
def _(campaign_dir, json, mo, public_fig, registry, save_history):
    mo.stop(not save_history.value, mo.md("Use the button to export the historical plot and its source table."))
    history_dir = campaign_dir / "eda/historical_public/plots"
    history_dir.mkdir(parents=True, exist_ok=True)
    public_fig.savefig(history_dir / "public_history.png", dpi=160, bbox_inches="tight")
    public_fig.savefig(history_dir / "public_history.svg", bbox_inches="tight")
    (history_dir.parent / "data.json").write_text(json.dumps(registry, indent=2) + "\n")
    mo.md("Saved PNG, SVG, and the source table in `eda/historical_public/`.")
    return


@app.cell(hide_code=True)
def _(campaign_dir, json, mo, refresh):
    refresh
    combo_paths = sorted((campaign_dir / "checks").glob("combo_merged_f*_report.json"))
    combo_rows = []
    for _path in combo_paths:
        _report = json.loads(_path.read_text())
        for _fold, _fold_report in _report["folds"].items():
            _contrast = _fold_report["paired_component_bootstrap"]["COMBO_minus_CTRL"]
            combo_rows.append({"fold": _fold, "delta": _contrast["mean_diff"], "low": _contrast["ci95"][0], "high": _contrast["ci95"][1]})
    mo.vstack([mo.md("### Complete predictor: provisional exposed-fold results\nEqual parent/head mixtures. Component intervals condition on these fitted models; no public-score claim."), mo.ui.table(combo_rows)])
    return (combo_rows,)


@app.cell(hide_code=True)
def _(campaign_dir, combo_rows, json, mo, plt):
    mo.stop(not combo_rows, mo.md("Awaiting the first fixed fold score."))
    combo_fig, combo_ax = plt.subplots(figsize=(8, 4))
    combo_ax.errorbar([row["fold"] for row in combo_rows], [row["delta"] for row in combo_rows], yerr=[[row["delta"] - row["low"] for row in combo_rows], [row["high"] - row["delta"] for row in combo_rows]], fmt="o", capsize=5, color="#146C94")
    combo_ax.axhline(0, color="black", linewidth=0.8)
    combo_ax.set(xlabel="Previously exposed evaluation fold", ylabel="Global WP: COMBO minus CTRL", title="Fixed equal parent/head mixtures; component bootstrap")
    combo_ax.grid(axis="y", alpha=0.2)
    combo_fig.tight_layout()
    _combo_export = campaign_dir / "eda" / "complete_predictor" / "plots"
    _combo_export.mkdir(parents=True, exist_ok=True)
    _combo_label = "folds_" + "_".join(row["fold"] for row in combo_rows)
    combo_fig.savefig(_combo_export / (_combo_label + ".png"), dpi=180)
    combo_fig.savefig(_combo_export / (_combo_label + ".svg"))
    (_combo_export.parent / (_combo_label + ".json")).write_text(json.dumps(combo_rows, indent=2) + "\n")
    combo_fig
    return


@app.cell(hide_code=True)
def _(campaign_dir, json, plt):
    width_report = json.loads((campaign_dir / "E409_width_runtime" / "result.json").read_text())
    width_rows = [{"payload": _name, **_values["kcycles_per_row"]} for _name, _values in width_report["summary"].items()]
    width_fig, width_ax = plt.subplots(figsize=(9, 4))
    width_ax.bar([row["payload"].replace("_", " ") for row in width_rows], [row["median"] for row in width_rows], color=["#507D9C", "#507D9C", "#A36B43", "#A36B43"], yerr=[[row["median"]-row["min"] for row in width_rows], [row["max"]-row["median"] for row in width_rows]], capsize=4)
    width_ax.set(ylabel="Thousand CPU cycles per row", title="E409: measured native q16x payload cost")
    width_ax.text(0.01, -0.2, "Head2 duplicates one head for sizing. Whiskers: min/max of five trials. No public-time conversion.", transform=width_ax.transAxes, fontsize=9)
    width_fig.tight_layout()
    _width_export = campaign_dir / "eda" / "width_runtime" / "plots"
    _width_export.mkdir(parents=True, exist_ok=True)
    width_fig.savefig(_width_export / "cycles.png", dpi=180, bbox_inches="tight")
    width_fig.savefig(_width_export / "cycles.svg", bbox_inches="tight")
    (_width_export.parent / "data.json").write_text(json.dumps(width_rows, indent=2) + "\n")
    width_fig
    return


if __name__ == "__main__":
    app.run()
