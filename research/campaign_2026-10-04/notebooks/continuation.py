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

    Public incumbent: **0.6655**, confirmed by user-uploaded E412 (`10tq248b`): **+0.0014** over 0.6641, runtime **37m47s**, memory **625MB**. The target gap is **0.0005**. The completed local comparison gained **0.00241** on exposed folds; mechanism attribution remains unresolved.

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
    combo_all_path = campaign_dir / "checks/combo_merged_all_folds_report.json"
    combo_paths = [combo_all_path] if combo_all_path.exists() else sorted((campaign_dir / "checks").glob("combo_merged_f*_report.json"))
    combo_rows = []
    for _path in combo_paths:
        _report = json.loads(_path.read_text())
        for _fold, _fold_report in _report["folds"].items():
            _contrast = _fold_report["paired_component_bootstrap"]["COMBO_minus_CTRL"]
            combo_rows.append({"fold": _fold, "delta": _contrast["mean_diff"], "low": _contrast["ci95"][0], "high": _contrast["ci95"][1]})
    mo.vstack([mo.md("### Complete predictor: four exposed folds\nEqual parent/head mixtures. Pooled delta +0.002414; component 95% interval [+0.001566, +0.003449]. Fresh CTRL head tensors reproduce exactly. Intervals condition on these fitted models; no public-score claim."), mo.ui.table(combo_rows)])
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


@app.cell(hide_code=True)
def _(campaign_dir, json, mo, plt):
    q8_result = json.loads((campaign_dir / "E411_q8_model/result.json").read_text())
    q8_rows = [{"preset": _name.replace("_", " "), "delta": _case["global_wp_delta"], "low": _case["paired_component_bootstrap"]["ci95"][0], "high": _case["paired_component_bootstrap"]["ci95"][1]} for _name, _case in q8_result["cases"].items() if "global_wp_delta" in _case]
    q8_fig, q8_ax = plt.subplots(figsize=(8, 4))
    q8_ax.errorbar([row["preset"] for row in q8_rows], [row["delta"] for row in q8_rows], yerr=[[row["delta"]-row["low"] for row in q8_rows], [row["high"]-row["delta"] for row in q8_rows]], fmt="o", capsize=5, color="#8F5C38")
    q8_ax.axhline(0, color="black", linewidth=0.8)
    q8_ax.set(ylabel="Global WP difference from float weights", title="E411: fixed q8 weight representations on 64 components")
    q8_ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    q8_ax.grid(axis="y", alpha=0.2)
    q8_fig.tight_layout()
    _q8_export = campaign_dir / "eda/quantization/plots"
    _q8_export.mkdir(parents=True, exist_ok=True)
    q8_fig.savefig(_q8_export / "weight_degradation.png", dpi=180)
    q8_fig.savefig(_q8_export / "weight_degradation.svg")
    (_q8_export.parent / "data.json").write_text(json.dumps(q8_rows, indent=2) + "\n")
    mo.vstack([mo.md("### Q8 accuracy pilot\nBoth fixed presets are reported. Float32 arithmetic evaluates represented byte weights; compressed-kernel speed remains unmeasured."), q8_fig])
    return


@app.cell(hide_code=True)
def _(campaign_dir, json, mo, refresh):
    refresh
    catalog_view = json.loads((campaign_dir / "experiment_catalog.json").read_text())
    mo.vstack([mo.md("### Research inventory\nHistorical negatives remain eligible for bounded retests. Implementation checks and scientific results are tracked separately."), mo.ui.table([{ "id": card["id"], "experiment": card["name"], "mechanism": card["mechanism_family"], "implementation": card["implementation_status"], "evidence_status": card["status"]} for card in catalog_view["cards"]], selection=None, page_size=12)])
    return


@app.cell(hide_code=True)
def _(campaign_dir, json, mo, refresh):
    refresh
    worker_rows = []
    for _worker_path in sorted((campaign_dir / "runner/attempts").glob("*.json")):
        _worker = json.loads(_worker_path.read_text())
        worker_rows.append({"attempt": _worker["id"], "status": _worker["status"], "kind": _worker["kind"], "peak_RAM_GiB": round(_worker.get("peak_ram_bytes", 0)/1024**3, 2), "peak_VRAM_GB": round(_worker.get("peak_observed_vram_bytes", 0)/1e9, 2), "OOM_kills": _worker.get("final_memory_events", _worker.get("memory_events", {})).get("oom_kill", 0), "failure": _worker.get("failure", "")})
    mo.vstack([mo.md("### Local execution evidence\nRefresh reads current durable worker records. RAM includes filesystem cache. Full logs, frozen sources and limits are retained with each attempt."), mo.ui.table(worker_rows, selection=None, page_size=10)])
    return


@app.cell(hide_code=True)
def _(campaign_dir, json, mo, plt, refresh):
    refresh
    feature_pooled = json.loads((campaign_dir / "checks/e415_pooled_v1.json").read_text())
    feature_rows = []
    for _cue, _case in feature_pooled["families"].items():
        for _reference in ("capacity", "unchanged"):
            _contrast = _case["pooled"]["candidate_minus_" + _reference]
            feature_rows.append({"cue": _cue, "reference": _reference, "delta": _contrast["mean_diff"], "low": _contrast["ci95"][0], "high": _contrast["ci95"][1]})
    feature_fig, feature_axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
    for _ax, _reference in zip(feature_axes, ("capacity", "unchanged")):
        _rows = [row for row in feature_rows if row["reference"] == _reference]
        _ax.errorbar([row["delta"] for row in _rows], range(len(_rows)), xerr=[[row["delta"]-row["low"] for row in _rows], [row["high"]-row["delta"] for row in _rows]], fmt="o", capsize=3, color="#146C94" if _reference == "capacity" else "#A05235")
        _ax.axvline(0, color="black", linewidth=0.8)
        _ax.set_yticks(range(len(_rows)), [row["cue"].replace("_", " ") for row in _rows])
        _ax.set(xlabel="Selected-row Global WP difference", title="Candidate minus " + _reference)
        _ax.grid(axis="x", alpha=0.2)
    feature_axes[0].set_xticks([-0.005, 0, 0.005, 0.01, 0.015])
    feature_axes[1].set_xticks([-0.04, -0.03, -0.02, -0.01, 0])
    feature_axes[0].invert_yaxis()
    feature_fig.suptitle("E415: relative control wins do not improve the untouched GRU")
    feature_fig.tight_layout()
    _feature_export = campaign_dir / "eda/conditional_features/plots"
    _feature_export.mkdir(parents=True, exist_ok=True)
    feature_fig.savefig(_feature_export / "proxy_heads.png", dpi=180)
    feature_fig.savefig(_feature_export / "proxy_heads.svg")
    (_feature_export.parent / "data.json").write_text(json.dumps(feature_rows, indent=2) + "\n")
    mo.vstack([mo.md("### Conditional-feature proxy screen\nAll nine families, both reciprocal directions. 6,158 rows / 460 sequences / 140 components. Every fitted candidate loses to the untouched predictor; exact feature mechanisms remain separate experiments. Component intervals condition on these fitted heads and are descriptive on exposed data."), feature_fig])
    return


@app.cell(hide_code=True)
def _(campaign_dir, json, mo, plt):
    q8_timing = json.loads((campaign_dir / "E416_native_q8_confirmation/result.json").read_text())["summary"]
    q8_timing_rows = [{"case": name, "streamed_mb": item["weight_bytes"] / 1e6, **item["us_per_row"]} for name, item in q8_timing.items()]
    q8_timing_fig, q8_timing_axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for _ax, _width in zip(q8_timing_axes, ("d256", "w384")):
        _rows = [row for row in q8_timing_rows if row["case"].startswith(_width)]
        for _row, _color in zip(_rows, ("#507D9C", "#A36B43", "#55805E")):
            _ax.errorbar(_row["streamed_mb"], _row["median"], yerr=[[_row["median"]-_row["min"]], [_row["max"]-_row["median"]]], fmt="o", color=_color, capsize=4, label=_row["case"].split("_", 1)[1])
        _ax.set(xlabel="Streamed weights (decimal MB)", ylabel="Microseconds per row", title=_width.upper() + " fixed trained payload")
        _ax.legend()
        _ax.grid(alpha=.2)
    q8_timing_fig.suptitle("E416: fewer weight bytes did not reduce elapsed time")
    q8_timing_fig.tight_layout()
    _q8_timing_export = campaign_dir / "eda/native_byte_timing/plots"
    _q8_timing_export.mkdir(parents=True, exist_ok=True)
    q8_timing_fig.savefig(_q8_timing_export / "bytes_vs_time.png", dpi=180)
    q8_timing_fig.savefig(_q8_timing_export / "bytes_vs_time.svg")
    (_q8_timing_export.parent / "data.json").write_text(json.dumps(q8_timing_rows, indent=2) + "\n")
    mo.vstack([mo.md("All six trained cases passed their fixed numerical gates on three full streams. Whiskers show the range of five clean alternating trials on the local CPU. D256 uses the real merged head; W384 uses one head. This is not a public-runtime forecast."), q8_timing_fig])
    return


@app.cell(hide_code=True)
def _(campaign_dir, json, mo, plt):
    reset_result = json.loads((campaign_dir / "feature_reset_results/f0_gpu_v1/report.json").read_text())
    reset_rows = [{"lag": int(lag), **report["candidate_minus_unchanged"]} for lag, report in reset_result["reports"].items()]
    reset_fig, reset_ax = plt.subplots(figsize=(8, 4))
    reset_ax.errorbar([str(row["lag"]) for row in reset_rows], [row["mean_diff"] for row in reset_rows], yerr=[[row["mean_diff"]-row["ci95"][0] for row in reset_rows], [row["ci95"][1]-row["mean_diff"] for row in reset_rows]], fmt="o", capsize=5, color="#146C94")
    reset_ax.axhline(0, color="black", linewidth=.8)
    reset_ax.set(xlabel="State reset every N rows after warmup", ylabel="Reset minus unchanged Global WP", title="E417: fixed-weight state disruption on a small exposed panel")
    reset_ax.grid(axis="y", alpha=.2)
    reset_fig.tight_layout()
    _reset_dir = campaign_dir / "eda/reset_state/plots"
    _reset_dir.mkdir(parents=True, exist_ok=True)
    reset_fig.savefig(_reset_dir / "reset_curve.png", dpi=180)
    reset_fig.savefig(_reset_dir / "reset_curve.svg")
    (_reset_dir.parent / "data.json").write_text(json.dumps(reset_rows, indent=2) + "\n")
    mo.vstack([mo.md("498 scored rows,32 streams,15 components. Intervals are paired component bootstraps conditional on fixed weights and rows. Features retain their complete causal histories. This measures state disruption, not useful gradient length."), reset_fig])
    return


@app.cell(hide_code=True)
def _(campaign_dir, json, mo, plt):
    literal_result = json.loads((campaign_dir / "checks/e418_pooled_v1.json").read_text())
    literal_rows = [{"family": family, **scores["candidate_minus_unchanged"]} for family, scores in literal_result["families"].items()]
    literal_fig, literal_ax = plt.subplots(figsize=(10, 5))
    literal_ax.errorbar([row["mean_diff"] for row in literal_rows], range(len(literal_rows)), xerr=[[row["mean_diff"]-row["ci95"][0] for row in literal_rows], [row["ci95"][1]-row["mean_diff"] for row in literal_rows]], fmt="o", capsize=4, color="#146C94")
    literal_ax.set_yticks(range(len(literal_rows)), [row["family"].replace("_", " ") for row in literal_rows])
    literal_ax.axvline(0, color="black", linewidth=.8)
    literal_ax.set(xlabel="Literal residual head minus untouched Global WP", title="E418: all fixed literal head candidates on reciprocal components")
    literal_ax.grid(axis="x", alpha=.2)
    literal_fig.tight_layout()
    _literal_dir = campaign_dir / "eda/literal_features/plots"
    _literal_dir.mkdir(parents=True, exist_ok=True)
    literal_fig.savefig(_literal_dir / "all_candidates.png", dpi=180)
    literal_fig.savefig(_literal_dir / "all_candidates.svg")
    (_literal_dir.parent / "data.json").write_text(json.dumps(literal_rows, indent=2) + "\n")
    mo.vstack([mo.md("6158 fixed scored rows,460 streams,140 exposed components; fitted heads exchange disjoint component partitions. All eight candidates lose to unchanged predictions. These are literal frozen-state screens; they do not test training the full trunk with these features. Intervals condition on fitted models and sampled rows; no family-search correction."), literal_fig])
    return


@app.cell(hide_code=True)
def _():
    import json as e413_json
    from pathlib import Path as E413Path
    import matplotlib.pyplot as e413_plt
    _e413_root = E413Path('/home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-04')
    _e413_result = e413_json.loads((_e413_root/'checks/e413_paired_report.json').read_text())
    _e413_p = _e413_result['folds']['0']['paired_component_bootstrap']['HISTORY_minus_CURRENT']
    _e413_classes = _e413_result['folds']['0']['by_class']
    _e413_deltas = [_e413_classes[str(k)]['HISTORY']['wp']-_e413_classes[str(k)]['CURRENT']['wp'] for k in range(4)]
    _e413_fig, _e413_axes = e413_plt.subplots(1,2,figsize=(10,3.5),layout='constrained')
    _e413_axes[0].errorbar([_e413_p['mean_diff']],[0],xerr=[[_e413_p['mean_diff']-_e413_p['ci95'][0]],[_e413_p['ci95'][1]-_e413_p['mean_diff']]],fmt='o',capsize=5)
    _e413_axes[0].axvline(0,color='gray',linewidth=1)
    _e413_axes[0].set(yticks=[0],yticklabels=['History − current'],xlabel='Global WP difference',title='Paired component interval')
    _e413_axes[1].bar(range(4),_e413_deltas)
    _e413_axes[1].axhline(0,color='gray',linewidth=1)
    _e413_axes[1].set(xticks=range(4),xlabel='Class',ylabel='WP difference',title='Descriptive class differences')
    _e413_fig.suptitle('E413: volume history, one exposed fold / seed3')
    _e413_out = _e413_root/'eda/volume_history/plots'
    _e413_out.mkdir(parents=True,exist_ok=True)
    for _e413_ext in ('png','svg'):
        _e413_fig.savefig(_e413_out/f'paired_volume.{_e413_ext}',dpi=180)
    (_e413_out/'data.json').write_text(e413_json.dumps({'pooled':_e413_p,'class_deltas':_e413_deltas},indent=2)+'\n')
    _e413_fig
    return


@app.cell(hide_code=True)
def _():
    import json as _gain_json
    from pathlib import Path as _gain_Path
    import matplotlib.pyplot as _gain_plt
    _gain_root = _gain_Path("/home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-04")
    _gain_result = _gain_json.loads((_gain_root / "checks/e422_reciprocal_report.json").read_text())
    _gain_dir = _gain_root / "eda/gain_probe/plots"
    _gain_dir.mkdir(parents=True, exist_ok=True)
    _gain_fig, _gain_axes = _gain_plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for _gain_d, _gain_report in enumerate(_gain_result["directions"]):
        for _gain_arm, _gain_color in [("candidate", "#176fa5"), ("input_summary", "#d78720")]:
            _gain_values = _gain_report["recoverability"][_gain_arm]["r2_vs_fit_mean_per_target"]
            _gain_axes[0].plot([2 * _gain_d, 2 * _gain_d + 1], _gain_values, "o-", color=_gain_color, label=_gain_arm if _gain_d == 0 else None)
        _gain_metric = _gain_report["reports"]["candidate_minus_unchanged"]
        _gain_mean = _gain_metric["mean_diff"]
        _gain_lo, _gain_hi = _gain_metric["ci95"]
        _gain_axes[1].errorbar(_gain_d, _gain_mean, yerr=[[_gain_mean - _gain_lo], [_gain_hi - _gain_mean]], fmt="o", color="#176fa5", capsize=5)
    _gain_axes[0].set_xticks(range(4), ["A→B t0", "A→B t1", "B→A t0", "B→A t1"])
    _gain_axes[0].set_ylabel("Gain R² versus fit mean")
    _gain_axes[0].legend(labels=["Hidden state", "Input summary"])
    _gain_axes[1].set_xticks([0, 1], ["A→B", "B→A"])
    _gain_axes[1].set_ylabel("State gain minus unchanged WP")
    for _gain_ax in _gain_axes:
        _gain_ax.axhline(0, color="black", linewidth=0.8)
    _gain_fig.suptitle("E422: fixed gain probe on disjoint episodes\nExposed fold 0; component intervals; no architecture fit")
    for _gain_suffix in ["png", "svg"]:
        _gain_fig.savefig(_gain_dir / f"gain_probe.{_gain_suffix}", dpi=160)
    _gain_fig

    return


app._unparsable_cell(
    """
    refresh
    _progress_state = json.loads((campaign_dir / \"implementation_state.json\").read_text())
    _progress_inv = json.loads((campaign_dir / \"checks/kaggle_kernel_inventory_20261004T1138Z.json\").read_text())
    _progress_ledger = [json.loads(line) for line in (project_root / \"research/experiments.jsonl\").read_text().splitlines() if line.strip()]
    _progress_rows = []
    for _row in _progress_ledger:
        _eid = str(_row.get(\"id\", \"\"))
        if _row.get(\"date\") != \"2026-10-04\" or not _eid.startswith(\"E-\"):
            continue
        try:
            _number = int(_eid.split(\"-\")[1])
        except ValueError:
            continue
        if 407 <= _number <= 424:
            _progress_rows.append({\"experiment\": _eid, \"status\": _row.get(\"status\", \"missing\"), \"question\": _row.get(\"kind\", \"\")})
    _progress_counts = _progress_state[\"campaign_progress\"]
    _kaggle_rows = [{\"kernel\": _row[\"slug\"], \"provider\": _row[\"status\"], \"result / current observation\": _row[\"result\"][:180]} for _row in _progress_inv[\"kernels\"]]
    mo.vstack([
        mo.md(f\"\"\"## Live run and compute status

    **Ledger:** {_progress_counts['completed']}/{_progress_counts['total']} complete; {_progress_counts['partial']} partial; {_progress_counts['queued']} queued; {_progress_counts.get('prepared', 0)} prepared.

    **Immediate local tranche:** E-424 has 8 full-recipe model-fit arms (4 primary contrasts, then 4 router/adapter controls). Its 30-GiB spec was rejected before worker admission; corrected v2 specs use the enforced24-GiB cap.

    **Compute:** local block {_progress_state['local_block_gpu_job_hours']}h; 20.44h remain. Kaggle {_progress_inv['quota']['gpu_remaining_hours']:.2f}/30 GPU-h and {_progress_inv['quota']['tpu_remaining_hours']}/20 TPU-h remain. E-419 v4 is **RUNNING** without worker logs/peaks yet; the TPU probe remains **QUEUED**.

    **80-GPU-hour plan:** cumulative across providers and refreshed blocks, not an 80-hour Kaggle reservation. The staged first slate is nine programs / about28 new fitted models; continue to add independent replications and open review ideas as they become executable.

    Refresh this cell with the button after the saved status snapshots change.\"\"\"),
        mo.ui.table(_progress_rows, selection=None, page_size=10),
        mo.md(\"### Kaggle notebook audit (provider state is not scientific completion)\"),
        mo.ui.table(_kaggle_rows, selection=None, page_size=8),
    ])
    """,
    column=None, disabled=False, hide_code=True, expand_output=False, name="Live experiment and compute status"
)


@app.cell(hide_code=True)
def _(campaign_dir, json, mo, refresh):
    refresh
    _validation_card = json.loads((campaign_dir / "checks/e459_validation_scorecard_v1.json").read_text())
    _validation_rows = []
    for _name, _scores in _validation_card["scores"].items():
        _validation_rows.append(f"| {_name} | {_scores['all_supplied_validation']['wp']:.6f} | {_scores['excluded_from_final_fit']['wp']:.6f} |")
    mo.md("Whole supplied datasets/valid.parquet (1,873 sequences), fixed cached predictions:\n\n| Predictor | All validation | 368 excluded from final fit |\n|---|---:|---:|\n" + "\n".join(_validation_rows) + "\n\nRecipe CV pooled combo/control: 0.693658 / 0.691244. Full validation includes fitted sequences. The combo uses FP32 exports; exact native replay is running. See VALIDATION_SCORECARD.md.")
    return


if __name__ == "__main__":
    app.run()
