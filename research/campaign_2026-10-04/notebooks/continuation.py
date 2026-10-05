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
    return Path, campaign_dir, json, mo, plt, project_root


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Connectome continuation — resumed October 5

    Current public best: **0.6660**, fixed seed-4 complete combo, 37m07s /626 MB.
    User authorized local GPU and Kaggle GPU/TPU research again. Only the October 4
    controlled queue is reopened; historic pause and failure records are preserved.

    Next: eight fixed architecture heads; W384 fold-1 matched replication;
    aligned/stale/sham causal volume gating; existing TPU preflight then distributed
    FP32/BF16 continuation after actual-device checks. No competition uploads.

    Local ceiling: 12 CPUs /36 GiB aggregate RAM /5 GB observed VRAM, one GPU worker.
    Kaggle allocation preserves memory and quota reserve. Exposed validation is a
    conditional comparison, not a public-score forecast.
    """)
    return


@app.cell(hide_code=True)
def _(json, mo, project_root, refresh):
    refresh
    _oct5_root = project_root / "research/campaign_2026-10-05"
    _oct5_records = []
    for _oct5_path in sorted((_oct5_root / "checks").glob("*.json")):
        _oct5_payload = json.loads(_oct5_path.read_text())
        _oct5_records.append({"record": _oct5_path.name, "status": _oct5_payload.get("status", "receipt"), "utc": _oct5_payload.get("utc", ""), "details": _oct5_payload})
    mo.vstack([mo.md("## October 5 resumed experiments"), mo.ui.table(_oct5_records)])
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
    _progress_ledger = [json.loads(line) for line in (project_root / \"research/experiments.jsonl\").read_text().splitlines() if line.strip()]
    _progress_rows = [{\"experiment\": _row[\"id\"], \"status\": _row.get(\"status\", \"missing\"), \"question\": _row.get(\"kind\", \"\")} for _row in _progress_ledger if _row.get(\"date\") == \"2026-10-04\"]
    _progress_counts = _progress_state[\"campaign_progress\"]
    _progress_quota = _progress_state[\"kaggle_quota\"]
    _progress_e424_fits = []
    for _path in (campaign_dir / \"runner/attempts\").glob(\"e424-h*-v2.json\"):
        _record = json.loads(_path.read_text())
        _progress_e424_fits.append({\"attempt\": _record[\"id\"], \"state\": _record[\"status\"], \"exit\": _record.get(\"exit_code\"), \"peak GPU GB\": round((_record.get(\"peak_observed_vram_bytes\") or 0) / 1e9, 3)})
    _progress_e424_complete = sum(_row[\"state\"] == \"finished\" and _row[\"exit\"] == 0 for _row in _progress_e424_fits)
    mo.vstack([
        mo.md(f\"\"\"## Research and compute status

    Snapshot: {_progress_state['as_of_utc']}. Original E407–427 programs: **{_progress_counts['completed']}/{_progress_counts['total']} scientifically complete**, {_progress_counts['partial']} partial, {_progress_counts['running']} running, {_progress_counts['queued']} queued, {_progress_counts['prepared']} prepared. Tests and source preparation do not count as completed scientific fits.

    **E424 model fits: {_progress_e424_complete}/8 finished.** Seed4 replication is durably queued after this controller's successful completion. Primary paired architecture intervals include zero; remaining controls still matter.

    **Remote:** {_progress_state['active_remote']}

    Last confirmed provider quota ({_progress_quota['as_of_utc']}): {_progress_quota['gpu_remaining_hours']} GPU-h and {_progress_quota['tpu_remaining_hours']} TPU-h remaining. These are stale snapshots during live jobs, distinct from cumulative device-time accounting.

    Accepted plan: **80 cumulative device-GPU-hours** across local and Kaggle, TPU separate; local one GPU trainer, 12 CPUs/36 GiB/no swap, observed VRAM stop4.8GB. Continue until explicit user stop within admitted budgets. All review/retest programs are listed below, including unimplemented ones.
    \"\"\"),
        mo.ui.table(_progress_e424_fits, selection=None),
        mo.ui.table(_progress_rows, selection=None, page_size=12),
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
    mo.md("Whole supplied datasets/valid.parquet (1,873 sequences), fixed cached predictions:\n\n| Predictor | All validation | 368 excluded from final fit |\n|---|---:|---:|\n" + "\n".join(_validation_rows) + "\n\nRecipe CV pooled combo/control: 0.693658 / 0.691244. Full validation includes fitted sequences. The combo uses FP32 exports; exact submitted i16 ZIP replay is complete: WP 0.700503766, t0 0.696316661, t1 0.704690871 (E460). See VALIDATION_SCORECARD.md.")
    return


@app.cell(hide_code=True)
def _(campaign_dir, json, mo, plt, project_root, refresh):
    refresh
    import hashlib as _research_hashlib
    _research_e424_path = campaign_dir / "checks/e424_report_core_fp32.json"
    _research_e419_path = campaign_dir / "checks/e419_v4_recovery/e419_recovered_paired_report.json"
    mo.stop(not (_research_e424_path.exists() and _research_e419_path.exists()), mo.md("Awaiting paired saved reports."))
    _research_e424 = json.loads(_research_e424_path.read_text())
    _research_e419 = json.loads(_research_e419_path.read_text())
    _research_panels = [
        ("Specialist / adapter, 368 exposed sequences", _research_e424, [
            ("Learned specialist − matched GRU", "H093_LEARNED_minus_H093_GRU"),
            ("Conditional adapter − matched GRU", "H094_CONDITIONAL_minus_H094_GRU"),
            ("Specialist parent − complete combo", "H093_LEARNED_minus_COMBO"),
            ("Adapter parent − complete combo", "H094_CONDITIONAL_minus_COMBO")]),
        ("Quantization, exposed fold 0 (468 sequences)", _research_e419, [
            ("Recurrent QAT − FP32 continuation", "QAT_REC_minus_FP32_REC"),
            ("Recurrent/head QAT − FP32 continuation", "QAT_REC_HEAD_minus_FP32_REC_HEAD"),
            ("Recurrent PTQ − FP32 continuation", "FP32_REC_PTQ_minus_FP32_REC"),
            ("Recurrent/head PTQ − FP32 continuation", "FP32_REC_HEAD_PTQ_minus_FP32_REC_HEAD")])]
    _research_fig, _research_axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    _research_rows = []
    for _research_ax, (_research_title, _research_report, _research_items) in zip(_research_axes, _research_panels, strict=True):
        for _research_index, (_research_label, _research_key) in enumerate(_research_items):
            _research_value = _research_report["pooled"]["paired_component_bootstrap"][_research_key]
            _research_delta = _research_value["mean_diff"]
            _research_low, _research_high = _research_value["ci95"]
            _research_ax.errorbar(_research_delta, _research_index, xerr=[[_research_delta-_research_low], [_research_high-_research_delta]], fmt="o", capsize=4)
            _research_rows.append({"comparison": _research_label, "delta": _research_delta, "ci95": [_research_low, _research_high], "sequences": _research_value["n_sequences"], "components": _research_value["n_components"]})
        _research_ax.set_yticks(range(len(_research_items)), [_research_label for _research_label, _research_key in _research_items])
        _research_ax.axvline(0, color="black", linewidth=1)
        _research_ax.invert_yaxis()
        _research_ax.set_title(_research_title)
        _research_ax.set_xlabel("Global WP difference; paired component 95% interval (2,000 draws)")
        _research_ax.grid(axis="x", alpha=.2)
    _research_fig.suptitle("New experiments: uncertainty remains; no score-gain promotion")
    _research_plot_dir = campaign_dir / "eda/paired_research_update"
    _research_plot_dir.mkdir(parents=True, exist_ok=True)
    _research_fig.savefig(_research_plot_dir / "paired_results.png", dpi=160)
    _research_fig.savefig(_research_plot_dir / "paired_results.svg")
    (_research_plot_dir / "source.json").write_text(json.dumps({"reports": [{"path": str(_research_path.relative_to(project_root)), "sha256": _research_hashlib.sha256(_research_path.read_bytes()).hexdigest()} for _research_path in [_research_e424_path, _research_e419_path]], "comparisons": _research_rows, "scope": "Conditional on fixed fitted seed3 models; exposed evaluation, no training-seed uncertainty; parent-vs-complete-combo comparisons are diagnostic, not isolated architecture effects."}, indent=2) + "\n")
    mo.vstack([mo.md("Paired results are saved with source hashes. Small positive architecture effects and nearly lossless quantization are hypotheses for follow-up, not confirmed improvements. The two complete-combo comparisons also change the final-head procedure."), mo.ui.table(_research_rows), mo.mpl.interactive(_research_fig)])
    return


@app.cell(hide_code=True)
def _(campaign_dir, json, mo, plt, refresh):
    refresh
    import hashlib as _clock_hashlib
    import numpy as _clock_np
    _clock_path = campaign_dir / "checks/e426_event_count_audit_f0_s3_v1.json"
    _clock_result = json.loads(_clock_path.read_text())
    _clock_counts = _clock_np.asarray(_clock_result["schedules"]["aligned"]["observed_counts"], dtype=float)
    _clock_shuffled = _clock_np.asarray(_clock_result["schedules"]["shuffled"]["applied_counts"], dtype=float)
    _clock_mean = float(_clock_counts.mean())
    _clock_fig, _clock_axes = plt.subplots(1, 2, figsize=(12, 4))
    _clock_axes[0].hist(_clock_counts, bins=40, color="#36719c", alpha=.85)
    _clock_axes[0].axvline(_clock_mean, color="#b54232", label=f"Mean {_clock_mean:.1f}")
    _clock_axes[0].set(xlabel="Endpoint events per optimizer update (batch 32 × 1000 rows)", ylabel="Updates", title="Actual fold-0 training inputs: event density")
    _clock_axes[0].legend()
    _clock_axes[1].plot(_clock_counts[:400] / _clock_mean, label="Aligned", lw=1)
    _clock_axes[1].plot(_clock_shuffled[:400] / _clock_mean, label="Shuffled", lw=1, alpha=.65)
    _clock_axes[1].axhline(1, color="black", ls="--", label="Constant")
    _clock_axes[1].set(xlabel="Optimizer-update index (first 400)", ylabel="Teacher EMA log-retention multiplier", title="Matched epoch-total retention; different alignment")
    _clock_axes[1].legend()
    _clock_fig.suptitle("E426 input-only audit: no model fitted, no predictive claim")
    _clock_fig.tight_layout()
    _clock_output = campaign_dir / "eda/event_teacher_clock"
    _clock_output.mkdir(parents=True, exist_ok=True)
    _clock_fig.savefig(_clock_output / "event_clock_schedule.png", dpi=160)
    _clock_fig.savefig(_clock_output / "event_clock_schedule.svg")
    _clock_summary = {"source_report_sha256": _clock_hashlib.sha256(_clock_path.read_bytes()).hexdigest(), "updates": len(_clock_counts), "mean_events": _clock_mean, "std_events": float(_clock_counts.std()), "event_count_cv": float(_clock_counts.std() / _clock_mean), "min_events": float(_clock_counts.min()), "max_events": float(_clock_counts.max()), "scope": "Input-only count schedule, no model score or fit", "decay_products": {k: v["decay_product"] for k, v in _clock_result["schedules"].items()}}
    (_clock_output / "source.json").write_text(json.dumps(_clock_summary, indent=2) + "\n")
    mo.vstack([mo.md(f"**E426 real-input clock audit:** {len(_clock_counts):,} updates, event-count CV {_clock_summary['event_count_cv']:.1%}. Fold-0 TRAIN 528 is rejected before input counting; 529 remains included. The constant, aligned and shuffled schedules have equal epoch-total retention. This is input evidence; three model fits remain pending."), mo.mpl.interactive(_clock_fig)])
    return


@app.cell(hide_code=True)
def _(campaign_dir, json, mo, plt, refresh):
    refresh
    import hashlib as _allarch_hashlib
    _allarch_path = campaign_dir / "checks/e424_report_all_fp32_v1.json"
    _allarch_report = json.loads(_allarch_path.read_text())
    _allarch_boot = _allarch_report["pooled"]["paired_component_bootstrap"]
    _allarch_contrasts = [
        ("Learned specialists vs GRU", "H093_LEARNED_minus_H093_GRU"),
        ("Hash specialists vs GRU", "H093_HASH_minus_H093_GRU"),
        ("Uniform specialists vs GRU", "H093_UNIVERSAL_minus_H093_GRU"),
        ("Conditional adapter vs GRU", "H094_CONDITIONAL_minus_H094_GRU"),
        ("Unconditional adapter vs GRU", "H094_UNCONDITIONAL_minus_H094_GRU"),
        ("Diagonal adapter vs GRU", "H094_DIAGONAL_minus_H094_GRU"),
        ("Learned routing vs uniform", "H093_LEARNED_minus_H093_UNIVERSAL"),
        ("Learned routing vs hash", "H093_LEARNED_minus_H093_HASH"),
        ("Conditional vs diagonal adapter", "H094_CONDITIONAL_minus_H094_DIAGONAL"),
        ("Conditional vs unconditional adapter", "H094_CONDITIONAL_minus_H094_UNCONDITIONAL"),
    ]
    _allarch_dir = campaign_dir / "eda/architecture_controls"
    _allarch_dir.mkdir(parents=True, exist_ok=True)
    _allarch_fig, _allarch_ax = plt.subplots(figsize=(10, 6))
    for _allarch_i, (_allarch_label, _allarch_key) in enumerate(_allarch_contrasts):
        _allarch_value = _allarch_boot[_allarch_key]
        _allarch_delta = _allarch_value["mean_diff"] * 1000
        _allarch_low, _allarch_high = [v * 1000 for v in _allarch_value["ci95"]]
        _allarch_ax.errorbar(_allarch_delta, _allarch_i, xerr=[[_allarch_delta-_allarch_low], [_allarch_high-_allarch_delta]], fmt="o", color="#245b85" if _allarch_i < 6 else "#ad6f28", capsize=3)
    _allarch_ax.axvline(0, color="#666666", linewidth=1)
    _allarch_ax.set_yticks(range(len(_allarch_contrasts)), [v[0] for v in _allarch_contrasts])
    _allarch_ax.invert_yaxis()
    _allarch_ax.set_xlabel("Global WP difference × 1,000; paired component-bootstrap 95% interval")
    _allarch_ax.set_title("Eight completed architecture fits: mechanism controls\n368 exposed excluded sequences; one fitted seed; common FP32 exports")
    _allarch_ax.grid(axis="x", alpha=.2)
    _allarch_fig.tight_layout()
    _allarch_fig.savefig(_allarch_dir / "paired_architecture_controls.png", dpi=180)
    _allarch_fig.savefig(_allarch_dir / "paired_architecture_controls.svg")
    (_allarch_dir / "source.json").write_text(json.dumps({"report": str(_allarch_path), "report_sha256": _allarch_hashlib.sha256(_allarch_path.read_bytes()).hexdigest(), "contrasts": _allarch_contrasts, "last_four_contrasts": "Exploratory reporting after the automatic all-arm report was inspected; not new selection gates", "scope": "Conditional on seed3 models; no training-seed or selection uncertainty; historically exposed panel"}, indent=2) + "\n")
    mo.vstack([mo.md("Eight architecture fits are complete. All displayed intervals include zero. The diagonal adapter has the largest observed adapter gain, but it needs independent confirmation. Orange contrasts directly compare mechanisms and are exploratory."), _allarch_fig])

    return


@app.cell(hide_code=True)
def _(campaign_dir, json, mo, plt, refresh):
    refresh
    import hashlib as _seed4_hashlib
    _seed4_path = campaign_dir / "checks/e450_report_seed4_bag_fp32_v1.json"
    _seed4_report = json.loads(_seed4_path.read_text())
    _seed4_boot = _seed4_report["pooled"]["paired_component_bootstrap"]
    _seed4_keys = [("Seed 4 vs seed 3", "COMBO_S4_minus_COMBO_S3"), ("Equal seed bag vs seed 3", "BAG_S3_S4_minus_COMBO_S3"), ("Equal seed bag vs seed 4", "BAG_S3_S4_minus_COMBO_S4"), ("Seed-4 head vs parent", "HEAD_S4_minus_PARENT_S4")]
    _seed4_fig, _seed4_ax = plt.subplots(figsize=(10, 4.3))
    for _seed4_i, (_seed4_label, _seed4_key) in enumerate(_seed4_keys):
        _seed4_result = _seed4_boot[_seed4_key]
        _seed4_delta = 1000 * _seed4_result["mean_diff"]
        _seed4_lo, _seed4_hi = [1000 * v for v in _seed4_result["ci95"]]
        _seed4_ax.errorbar(_seed4_delta, _seed4_i, xerr=[[_seed4_delta-_seed4_lo], [_seed4_hi-_seed4_delta]], fmt="o", capsize=4, color="#245b85")
    _seed4_ax.axvline(0, color="#666", linewidth=1)
    _seed4_ax.set_yticks(range(4), [v[0] for v in _seed4_keys])
    _seed4_ax.invert_yaxis()
    _seed4_ax.set_xlabel("Global WP difference × 1,000; paired component-bootstrap 95% interval")
    _seed4_ax.set_title("Fixed seed-4 replication and equal seed bag\n368 historically exposed excluded sequences; common FP32 exports")
    _seed4_ax.grid(axis="x", alpha=.2)
    _seed4_fig.tight_layout()
    _seed4_dir = campaign_dir / "eda/seed4_replication"
    _seed4_dir.mkdir(parents=True, exist_ok=True)
    _seed4_fig.savefig(_seed4_dir / "fixed_seed4_and_bag.png", dpi=180)
    _seed4_fig.savefig(_seed4_dir / "fixed_seed4_and_bag.svg")
    (_seed4_dir / "source.json").write_text(json.dumps({"report": str(_seed4_path), "sha256": _seed4_hashlib.sha256(_seed4_path.read_bytes()).hexdigest(), "scope": "All contrasts preregistered before seed4 scoring; no independent holdout or training-seed uncertainty claim", "contrasts": _seed4_keys}, indent=2) + "\n")
    mo.vstack([mo.md("Seed 4 scores 0.701695 versus seed 3 at 0.699822 on the excluded panel. The fixed equal seed bag scores 0.701420. All displayed 95% intervals include zero; public scores and deployment readiness remain unmeasured."), _seed4_fig])
    return


@app.cell(hide_code=True)
def _(campaign_dir, json, mo, plt, project_root, refresh):
    refresh
    import hashlib as _fullseed_hashlib
    _fullseed_path = campaign_dir / "checks/e450_full_valid_fixed_models_v1.json"
    _fullseed_report = json.loads(_fullseed_path.read_text())
    _fullseed_parts = ("all_supplied_validation", "validation_in_final_fit", "excluded_from_final_fit")
    _fullseed_labels = ("All supplied VALID\n1,873 sequences", "Used for final fitting\n1,505 sequences", "Excluded, exposed\n368 sequences")
    _fullseed_colors = {"COMBO_S4": "#275DAD", "BAG_S3_S4": "#C15A19"}
    _fullseed_fig, _fullseed_ax = plt.subplots(figsize=(9, 4.7))
    for _fullseed_index, _fullseed_name in enumerate(("COMBO_S4", "BAG_S3_S4")):
        _fullseed_values = [1000 * (_fullseed_report["scores"][_fullseed_name][_part]["wp"] - _fullseed_report["scores"]["COMBO_S3"][_part]["wp"]) for _part in _fullseed_parts]
        _fullseed_x = [_position + (_fullseed_index - .5) * .32 for _position in range(3)]
        _fullseed_bars = _fullseed_ax.bar(_fullseed_x, _fullseed_values, width=.30, label={"COMBO_S4":"Seed 4", "BAG_S3_S4":"Equal seed bag"}[_fullseed_name], color=_fullseed_colors[_fullseed_name])
        _fullseed_ax.bar_label(_fullseed_bars, fmt="%+.3f", padding=4, fontsize=9)
    _fullseed_ax.axhline(0, color="#444444", linewidth=.8)
    _fullseed_ax.set_xticks(range(3), _fullseed_labels)
    _fullseed_ax.set_ylabel("Global WP change versus seed 3 (×1,000)")
    _fullseed_ax.set_title("Fixed mixes on the supplied validation file")
    _fullseed_ax.legend(frameon=False)
    _fullseed_ax.spines[["top", "right"]].set_visible(False)
    _fullseed_ax.set_ylim(-.35, 2.15)
    _fullseed_fig.text(.5, .01, "Descriptive point estimates; 1,505 sequences fitted. Excluded-panel paired intervals include zero.", ha="center", fontsize=9)
    _fullseed_fig.tight_layout(rect=(0, .04, 1, 1))
    _fullseed_dir = campaign_dir / "eda/seed4_replication"
    _fullseed_dir.mkdir(parents=True, exist_ok=True)
    for _suffix in ("png", "svg"):
        _fullseed_fig.savefig(_fullseed_dir / ("full_valid_fixed_mixes." + _suffix), dpi=170)
    (_fullseed_dir / "full_valid_source.json").write_text(json.dumps({"report":str(_fullseed_path.relative_to(project_root)), "sha256":_fullseed_hashlib.sha256(_fullseed_path.read_bytes()).hexdigest(), "metric":"global_wp_clip2_v1", "scope":_fullseed_report["scope"]}, indent=2)+"\n")
    mo.vstack([mo.md("**E450 full supplied VALID:** seed 3 0.700503744; seed 4 0.700771292; fixed equal seed bag 0.701106873. Most rows were fitted. These are reporting-only FP32 mixes, not independent public results."), _fullseed_fig])
    return


@app.cell(hide_code=True)
def _(campaign_dir, json, mo, plt, project_root):
    import hashlib as _e427_hashlib
    _e427_stats_path = campaign_dir / "checks/e427_context_stats_s3_v1.json"
    _e427_stats = json.loads(_e427_stats_path.read_text())
    _e427_examples = []
    for _e427_split in ("train", "valid"):
        _e427_candidates = sorted((_s for _s in _e427_stats["endpoint_samples"] if _s["split"] == _e427_split and _s["first_four_endpoint_windows"]), key=lambda _s: _s["sample_key_sha256"])
        _e427_examples.extend(_e427_candidates[:2])
    _e427_fig, _e427_axes = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
    for _e427_ax, _e427_sample in zip(_e427_axes.flat, _e427_examples):
        _e427_window = _e427_sample["first_four_endpoint_windows"][0]
        _e427_positions = [r - _e427_window["endpoint_row"] for r in range(*_e427_window["window_rows"])]
        for _e427_column in (0, 11):
            _e427_ax.plot(_e427_positions, [r[_e427_column] for r in _e427_window["raw_volume_22"]], marker=".", label=f"anonymous x{22 + _e427_column}")
        _e427_ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        _e427_ax.set_title(f'{_e427_sample["split"]}: group {_e427_sample["row_group"]}')
        _e427_ax.set_xlabel("Rows relative to endpoint marker")
        _e427_ax.set_ylabel("Raw input value")
        _e427_ax.legend(fontsize=8)
    _e427_fig.suptitle("E427: fit-input endpoint examples (no predictive score)")
    _e427_dir = campaign_dir / "eda/e427_volume_gate"
    _e427_dir.mkdir(parents=True, exist_ok=True)
    for _e427_suffix in ("png", "svg"):
        _e427_fig.savefig(_e427_dir / f"fit_endpoint_examples.{_e427_suffix}", dpi=160)
    (_e427_dir / "fit_endpoint_examples.source.json").write_text(json.dumps({"source": str(_e427_stats_path.relative_to(project_root)), "sha256": _e427_hashlib.sha256(_e427_stats_path.read_bytes()).hexdigest(), "groups": [{"split": _s["split"], "row_group": _s["row_group"], "sample_key_sha256": _s["sample_key_sha256"]} for _s in _e427_examples], "selection": "First two SHA-ranked nonempty endpoint samples per fitting split; first sampled endpoint, raw x22/x33", "scope": "Input-only descriptive EDA; future-window rows illustrated, never passed as noncausal model inputs", "fit_rows": _e427_stats["row_count"], "heldout_rows_used": False}, indent=2) + "\n")
    mo.vstack([mo.md("E427's fallback statistics cover **242,220,000 fitting rows**. The 368 excluded VALID sequences and labels were not read. These endpoint illustrations show anonymous raw inputs, not a forecasting result."), _e427_fig])
    return


@app.cell(hide_code=True)
def _(campaign_dir, mo, refresh):
    refresh
    import json as _ema_json
    import hashlib as _ema_hash
    import matplotlib.pyplot as _ema_plt
    _ema_source = campaign_dir / "staging/e426_clock_comparison/results/e426_clock_screen_v2.json"
    _ema_result = _ema_json.loads(_ema_source.read_text())
    _ema_pairs = _ema_result["folds"]["0"]["paired_component_bootstrap"]
    _ema_names = list(_ema_pairs)
    _ema_means = [_ema_pairs[k]["mean_diff"] for k in _ema_names]
    _ema_low = [_ema_pairs[k]["ci95"][0] for k in _ema_names]
    _ema_high = [_ema_pairs[k]["ci95"][1] for k in _ema_names]
    _ema_fig, _ema_ax = _ema_plt.subplots(figsize=(8.5, 3.5), layout="constrained")
    for _ema_idx, _ema_name in enumerate(_ema_names):
        _ema_ax.plot([_ema_low[_ema_idx], _ema_high[_ema_idx]], [_ema_idx, _ema_idx], color="#476c9b", linewidth=3)
        _ema_ax.plot(_ema_means[_ema_idx], _ema_idx, "o", color="#243a58")
    _ema_ax.axvline(0, color="black", linewidth=1, linestyle="--")
    _ema_ax.set_yticks(range(3), [k.replace("_", " ") for k in _ema_names])
    _ema_ax.set_xlabel("Paired global weighted-Pearson difference (95% component interval)")
    _ema_ax.set_title("E426: matched event-clock EMA screen — fixed epoch 3, fold 0")
    _ema_ax.grid(axis="x", alpha=.2)
    _ema_folder = campaign_dir / "eda/e426_event_ema"
    _ema_folder.mkdir(parents=True, exist_ok=True)
    _ema_fig.savefig(_ema_folder / "paired_clock_screen.png", dpi=170)
    _ema_fig.savefig(_ema_folder / "paired_clock_screen.svg")
    _ema_plt.close(_ema_fig)
    (_ema_folder / "figure_provenance.json").write_text(_ema_json.dumps({"source": str(_ema_source.relative_to(campaign_dir)), "source_sha256": _ema_hash.sha256(_ema_source.read_bytes()).hexdigest(), "n_sequences": 468, "n_components": 141, "n_boot": 2000, "scope": "exposed fold0, one seed; no public effect claim"}, indent=2) + "\n")
    mo.vstack([mo.md("### E426 event-clock EMA: measured null screen\nThree matched full fits, canonical FP32 exports, fixed EP3. All paired 95% component intervals include zero; exposed fold0/seed3. This does not establish a general absence of event-time mechanisms."), mo.image(str(_ema_folder / "paired_clock_screen.png"))])
    return


@app.cell(hide_code=True)
def _(campaign_dir, json, mo):
    climb_history = json.loads((campaign_dir / "eda/leaderboard_climb/leaderboard_history.json").read_text())
    climb_artifact = campaign_dir / "eda/leaderboard_climb/leaderboard_climb.html"
    mo.vstack([
        mo.md("## Public climb and validation on one WP scale\n\nBest public **0.6660**. Download the offline interactive timeline; old forecasts and raw validation are separate series. No new experiments."),
        mo.download(climb_artifact.read_bytes(), filename="Connectome_leaderboard_climb.html", mimetype="text/html", label="Download interactive HTML"),
        mo.ui.table([{ "when_Kyiv": row["time"], "kind": row["kind"], "package": row["name"], "public": row.get("public"), "prior_expected_public": row.get("expected_public"), "full_validation": row.get("validation_full"), "excluded_368": row.get("validation_368"), "change": row.get("what") } for row in climb_history["rows"]], selection=None),
    ])
    return (climb_history,)


@app.cell(hide_code=True)
def _(campaign_dir, climb_history, plt):
    climb_fig, climb_ax = plt.subplots(figsize=(12, 6))
    _climb_series = [
        ("public", "Public (ours)", "#286bd1", "-"),
        ("validation_368", "Excluded 368 validation", "#ad4da1", "--"),
        ("validation_full", "Full supplied validation (mostly fitted)", "#e19b22", "--"),
        ("expected_public", "Historical expected public", "#8a6e24", ":"),
    ]
    for _key, _label, _color, _style in _climb_series:
        _observed = [row for row in climb_history["rows"] if row.get(_key) is not None and (_key != "public" or row["kind"] == "own")]
        climb_ax.plot([row["index"] for row in _observed], [row[_key] for row in _observed], marker="o", markersize=4, linewidth=1.4, linestyle=_style, color=_color, label=_label)
    climb_ax.axhline(0.6660, color="#008474", linestyle=":", linewidth=1, label="Public target 0.6660 (reached)")
    _climb_ticks = climb_history["rows"][::4]
    climb_ax.set_xticks([row["index"] for row in _climb_ticks], [row["time"][5:10] for row in _climb_ticks])
    climb_ax.set_xlabel("Chronological events (Kyiv clock labels; time gaps not proportional)")
    climb_ax.set_ylabel("Global clipped weighted Pearson — one numeric scale")
    climb_ax.set_title("Connectome public climb: validation levels do not equal public expectations")
    climb_ax.grid(alpha=0.2)
    climb_ax.legend(loc="upper left", fontsize=9)
    climb_fig.tight_layout()
    climb_fig.savefig(campaign_dir / "eda/leaderboard_climb/score_series_shared_scale.png", dpi=180)
    climb_fig
    return


@app.cell(hide_code=True)
def _(json, mo, project_root, refresh):
    refresh
    _oct5_component_dir = project_root / "research/campaign_2026-10-05/sequence_components"
    _oct5_e464_file = project_root / "research/campaign_2026-10-04/staging/e464/results/e464_report_all_fp32_v1.json"
    if _oct5_e464_file.exists():
        _oct5_e464_scores = json.loads(_oct5_e464_file.read_text())["pooled"]["exact_global_wp"]
        mo.vstack([mo.md("## Completed architecture comparisons and new memory components\nE464: 8/8 fits, 8/8 exports and 16 paired comparisons completed. Scores below use the exposed 368-sequence holdout; they are not public scores."), mo.ui.table([{"arm": name, **score} for name, score in _oct5_e464_scores.items()]), mo.md("New components: stable diagonal memory, fixed-state associative attention, and additive spline edges. CPU checks passed; actual-shape GPU probe precedes fitting. Review and source: `research/campaign_2026-10-05/sequence_components/`. No fresh validation or expected-public-0.70 claim.")])
    else:
        mo.md("Awaiting the fixed architecture score artifact.")

    return


@app.cell(hide_code=True)
def _(json, mo, project_root, refresh):
    refresh
    _oct5_sequence_result_path = project_root / "research/campaign_2026-10-05/sequence_components/runs/ten_arm_v1/result.json"
    _oct5_sequence_images = project_root / "research/campaign_2026-10-05/sequence_components/eda"
    if _oct5_sequence_result_path.exists():
        _oct5_sequence_result = json.loads(_oct5_sequence_result_path.read_text())
        _oct5_sequence_view = [mo.md(f"## Linear memory, attention and spline experiments\nCompleted model outputs: {len(_oct5_sequence_result['arms'])}/10. One fixed epoch, 24 fit and 16 component-separated test sequences. Historically exposed; mechanism screen, no public forecast."), mo.ui.table([{k: a[k] for k in ("arm", "parameters", "wp", "fit_seconds", "steps")} for a in _oct5_sequence_result["arms"]])]
        if (_oct5_sequence_images / "ten_arm_scores.png").exists():
            _oct5_sequence_view += [mo.image(str(_oct5_sequence_images / "ten_arm_scores.png")), mo.image(str(_oct5_sequence_images / "learned_spline_edges.png"))]
        mo.vstack(_oct5_sequence_view)
    else:
        mo.md("The ten-model mechanism screen is preparing its first fixed-checkpoint output.")

    return


@app.cell(hide_code=True)
def _(json, mo, project_root, refresh):
    refresh
    _joint_result = project_root / "research/campaign_2026-10-05/joint_target_model/RESULT.json"
    _monitor_file = project_root / "research/campaign_2026-10-05/checks/monitor_runtime/STATUS.json"
    _joint = json.loads(_joint_result.read_text())
    _monitor = json.loads(_monitor_file.read_text()) if _monitor_file.exists() else {}
    mo.vstack([
        mo.md("### E468: five frozen heads and task monitoring\nAll five fixed GPU fits and ten between-arm comparisons completed. Historical fold0 only; none improves on its unchanged parent. Current public best stays **0.6660**."),
        mo.ui.table([{"arm": _name, "WP": _v["global_wp"]["wp"]} for _name, _v in _joint["evaluation"]["scores"].items()]),
        mo.image(str(_joint_result.parent / "eda/five_heads.png")),
        mo.md("**Latest checker:** " + _monitor.get("utc", "waiting") + "\n\nLocal caps: 12 CPU cores, 30 GB RAM (27 GB soft), 5 GB VRAM. Five-minute status / fifteen-minute Kaggle discovery and quota."),
        mo.ui.table([{"kernel": _k.split("/")[-1], "status": _v.get("status"), "observed_unix": _v.get("observed_unix")} for _k,_v in _monitor.get("provider", {}).get("kernels", {}).items()]),
    ])
    return


@app.cell(hide_code=True)
def _(json, mo, project_root, refresh):
    refresh
    _activation_folder = project_root / "research/campaign_2026-10-05/activation_optimizer"
    _activation_raw = json.loads((_activation_folder / "FIT_RESULT.json").read_text())
    _activation_rows = [{"Activation": _a["activation"], "Optimizer": _a["optimizer"], "WP": round(_a["wp"], 6), "Updates": _a["steps"]} for _a in _activation_raw["fit"]["arms"]]
    _activation_plot = _activation_folder / "eda/sixteen_v2/activation_optimizer_wp_heatmap.png"
    mo.vstack([mo.md("## Activation × optimizer: 16/16 complete\nSmall GRU, 24 fit / 16 historically exposed components. One seed; no public forecast. AdamW sine vs plain GELU: +0.0017, interval crosses zero. Muon sine vs plain GELU: +0.0088, positive conditional interval, still below tanh+Muon. All 30 exploratory contrasts retained."), mo.ui.table(_activation_rows), mo.image(src=str(_activation_plot)), mo.image(src=str(_activation_folder / "eda/sixteen_v2/optimizer_contrasts_by_activation.png"))])
    return


@app.cell(hide_code=True)
def _(mo, project_root, refresh):
    refresh
    _seed_folder = project_root / "research/campaign_2026-10-05/sequence_components"
    mo.vstack([mo.md("## Sequence components: 30/30 fits across three seeds\nLearned-memory spline lead: +0.0386, +0.0288, +0.0125; the last conditional interval includes zero. Learned decay gains remain around +0.003. This exposed panel does not establish public transfer."), mo.image(src=str(_seed_folder / "eda/three_seed_scores.png"))])
    return


@app.cell(hide_code=True)
def _(mo, project_root, refresh):
    refresh
    _activation_eda = project_root / "research/campaign_2026-10-05/activation_optimizer/eda"
    mo.vstack([mo.md("## Activation shapes and fit-only occupancy\nFour fixed fit sequences streamed through all16 checkpoints; no fitting or test-row access. Most preactivations lie near zero. A linear-residual control is needed before attributing gains to periodic curvature."), mo.image(src=str(_activation_eda / "activation_curves.png")), mo.image(src=str(_activation_eda / "fit_pre_activation_summary.png"))])
    return


@app.cell(hide_code=True)
def _(json, mo, project_root, refresh):
    refresh.value
    _activation_reps = json.loads((project_root / "research/campaign_2026-10-05/activation_optimizer/THREE_SEED_ACTIVATION_RESULT.json").read_text())
    mo.vstack([
        mo.md("## Activation × optimizer: all 48 fixed fits. Three seeds, same exposed diagnostic components. Small-model scores, not public forecasts. Sine-minus-GELU stays positive; optimizer ranking varies by seed."),
        mo.ui.table([{"recipe": _name, **{f"seed {_seed}": _activation_reps["seeds"][_seed]["scores"][_name]["wp"] for _seed in ("271828", "271829", "271830")}} for _name in _activation_reps["descriptive_cross_seed"]]),
        mo.image(str(project_root / "research/campaign_2026-10-05/activation_optimizer/eda/seed_replication/all_activation_seeds.png")),
    ])
    return


@app.cell(hide_code=True)
def _(json, mo, project_root, refresh):
    refresh.value
    _mech = json.loads((project_root / "research/campaign_2026-10-05/activation_mechanism/RESULT.json").read_text())
    mo.vstack([
        mo.md("## Curvature versus local slope: ten fixed fits. GELU + 0.1x beats GELU + 0.1sin(x) in this one-seed pilot. Periodic curvature is not needed to explain the gain. No public-score claim."),
        mo.ui.table([{"recipe": _name, **_score} for _name, _score in _mech["arm_scores"].items()]),
        mo.image(str(project_root / "research/campaign_2026-10-05/activation_mechanism/eda/all_mechanism_contrasts.png")),
    ])
    return


@app.cell(hide_code=True)
def _(json, mo, project_root, refresh):
    _ = refresh.value
    _r = json.loads((project_root / "research/campaign_2026-10-05/recurrent_activation/RESULT.json").read_text())
    _rows = [{"recipe": _name, "WP": _score["wp"], "t0": _score["t0"], "t1": _score["t1"]} for _name, _score in _r["arm_scores"].items() if not _name.startswith("cudnn")]
    mo.vstack([mo.md("### E471: bounded recurrent candidate × optimizer\nSix fits, one exposed seed. Negative perturbation helps AdamW; positive helps Muon. All signs and optimizers are being replicated. No public-score claim."), mo.ui.table(_rows), mo.image(str(project_root / "research/campaign_2026-10-05/recurrent_activation/eda/all_recurrent_contrasts.png"))])
    return


@app.cell(hide_code=True)
def _(json, mo, project_root, refresh):
    _ = refresh.value
    _r = json.loads((project_root / "research/campaign_2026-10-05/recurrent_activation/THREE_SEED_ACTIVATION_RESULT.json").read_text())
    _rows = [{"seed": _seed, "recipe": _arm, "WP": _score["wp"]} for _seed, _fit in _r["attempts_and_scores"].items() for _arm, _score in _fit["scores"].items()]
    mo.vstack([mo.md("### E475: all recurrent recipes replicated\nPositive bounded candidate alpha improves Muon at all three fit seeds (+0.0061/+0.0153/+0.0116). Negative alpha hurts Muon; AdamW effects change sign. Same exposed panel, conditional component intervals, no public-score forecast."), mo.ui.table(_rows), mo.image(str(project_root / "research/campaign_2026-10-05/recurrent_activation/eda/all_recurrent_seeds.png"))])
    return


@app.cell(hide_code=True)
def _(json, mo, project_root):
    _placement_path = project_root / 'research/campaign_2026-10-05/activation_placement'
    _placement_result = json.loads((_placement_path / 'scores/twelve_cpu_v1/score_result.json').read_text())
    mo.vstack([mo.md('## Activation placement: all 12 fixed fits\nTwo-layer H64;16 exposed components. Sine embedding is seed-dependent; head changes are near-neutral. This is not a public-score forecast.'), mo.ui.table(list(_placement_result['arm_scores'].values())), mo.image(str(_placement_path / 'eda/all_placement_arms.png'))])
    return


@app.cell(hide_code=True)
def _(json, mo, project_root):
    _scale_path = project_root / 'research/campaign_2026-10-05/recurrent_activation_scale'
    _scale_result = json.loads((_scale_path / 'eda/six_v1/RESULT.json').read_text())
    mo.vstack([mo.md('## Recurrent candidate scaling: three seeds\nPositive point gains at all three seeds; two conditional intervals exclude zero. Six fixed H64 fits, no public forecast.'), mo.ui.table(list(_scale_result['arms'].values())), mo.image(str(_scale_path / 'eda/all_scaling_arms.png'))])
    return


@app.cell(hide_code=True)
def _(json, mo, project_root):
    _schedule_path = project_root / 'research/campaign_2026-10-05/activation_schedule'
    _schedule_result = json.loads((_schedule_path / 'eda/twelve_v1/scores.json').read_text())
    mo.vstack([mo.md('## Activation × optimizer × LR protocol: all 12 fixed fits\nWarmup/cosine changes the activation ranking. One seed and an exposed diagnostic panel; seed replication is needed.'), mo.ui.table([dict(arm=_arm, **_score) for _arm, _score in _schedule_result['arm_scores'].items()]), mo.image(str(_schedule_path / 'eda/all_schedule_arms.png'))])
    return


@app.cell(hide_code=True)
def _(mo, project_root):
    _curvature_plot = project_root / "research/campaign_2026-10-05/recurrent_activation_curvature/eda/candidate_maps_and_slopes.png"
    mo.vstack([mo.md("### Candidate shapes: what the sine control actually changes\nE481 is registered, not fitted yet. The cubic matches the sine map at the center and endpoints but changes the slope distribution. These scalar bounds do not prove stability of the whole recurrent model."), mo.image(str(_curvature_plot))])
    return


@app.cell(hide_code=True)
def _(json, mo, project_root):
    _head_dir = project_root / "research/campaign_2026-10-05/activation_head_mixture"
    _head_result = json.loads((_head_dir / "eda/twelve_v6b/RESULT.json").read_text())
    mo.vstack([mo.md("### E477: mixed activation heads × dropout\nAll12 fits and independent grades landed. Effects reverse with seed: split versus GELU without dropout is −0.017395 / +0.006980. All18 primary/secondary contrasts remain visible. Exposed diagnostic panel; no public forecast."), mo.image(str(_head_dir / "eda/all_head_mixture_arms.png")), mo.ui.table([{"arm": _key, "global_wp": _value["global_wp"], **_value["target_wp"]} for _key, _value in _head_result["arms"].items()]), mo.ui.table([{"contrast": _key, "delta": _value["mean_diff"], "ci95": str(_value["ci95"]), "family": _value["family"]} for _key, _value in _head_result["contrasts"].items()])])
    return


@app.cell(hide_code=True)
def _(mo, project_root):
    _path = project_root / "research/campaign_2026-10-05/recurrent_activation_adaptive/eda/coefficient_trajectories.png"
    mo.vstack([
        mo.md("### Learned recurrent shape: training telemetry\nAll eight learned cells and both layers are shown. Coefficients stayed close to their initial values over 474 updates. This does not establish that learned activations cannot help; scoring is still pending. Scalar LR is fixed at 0.001, with zero coefficient weight decay."),
        mo.image(str(_path)),
    ])
    return


@app.cell(hide_code=True)
def e480_three_seed_schedule_replication(json, mo, project_root):
    _e480_dir = project_root / "research/campaign_2026-10-05/activation_schedule_replication"
    _e480_scores = json.loads((_e480_dir / "eda/twentyfour_v1/scores.json").read_text())
    _e476_scores = json.loads((project_root / "research/campaign_2026-10-05/activation_schedule/eda/twelve_v1/scores.json").read_text())
    _e480_primary_rows = []
    for _name, _effect in _e476_scores["pair_contrasts"].items():
        if _effect.get("family") == "warmcos_minus_constant_mean":
            _e480_primary_rows.append({"fit_seed": 271828, "run": "E476 original", "contrast": _name, "delta": _effect["mean_diff"], "ci95": _effect["ci95"]})
    for _seed, _effects in _e480_scores["primary_warmcos_minus_constant_mean_by_seed"].items():
        for _name, _effect in _effects.items():
            _e480_primary_rows.append({"fit_seed": int(_seed), "run": "E480 replication", "contrast": _name, "delta": _effect["mean_diff"], "ci95": _effect["ci95"]})
    _e480_contrast_rows = []
    for _family_key in ("pair_contrasts_by_seed", "joint_interactions_by_seed"):
        for _seed, _effects in _e480_scores[_family_key].items():
            for _name, _effect in _effects.items():
                _e480_contrast_rows.append({"fit_seed": int(_seed), "family": _effect["family"], "contrast": _name, "delta": _effect["mean_diff"], "ci95": _effect["ci95"], "p_le_0": _effect["p_le_0"]})
    _e480_arm_rows = [{"arm": _name, **_score} for _name, _score in _e480_scores["arm_scores"].items()]
    mo.vstack([
        mo.md("""## Schedule replication: seed effects reverse
    E476's original seed 271828 had four positive warm-cosine-minus-constant-mean effects. The two registered E480 replications (271829, 271830) have eight negative point estimates, six with component-bootstrap CIs below zero. The same 16 previously exposed components are reused; this is paired descriptive evidence, not an untouched holdout or a seed-population estimate. The E480 CPU grade attempt finished exit 0 with verified teardown. Full report: `research/campaign-2026-10-05/activation_schedule_replication/THREE_SEED_REPORT.md`."""),
        mo.image(str(_e480_dir / "eda/three_seed_v1/three_seed_scores.png")),
        mo.image(str(_e480_dir / "eda/three_seed_v1/three_seed_primary_contrasts.png")),
        mo.md("### All 12 paired primary effects"),
        mo.ui.table(_e480_primary_rows),
        mo.md("### All 36 E476/E480 official arm scores"),
        mo.ui.table(_e480_arm_rows),
        mo.md("### All 44 registered E480 contrasts (20 pairs + 2 joint interactions per seed)"),
        mo.ui.table(_e480_contrast_rows),
    ])
    return


@app.cell(hide_code=True)
def e479_learned_recurrent_activation(json, mo, project_root):
    _e479_dir = project_root / "research/campaign_2026-10-05/recurrent_activation_adaptive"
    _e479_result = json.loads((_e479_dir / "eda/sixteen_v2/RESULT.json").read_text())
    _e479_arm_rows = []
    for _e479_key, _e479_arm in _e479_result["arms"].items():
        _e479_arm_rows.append({"fit_seed": _e479_arm["seed"], "optimizer": _e479_arm["optimizer"], "condition": _e479_arm["condition"], "global_wp": _e479_arm["global_wp"], "t0_wp": _e479_arm["target_wp"]["t0"], "t1_wp": _e479_arm["target_wp"]["t1"]})
    _e479_contrast_rows = []
    for _e479_key, _e479_effect in _e479_result["contrasts"].items():
        _e479_contrast_rows.append({"contrast": _e479_key, "difference": _e479_effect["difference"], "ci95": _e479_effect["ci95"], "fit_seed": _e479_effect["seed"]})
    mo.vstack([
        mo.md("""## E479: learned recurrent candidate shape — 16 fits, 28 contrasts
    This fixed two-seed study compares per-layer learned amplitudes in `u + α sin(πu)` against fixed controls using AdamW and Muon+AdamW. The learned-versus-fixed effects are generally small and mixed: AdamW seed 271828 has a +0.000650 effect for learned-from-0.1 versus fixed 0.1, but the sign does not repeat across all seed/optimizer pairs. Optimizer contrasts reverse direction across the two fit seeds. Results use 16 previously exposed components and are descriptive, not an untouched holdout or a public-score forecast. Full report: `research/campaign_2026-10-05/recurrent_activation_adaptive/REPORT.md`.

    The independent raw-export recomputation matched all 16 arm scores and 28 contrasts to maximum absolute error 2.44e-15."""),
        mo.image(str(_e479_dir / "eda/sixteen_v2/e479_scores.png")),
        mo.image(str(_e479_dir / "eda/sixteen_v2/e479_contrasts.png")),
        mo.image(str(_e479_dir / "eda/coefficient_trajectories.png")),
        mo.md("### All 16 registered arm scores"),
        mo.ui.table(_e479_arm_rows),
        mo.md("### All 28 registered paired contrasts"),
        mo.ui.table(_e479_contrast_rows),
    ])
    return


@app.cell(hide_code=True)
def e481_recurrent_candidate_curvature(json, mo, project_root):
    _e481_dir = project_root / "research/campaign_2026-10-05/recurrent_activation_curvature"
    _e481_scores = json.loads((_e481_dir / "eda/eighteen_v2/scores.json").read_text())
    _e481_arm_rows = []
    for _e481_key, _e481_arm in _e481_scores["arm_scores"].items():
        _e481_arm_rows.append({"fit_seed": _e481_arm["seed"], "optimizer": _e481_arm["optimizer"], "candidate": _e481_arm["shape"], "global_wp": _e481_arm["wp"], "t0_wp": _e481_arm["t0"], "t1_wp": _e481_arm["t1"]})
    _e481_contrast_rows = []
    for _e481_seed, _e481_effects in _e481_scores["contrasts_by_seed"].items():
        for _e481_name, _e481_effect in _e481_effects.items():
            _e481_contrast_rows.append({"fit_seed": int(_e481_seed), "family": _e481_effect["family"], "contrast": _e481_name, "difference": _e481_effect["mean_diff"], "ci95": _e481_effect["ci95"], "p_le_0": _e481_effect["p_le_0"]})
    mo.vstack([
        mo.md("""## E481: candidate shape depends on seed and optimizer
    The reset-after GRU candidate compared baseline `u`, bounded sine `u + 0.1 sin(πu)`, and matched-center cubic `u + 0.1πu(1−u²)`. Across three fixed fit seeds × two optimizers × three shapes, all 18 fits completed and were independently graded. In seed 271828, both nonlinearity variants sharply reduce AdamW score but slightly improve Muon+AdamW; the following two seeds show positive shape-vs-baseline estimates under both optimizers, with one Muon interval crossing zero. Cubic-vs-sine gaps are smaller and mostly uncertain. This uses 16 previously exposed components and does not predict public performance. The CPU grade attempt exited 0 with verified teardown. Full results and limitations: `research/campaign_2026-10-05/recurrent_activation_curvature/REPORT.md`."""),
        mo.image(str(_e481_dir / "eda/eighteen_v2/e481_scores.png")),
        mo.image(str(_e481_dir / "eda/eighteen_v2/e481_contrasts.png")),
        mo.md("### All 18 registered arm scores"),
        mo.ui.table(_e481_arm_rows),
        mo.md("### All 30 registered paired global contrasts"),
        mo.ui.table(_e481_contrast_rows),
    ])
    return


@app.cell(hide_code=True)
def e482_t4_replication_independent_grade(mo, project_root):

    import json as _E482Json
    _e482_dir = project_root / "research/campaign_2026-10-05/activation_t4_replication_v2"
    _e482_eda = _e482_dir / "eda/six_v2"
    _e482_scores = _E482Json.loads((_e482_eda / "scores.json").read_text())
    _e482_local_all = _E482Json.loads((project_root / "research/campaign_2026-10-05/recurrent_activation_curvature/eda/eighteen_v2/scores.json").read_text())
    _e482_local = {"arm_scores": {k: v for k, v in _e482_local_all["arm_scores"].items() if v["seed"] == 271828}, "contrasts": _e482_local_all["contrasts_by_seed"]["271828"]}
    _e482_proof = _E482Json.loads((project_root / "research/campaign_2026-10-05/checks/t4_activation_export_reproduction.json").read_text())
    _e482_arms = []
    for _key, _arm in _e482_scores["arm_scores"].items():
        _local_arm = _e482_local["arm_scores"][f"seed271828__{_key}"]
        _e482_arms.append({"optimizer": _arm["optimizer"], "shape": _arm["shape"], "T4 global WP": _arm["wp"], "T4 target 0": _arm["t0"], "T4 target 1": _arm["t1"], "T4 − local WP": _arm["wp"] - _local_arm["wp"]})
    _e482_contrasts = []
    for _key, _item in _e482_scores["contrasts"].items():
        _local_item = _e482_local["contrasts"][_key]
        _e482_contrasts.append({"contrast": _key.replace("__", " ").replace("_", " "), "T4 Δ WP": _item["mean_diff"], "T4 CI low": _item["ci95"][0], "T4 CI high": _item["ci95"][1], "local Δ WP": _local_item["mean_diff"], "local CI low": _local_item["ci95"][0], "local CI high": _local_item["ci95"][1]})
    _e482_summary = "## E482 private T4 replication — independent grade" + chr(10) + chr(10) + f"Six fits, one seed, and the same 16 exposed components. Root's raw-sum check matched all six arm scores and ten contrasts with max absolute error {_e482_proof['maximum_absolute_error']:.3e}. This is an arithmetic reproduction, not a second grader or new market confirmation. The public best remains 0.6660."
    mo.vstack([
        mo.md(_e482_summary),
        mo.ui.table(_e482_arms, selection=None),
        mo.image(src=str(_e482_eda / "arm_scores_t4_vs_local.png")),
        mo.ui.table(_e482_contrasts, selection=None),
        mo.image(src=str(_e482_eda / "contrasts_t4_vs_local.png")),
        mo.md("The T4 cubic-minus-sine Muon contrast interval is slightly positive while the same-seed local interval crosses zero. Treat this as hardware/optimizer numerical sensitivity, not new-seed evidence. All six arms and ten contrasts are shown; none is selected or dropped.")
    ])

    return


@app.cell(hide_code=True)
def e483_momentum_independent_v3_grade(mo, project_root):
    _e483_root = project_root / "research/campaign_2026-10-05/activation_momentum"
    _e483_report = (_e483_root / "REPORT.md").read_text()
    _e483_report = "\n".join(line for line in _e483_report.splitlines() if not line.startswith("!["))
    mo.vstack([
        mo.md(_e483_report),
        mo.md("### Figures"),
        mo.image(str(_e483_root / "eda/eighteen_v3/all18_arm_scores.png"), alt="All 18 seed, shape, and momentum arm scores"),
        mo.image(str(_e483_root / "eda/eighteen_v3/all27_contrasts.png"), alt="All 27 registered seed-specific paired contrasts"),
        mo.image(str(_e483_root / "eda/eighteen_v3/source_telemetry_means.png"), alt="Four-snapshot mean gradient and update RMS by source family"),
    ])
    return


@app.cell(hide_code=True)
def e484_context_mixture_independent_v2_grade(Path, mo):
    e484_root = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
    e484_base = e484_root / "research/campaign_2026-10-05/activation_context_mixture"
    e484_report = (e484_base / "REPORT.md").read_text()
    e484_report = "\n".join(line for line in e484_report.splitlines() if not line.startswith("!["))
    mo.vstack([
        mo.md(e484_report),
        mo.image(str(e484_base / "eda/twelve_v1/all12_arm_scores.png"), alt="All twelve E484 arm scores"),
        mo.image(str(e484_base / "eda/twelve_v1/all16_global_contrasts.png"), alt="All sixteen global contrasts"),
        mo.image(str(e484_base / "eda/twelve_v1/all32_target_contrasts.png"), alt="All thirty-two target-specific contrasts"),
    ])

    return


@app.cell(hide_code=True)
def e486_crossed_activation_independent_grade(Path, mo):
    e486_root = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
    e486_base = e486_root / "research/campaign_2026-10-05/activation_counterfactual"
    e486_report = (e486_base / "REPORT.md").read_text()
    e486_report = "\n".join(line for line in e486_report.splitlines() if not line.startswith("!["))
    mo.vstack([
        mo.md(e486_report),
        mo.image(str(e486_base / "eda/cross_v1/all36_global_cells.png"), alt="All 36 global crossed-activation cells"),
        mo.image(str(e486_base / "eda/cross_v1/all36_target0_cells.png"), alt="All 36 target 0 cells"),
        mo.image(str(e486_base / "eda/cross_v1/all36_target1_cells.png"), alt="All 36 target 1 cells"),
        mo.image(str(e486_base / "eda/cross_v1/all54_global_contrasts.png"), alt="All 54 paired global contrasts"),
    ])

    return


@app.cell(hide_code=True)
def _(json, mo, project_root):
    _e487_root = project_root / "research/campaign_2026-10-05"
    _e487_proof = json.loads((_e487_root / "checks/e487_saved_delta_reproduction.json").read_text())
    _e487_ok = _e487_proof["status"] == "all_checked_values_match"
    mo.vstack([
        mo.md("### E487: Muon precision diagnostic\n\nAll nine saved update-delta cells independently verified. BF16 native/custom update checks fail; FP32 checks pass on this fixed fit-only batch. This is numerical attribution, not a predictive or public-score gain. Original E485 remains unadmitted." if _e487_ok else "E487 independent verification is pending."),
        mo.image(str(_e487_root / "muon_orthogonalization_precision/eda/verified_v4/all_matrix_and_family_update_gates.png")) if _e487_ok else mo.md(""),
    ])
    return


if __name__ == "__main__":
    app.run()
