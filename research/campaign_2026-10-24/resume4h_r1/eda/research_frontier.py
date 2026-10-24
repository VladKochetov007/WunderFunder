import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import json
    from pathlib import Path
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    pa.set_cpu_count(4)
    pa.set_io_thread_count(2)
    repo = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
    eda_dir = repo / "research/campaign_2026-10-24/resume4h_r1/eda"
    return eda_dir, json, mo, np, pa, pq, repo


@app.cell
def research_overview(mo):
    mo.md("""
    # Research EDA: update clocks and learned corrections

    This notebook explores **96 fixed validation sequences (1.92 million rows)** and saved F0 predictions from the October 24 research cycle. Existing validation is **exposed**; the clock descriptions and separately admitted score diagnostics have different evidential scopes.

    - **Sequence-start censoring is rare:** 0.07438% of scored field–row pairs, with short update ages covering far more rows.
    - **Grouping changes the age representation:** 112 fields form 65 update-pattern classes in this sample. Seven-group clocks differ from individual ages on 14.39% of scored field–row pairs; nine families reduce that to 11.53%.
    - **Correction scale changes the F0 result:** all five saved corrections lose against the parent at their original scale, but gain about +0.00013 at one fixed additional scale of 0.1. Mean-only A gains almost as much as the relationship variants; this remains exploratory single-cell evidence.

    Use the field, weight and target selectors to inspect the patterns below. Every plot has a saved PNG and SVG beside this notebook. The official metric and frozen fitting recipes are unchanged. A label-derived slice is an explanatory view, never a serving feature.
    """)
    return


@app.cell
def sample_plan(eda_dir, json, mo, np, repo):
    data_dir = repo / "datasets"
    feature_names = [f"i{i}_{g}{j}" for i in range(2) for g, n in [("p", 22), ("v", 22), ("dp", 4), ("dv", 4)] for j in range(n)] + [f"a{i}" for i in range(8)]
    _eda_rng = np.random.default_rng(20261024)
    eda_sample = {name: np.sort(_eda_rng.choice(n, 96, replace=False)).tolist() for name, n in [("train", 10607), ("valid", 1873)]}
    eda_plan = {"id": "EDA-CENSORING-20261024-R1", "seed": 20261024, "sample": eda_sample, "selection": "96 uniform row groups per source without target/class stratification", "feature_columns": feature_names, "position_edges": [0, 99, 500, 2000, 10000, 20000], "age_thresholds": [1, 16, 64, 512, 4096, 10000], "questions": ["field-specific censored age coverage", "exact versus capped ages beyond position", "age and update activity by source and position"], "status": "frozen_before_new_values", "label_use": "descriptive clipped-label magnitude mass only; no predictor or coefficient fitting", "exposure": "previously inspected train/valid; not fresh holdout", "comparison_scope": "descriptive source samples; no official model score or causal conclusion", "input_identity": {name: {"path": str(data_dir/f"{name}.parquet"), "bytes": (data_dir/f"{name}.parquet").stat().st_size, "mtime_ns": (data_dir/f"{name}.parquet").stat().st_mtime_ns} for name in eda_sample}}
    _plan_path = eda_dir / "EDA_PLAN_R1.json"
    if _plan_path.exists():
        assert json.loads(_plan_path.read_text()) == eda_plan
    else:
        _plan_path.write_text(json.dumps(eda_plan, indent=2)+"\n")
    mo.md(f"Frozen sample: **{sum(map(len, eda_sample.values()))} whole sequences**, two source files, seed 20261024. One row group in memory at a time.")
    return data_dir, eda_sample, feature_names


@app.cell
def clock_statistics(np, pa):
    def describe_censoring(table: pa.Table, names: list[str]) -> dict[str, object]:
        """Summarize causal field clocks within one complete observed sequence."""
        x = np.column_stack([table[c].to_numpy() for c in names]).astype(np.float32)
        y = np.column_stack([table[c].to_numpy() for c in ["t0", "t1"]]).astype(np.float64)
        need = table["need_prediction"].to_numpy().astype(bool)
        scored = table["is_scored"].to_numpy().astype(bool) if "is_scored" in table.column_names else need
        selected = need & scored
        seq = table["seq_ix"].to_numpy()
        assert len(x) == 20000 and len(np.unique(seq)) == 1
        assert np.isfinite(x).all() and np.isfinite(y).all()
        idx = np.arange(len(x), dtype=np.int32)
        weight = np.abs(np.clip(y, -2, 2)) * selected[:, None]
        changed = np.zeros(x.shape, dtype=bool)
        changed[1:] = x[1:] != x[:-1]
        thresholds = np.asarray([1, 16, 64, 512, 4096, 10000])
        position_edges = [0, 99, 500, 2000, 10000, 20000]
        result = {
            "seq_ix": int(seq[0]), "rows": len(x), "selected_rows": int(selected.sum()),
            "selected_mass": weight.sum(0), "changes": changed.sum(0),
            "censored_rows": np.zeros(len(names), dtype=np.int64),
            "censored_mass": np.zeros((len(names), 2)),
            "long_age_rows": np.zeros((len(names), len(thresholds)), dtype=np.int64),
            "long_age_mass": np.zeros((len(names), len(thresholds), 2)),
            "censored_long_age_mass": np.zeros((len(names), len(thresholds), 2)),
            "position_selected_rows": np.zeros(5, dtype=np.int64),
            "position_censored_rows": np.zeros((5, len(names)), dtype=np.int64),
            "position_selected_mass": np.zeros((5, 2)),
            "position_censored_mass": np.zeros((5, len(names), 2)),
        }
        for k, (lo, hi) in enumerate(zip(position_edges[:-1], position_edges[1:])):
            result["position_selected_rows"][k] = selected[lo:hi].sum()
            result["position_selected_mass"][k] = weight[lo:hi].sum(0)
        for j in range(len(names)):
            seen = np.maximum.accumulate(changed[:, j])
            age = idx - np.maximum.accumulate(np.where(changed[:, j], idx, 0))
            censored = ~seen
            result["censored_rows"][j] = (censored & selected).sum()
            result["censored_mass"][j] = weight[censored].sum(0)
            for k, threshold in enumerate(thresholds):
                old = age >= threshold
                result["long_age_rows"][j, k] = (old & selected).sum()
                result["long_age_mass"][j, k] = weight[old].sum(0)
                result["censored_long_age_mass"][j, k] = weight[old & censored].sum(0)
            for k, (lo, hi) in enumerate(zip(position_edges[:-1], position_edges[1:])):
                result["position_censored_rows"][k, j] = (censored[lo:hi] & selected[lo:hi]).sum()
                result["position_censored_mass"][k, j] = weight[lo:hi][censored[lo:hi]].sum(0)
        return result

    # Exact synthetic invariant: a never-changing field stays censored; an observed
    # change ends censoring at that row, with zero age there. No labels are fitted.
    _test_change = np.array([False, False, True, False])
    _test_idx = np.arange(4)
    assert (~np.maximum.accumulate(_test_change)).tolist() == [True, True, False, False]
    assert (_test_idx - np.maximum.accumulate(np.where(_test_change, _test_idx, 0))).tolist() == [0, 1, 0, 1]
    return (describe_censoring,)


@app.cell
def bounded_scope(eda_dir, eda_sample, json, mo):
    import hashlib as _scope_hashlib
    eda_scope_r2 = {"id": "EDA-CENSORING-20261024-R2", "parent_plan_sha256": _scope_hashlib.sha256((eda_dir / "EDA_PLAN_R1.json").read_bytes()).hexdigest(), "sample": {"valid": eda_sample["valid"]}, "reason": "The first train-first attempt reached its750MiB worker cap before a completed result; this retry uses the validated smaller validation source within600MiB worker/1GiB aggregate limits.", "prior_new_values_seen": "One fixed validation-sequence probe from the original sample; no fitted scores", "exposure": "Exposed validation; descriptive EDA, not a fresh holdout", "denominator": "need_prediction AND is_scored; weight abs(clip(label,-2,2))", "training_source_status": "not completed"}
    _scope_path = eda_dir / "EDA_PLAN_R2.json"
    if _scope_path.exists():
        assert json.loads(_scope_path.read_text()) == eda_scope_r2
    else:
        _scope_path.write_text(json.dumps(eda_scope_r2, indent=2) + "\n")
    mo.md("**Revised scope:** the same 96 preselected validation sequences. Training-source EDA is deferred after the bounded memory failure; no train/valid comparison is claimed.")

    return (eda_scope_r2,)


@app.cell
def stream_sample(
    data_dir,
    describe_censoring,
    eda_dir,
    eda_scope_r2,
    feature_names,
    json,
    mo,
    np,
    pq,
):
    import hashlib as _hashlib
    import time as _time
    import resource as _resource
    _plan_sha = _hashlib.sha256((eda_dir / "EDA_PLAN_R2.json").read_bytes()).hexdigest()
    _result_path = eda_dir / "CENSORING_SAMPLE_R2.json"
    if _result_path.exists():
        eda_results = json.loads(_result_path.read_text())
        assert eda_results["plan_sha256"] == _plan_sha
    else:
        _start = _time.monotonic()
        eda_results = {"plan_sha256": _plan_sha, "new_fits": 0, "model_evaluations": 0, "sources": {}, "limitations": ["Uniform sequence sample, not a full census", "Train denominator is need_prediction; valid denominator additionally requires is_scored", "All data were previously exposed", "Censoring association does not establish a predictive benefit", "Raw source size/mtime and footer checked; entire raw Parquet payloads not rehashed"]}
        for _source, _groups in eda_scope_r2["sample"].items():
            _pf = pq.ParquetFile(data_dir / f"{_source}.parquet", memory_map=True)
            assert _pf.num_row_groups == {"train": 10607, "valid": 1873}[_source]
            _columns = feature_names + ["seq_ix", "need_prediction", "t0", "t1"]
            if _source == "valid":
                _columns.append("is_scored")
            _aggregate = None
            _members = []
            for _step, _group in enumerate(_groups):
                _table = _pf.read_row_group(_group, columns=_columns, use_threads=True)
                _row = describe_censoring(_table, feature_names)
                _members.append({"row_group": _group, "seq_ix": _row["seq_ix"], "selected_rows": _row["selected_rows"]})
                if _aggregate is None:
                    _aggregate = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in _row.items() if k != "seq_ix"}
                else:
                    for _key in _aggregate:
                        _aggregate[_key] += _row[_key]
                del _table, _row
                if (_step + 1) % 16 == 0:
                    print(f"{_source}: {_step + 1}/{len(_groups)} complete sequences")
            eda_results["sources"][_source] = {"members": _members, "statistics": {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in _aggregate.items()}}
            del _pf, _aggregate
        eda_results["wall_seconds"] = _time.monotonic() - _start
        eda_results["process_peak_rss_kib"] = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss
        _result_path.write_text(json.dumps(eda_results, indent=2) + "\n")
    mo.md(f"Completed **{sum(len(v['members']) for v in eda_results['sources'].values())} sequences** in {eda_results['wall_seconds']:.1f}s. Peak kernel RSS: {eda_results['process_peak_rss_kib']/1024:.0f} MiB. Results saved with the frozen sample-plan hash.")
    return (eda_results,)


@app.cell
def age_controls(eda_results, feature_names, mo, np):
    age_stats = eda_results["sources"]["valid"]["statistics"]
    age_thresholds = np.array([1, 16, 64, 512, 4096, 10000])
    feature_selector = mo.ui.dropdown(feature_names, value="i0_p12", label="Field")
    target_selector = mo.ui.dropdown({"t0 magnitude": 0, "t1 magnitude": 1}, value="t0 magnitude", label="Weight view")
    mo.hstack([feature_selector, target_selector], justify="start")
    return age_stats, age_thresholds, feature_selector, target_selector


@app.cell
def age_overview(age_stats, age_thresholds, eda_dir, feature_names, mo, np):
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 180, "font.size": 10})
    _age_rows = np.asarray(age_stats["long_age_rows"], dtype=float)
    _censored_rows = np.asarray(age_stats["censored_rows"], dtype=float)
    _top = np.argsort(_censored_rows)[-12:]
    _summary_fig, _axes = plt.subplots(1, 2, figsize=(11, 4.0), constrained_layout=True)
    _axes[0].barh(np.asarray(feature_names)[_top], 100 * _censored_rows[_top] / age_stats["selected_rows"], color="#236B8E")
    _axes[0].set(xlabel="Scored rows censored at sequence start (%)", title="Most affected fields")
    _quantiles = np.quantile(100 * _age_rows / age_stats["selected_rows"], [0.1, 0.5, 0.9], axis=0)
    _axes[1].fill_between(age_thresholds, _quantiles[0], _quantiles[2], alpha=.2, color="#236B8E", label="10th–90th percentile across fields")
    _axes[1].plot(age_thresholds, _quantiles[1], "o-", color="#236B8E", label="Median field")
    _axes[1].set(xscale="log", xlabel="Rows since last observed field change", ylabel="Scored-row coverage (%)", title="How much long-age signal exists?")
    _axes[1].legend(fontsize=8)
    _summary_fig.savefig(eda_dir / "censoring_overview_r2.png")
    _summary_fig.savefig(eda_dir / "censoring_overview_r2.svg")
    _overview_display = mo.mpl.interactive(_summary_fig)
    plt.close(_summary_fig)
    _overview_display
    return (plt,)


@app.cell
def age_detail(
    age_stats,
    age_thresholds,
    feature_names,
    feature_selector,
    mo,
    np,
    plt,
    target_selector,
):
    _field_index = feature_names.index(feature_selector.value)
    _target_index = target_selector.value
    _selected_mass = np.asarray(age_stats["selected_mass"])[_target_index]
    _old_mass = np.asarray(age_stats["long_age_mass"])[_field_index, :, _target_index]
    _censored_old_mass = np.asarray(age_stats["censored_long_age_mass"])[_field_index, :, _target_index]
    _age_fig, _age_axes = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
    _age_axes[0].plot(age_thresholds, 100 * _old_mass / _selected_mass, "o-", label="All observed-age rows")
    _age_axes[0].plot(age_thresholds, 100 * _censored_old_mass / _selected_mass, "o-", label="Still censored by sequence start")
    _age_axes[0].set(xscale="log", xlabel="Age threshold (rows)", ylabel="Clipped-label magnitude mass (%)", title=feature_selector.value)
    _age_axes[0].legend(fontsize=8)
    _position_mass = np.asarray(age_stats["position_selected_mass"])[:, _target_index]
    _position_censored = np.asarray(age_stats["position_censored_mass"])[:, _field_index, _target_index]
    _position_pct = np.divide(100 * _position_censored, _position_mass, out=np.zeros(5), where=_position_mass > 0)
    _age_axes[1].bar(["0–98", "99–499", "500–1999", "2000–9999", "10000–19999"], _position_pct, color="#C47731")
    _age_axes[1].tick_params(axis="x", labelrotation=25)
    _age_axes[1].set(xlabel="Sequence position", ylabel="Within-bin weighted censoring (%)", title="Boundary effect by position")
    _age_display = mo.mpl.interactive(_age_fig)
    plt.close(_age_fig)
    _age_display
    return


@app.cell
def age_interpretation(age_stats, feature_names, mo, np):
    _mean_censoring = 100 * np.asarray(age_stats["censored_rows"]).sum() / (age_stats["selected_rows"] * len(feature_names))
    _max_censoring = 100 * np.asarray(age_stats["censored_rows"]).max() / age_stats["selected_rows"]
    mo.md(f"""
    ## What this changes

    Across **{age_stats['selected_rows']:,} scored rows**, a uniformly selected field–row pair is censored **{_mean_censoring:.4f}%** of the time; the most affected single field reaches **{_max_censoring:.3f}%**. These are field–row denominators, not the fraction of rows with any censored field. In this sample, no censored field contributes label-magnitude mass at observed age ≥10,000.

    This weakens the idea that sequence-start censoring alone explains a broad age-feature benefit. It leaves open short-age update activity, feature-group interactions and whether the representation helps a properly trained model. The rare cases may still matter, and this sample does not establish their population frequency.

    The shaded band describes variation **across fields**, not sampling confidence. Weighting by clipped-label magnitude describes where the metric places mass; it does not measure a model's score. This is exposed-validation EDA, with **zero new fits**. The planned training-source comparison remains incomplete after the memory-limited attempt.
    """)
    return


@app.cell
def online_grade_plot(eda_dir, json, mo, plt, repo):
    online_grade_path = repo / "research/campaign_2026-10-24/resume4h_r1/feasibility_scout/F0_EXPLORATORY_GRADE_R2.json"
    online_grade = json.loads(online_grade_path.read_text())
    online_primary = [c for c in online_grade["contrasts"] if c["role"] == "primary"]
    _online_fig, _online_ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    for _i, _contrast in enumerate(online_primary):
        _point = _contrast["point_wp_delta"]
        _lo, _hi = _contrast["paired_component_bootstrap"]["ci95"]
        _online_ax.plot([1e4*_lo, 1e4*_hi], [_i, _i], color="#236B8E", linewidth=2)
        _online_ax.plot(1e4*_point, _i, "o", color="#236B8E")
    _online_ax.axvline(0, color="0.5", linestyle="--", linewidth=1)
    _online_ax.set(yticks=range(len(online_primary)), yticklabels=[c["name"] for c in online_primary], xlabel="Official score difference ×10,000", title="F0 online relationships: four prespecified primary contrasts")
    _online_ax.invert_yaxis()
    _online_fig.savefig(eda_dir / "f0_online_contrasts_r2.png")
    _online_fig.savefig(eda_dir / "f0_online_contrasts_r2.svg")
    _online_display = mo.mpl.interactive(_online_fig)
    plt.close(_online_fig)
    _online_display
    return (online_grade,)


@app.cell
def online_grade_interpretation(mo):
    mo.md("""
    ## A completed grade changes the next question

    The seven saved F0 endpoints are now graded with the unchanged official metric. The strongest head scores **0.68060978**, versus its frozen parent at **0.68115415**. Adding the online relationships changes the mean-only head by only about **−0.000068 to +0.000013**, depending on the contrast and lag. The larger question is why training the correction reduces the parent's score. The architecture already freezes the parent and adds a zero-initialized residual; zero initialization alone did not prevent the eventual loss.

    The chart shows **conditional component-bootstrap 95% intervals** for four prespecified primary contrasts, given these fixed predictions: 468 evaluation sequences, 141 components, one exposed fold and one fitting seed. It does not estimate seed, fold or public-transfer uncertainty. The original two-fold twelve-fit panel remains incomplete; no new model was fitted for this grade.

    A negative point estimate or an interval spanning zero does not falsify online adaptation. It lowers the priority of repeating this exact correction recipe. Fit-side parent optimism, objective mismatch and correction regularization are alternative explanations to test; none is established by this grade.
    """)
    return


@app.cell
def clock_overlap(age_stats, eda_dir, json, mo, np, plt, repo):
    clock_result_path = repo / "research/campaign_2026-10-24/resume4h_r1/feasibility_scout/clock_eda/CLOCK_STATISTICS_R1.json"
    clock_results = json.loads(clock_result_path.read_text())
    clock_matrix = np.asarray(clock_results["pairwise_update_events"]["jaccard"])
    clock_family_names = clock_results["position_activity_age"]["feature_group_order"]
    assert clock_results["eligible_rows"] == age_stats["selected_rows"]
    assert np.array_equal(np.asarray(clock_results["position_activity_age"]["censored_counts_by_feature"]), np.asarray(age_stats["position_censored_rows"]))
    _clock_family = np.zeros((9, 9))
    for _pair in clock_results["pairwise_update_events"]["group_pair_mean_jaccard"]:
        _i = clock_family_names.index(_pair["group_a"])
        _j = clock_family_names.index(_pair["group_b"])
        _clock_family[_i, _j] = _clock_family[_j, _i] = _pair["mean_jaccard"]
    _clock_fig, _clock_axes = plt.subplots(1, 2, figsize=(12, 5.0), constrained_layout=True)
    _clock_axes[0].imshow(clock_matrix, vmin=0, vmax=1, cmap="viridis", interpolation="nearest")
    _boundaries = np.cumsum([22, 22, 4, 4, 22, 22, 4, 4, 8])
    _centers = (_boundaries + np.r_[0, _boundaries[:-1]] - 1) / 2
    _clock_axes[0].set(xticks=_centers, yticks=_centers, xticklabels=clock_family_names, yticklabels=clock_family_names, title="Field update-event overlap")
    for _boundary in _boundaries[:-1] - .5:
        _clock_axes[0].axhline(_boundary, color="white", linewidth=.4)
        _clock_axes[0].axvline(_boundary, color="white", linewidth=.4)
    _clock_image = _clock_axes[1].imshow(_clock_family, vmin=0, vmax=1, cmap="viridis", interpolation="nearest")
    _clock_axes[1].set(xticks=range(9), yticks=range(9), xticklabels=clock_family_names, yticklabels=clock_family_names, title="Mean pairwise overlap within/between families")
    for _i in range(9):
        for _j in range(9):
            _clock_axes[1].text(_j, _i, f"{_clock_family[_i,_j]:.2f}", ha="center", va="center", color="black" if _clock_family[_i,_j]>.65 else "white", fontsize=8)
    for _axis in _clock_axes:
        _axis.tick_params(axis="x", labelrotation=65)
    _clock_fig.colorbar(_clock_image, ax=_clock_axes, label="Jaccard of changed fields on scored rows", shrink=.8)
    _clock_fig.savefig(eda_dir / "update_clock_overlap_r1.png")
    _clock_fig.savefig(eda_dir / "update_clock_overlap_r1.svg")
    _clock_display = mo.mpl.interactive(_clock_fig)
    plt.close(_clock_fig)
    _clock_display
    return (clock_matrix,)


@app.cell
def clock_neighbors(clock_matrix, feature_names, feature_selector, mo, np):
    _clock_index = feature_names.index(feature_selector.value)
    _clock_nearest = [int(i) for i in np.argsort(clock_matrix[_clock_index])[::-1] if i != _clock_index][:10]
    _clock_rows = "\n".join(f"| {feature_names[i]} | {clock_matrix[_clock_index,i]:.6f} |" for i in _clock_nearest)
    mo.md("## Fields sharing this update clock\n\nSelected field: **" + feature_selector.value + "**. This uses the same selector as the age curves.\n\n| Other field | Scored-row event Jaccard |\n|---|---:|\n" + _clock_rows + "\n\nJaccard counts co-occurring changes, not unchanged rows. Equality on scored rows does not prove equality across every row or equality of the underlying field values. The nine displayed families separate `dp` from `dv`; the current experimental G arm merges those into seven clock groups.")
    return


@app.cell
def clock_geometry_plot(age_stats, eda_dir, json, mo, plt, repo):
    clock_geometry = json.loads((repo / "research/campaign_2026-10-24/resume4h_r1/feasibility_scout/clock_eda/CLOCK_GEOMETRY_RESULT_R1.json").read_text())
    assert clock_geometry["selected_rows"] == age_stats["selected_rows"]
    _group_names = ["seven_existing_groups", "nine_named_families"]
    _group_labels = ["Current 7 groups", "Separate dp/dv: 9 groups"]
    _geometry_fig, _geometry_axes = plt.subplots(1, 2, figsize=(10, 3.4), constrained_layout=True)
    _geometry_axes[0].bar(_group_labels, [100*clock_geometry["grouping_age_comparison"][g]["unequal_pair_rate"] for g in _group_names], color=["#C47731", "#236B8E"])
    _geometry_axes[0].set(ylabel="Scored field–row pairs with altered age (%)", title="Grouping changes some clocks")
    _geometry_axes[1].bar(_group_labels, [clock_geometry["grouping_age_comparison"][g]["mean_normalized_log_age_gap"] for g in _group_names], color=["#C47731", "#236B8E"])
    _geometry_axes[1].set(ylabel="Mean normalized log-age difference", title="Cap 10,000; normalization by log(10,001)")
    for _axis in _geometry_axes:
        _axis.tick_params(axis="x", labelrotation=10)
    _geometry_fig.savefig(eda_dir / "clock_grouping_geometry_r1.png")
    _geometry_fig.savefig(eda_dir / "clock_grouping_geometry_r1.svg")
    _geometry_display = mo.mpl.interactive(_geometry_fig)
    plt.close(_geometry_fig)
    _geometry_display
    return (clock_geometry,)


@app.cell
def clock_geometry_interpretation(clock_geometry, feature_selector, mo):
    _exact_class = next(c for c in clock_geometry["exact_event_classes"] if feature_selector.value in c)
    mo.md(f"""
    ## Whole-sequence check: overlap is real, but not universal

    The 112 fields form **{len(clock_geometry['exact_event_classes'])} distinct event-pattern classes** across all **1.92 million sampled rows**, including unscored rows. Several eleven-field price blocks share exactly the same event pattern in this sample. Some four-field blocks that looked identical on scored rows differ by a single event elsewhere; scored-row Jaccard alone would have missed that.

    For selected field **{feature_selector.value}**, the complete-sample matching class is: {', '.join(_exact_class)}.

    Replacing individual ages with the current seven group clocks changes **14.39%** of scored field–row ages. Separating `dp` and `dv` into nine groups reduces that to **11.53%**. Thus “group age versus field age” also tests a particular grouping decision. It does not isolate the value of 112 independent clocks.

    These hashes certify equality only on the observed sample. They do not prove future equivalence, equality of field values, or a model improvement. The nine-group construction is a descriptive alternative, not an added arm in the frozen age experiment.
    """)
    return


@app.cell(hide_code=True)
def correction_shrink_plot(eda_dir, json, mo, online_grade, plt, repo):
    shrink_grade = json.loads((repo / "research/campaign_2026-10-24/resume4h_r1/feasibility_scout/correction_shrink_r1/CORRECTION_SHRINK_GRADE_R3.json").read_text())
    shrink_primary = [c for c in shrink_grade["contrasts"] if c["role"] == "primary"]
    _shrink_cells = shrink_grade["shrink"]["unique_candidates"]
    _raw_lookup = {c["candidate"]: c for c in online_grade["contrasts"] if c["reference"] == "parent"}
    _shrink_lookup = {c["candidate"].removeprefix("shrink_"): c for c in shrink_primary}
    _shrink_fig, _shrink_axes = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True, sharey=True)
    for _ax, _lookup, _color, _title in zip(_shrink_axes, [_raw_lookup, _shrink_lookup], ["#B44E35", "#237D78"], ["Original correction: scale 1", "Fixed additional correction scale 0.1"]):
        for _row, _name in enumerate(_shrink_cells):
            _contrast = _lookup[_name]
            _lo, _hi = _contrast["paired_component_bootstrap"]["ci95"]
            _ax.plot([1e4*_lo, 1e4*_hi], [_row, _row], color=_color, linewidth=2)
            _ax.plot(1e4*_contrast["point_wp_delta"], _row, "o", color=_color)
        _ax.axvline(0, color="0.5", linestyle="--", linewidth=1)
        _ax.set(yticks=range(5), yticklabels=_shrink_cells, xlabel="Official score difference from parent ×10,000", title=_title)
    _shrink_axes[0].invert_yaxis()
    _shrink_fig.suptitle("F0 correction scale: conditional component 95% intervals; different x-axis ranges", fontsize=12)
    _shrink_fig.savefig(eda_dir / "correction_shrink_f0_r3.png", dpi=150)
    _shrink_fig.savefig(eda_dir / "correction_shrink_f0_r3.svg")
    _shrink_display = mo.mpl.interactive(_shrink_fig)
    plt.close(_shrink_fig)
    _shrink_display
    return (shrink_primary,)


@app.cell(hide_code=True)
def correction_shrink_interpretation(mo):
    mo.md("""
    ## Smaller corrections retain a modest positive direction

    A fixed **0.1 multiplier on the exported correction** changes all five parent-relative differences from negative to **+0.000129 to +0.000136**. The mean-only A control gains **+0.000131**, nearly the same as B and C. This makes correction scale a useful follow-up and weakens a relationship-specific explanation for this particular gain.

    The architecture already multiplies its learned residual by 0.1; this diagnostic adds another factor of 0.1 to the exported correction. We evaluated one coefficient uniformly, selected after the original F0 grade, without a coefficient search or new fitting. This is exploratory evidence on the same exposed F0/seed1001 cell. The intervals are conditional on fixed predictions and all span zero; they are not simultaneous bounds across the five correlated endpoints.

    This does not establish optimizer overshoot or the cause of the original loss. A shared shrink benefit can arise from regularization, prediction geometry or a fit/evaluation residual mismatch. The next inexpensive check is the same frozen coefficient on the already saved F1 mean-only control, with its own ancestry verified before reading labels.
    """)
    return


@app.cell(hide_code=True)
def correction_time_controls(mo):
    correction_target = mo.ui.dropdown({"Official mean score": "wp", "Target t0": "t0", "Target t1": "t1"}, value="Official mean score", label="Position diagnostic")
    correction_target
    return (correction_target,)


@app.cell(hide_code=True)
def correction_time_plot(
    correction_target,
    eda_dir,
    mo,
    np,
    plt,
    shrink_primary,
):
    _correction_bands = ["rows_99_500", "rows_500_2000", "rows_2000_10000", "rows_10000_20000"]
    _correction_values = np.array([[c["time_bin_deltas"][b][correction_target.value] for b in _correction_bands] for c in shrink_primary]) * 1e4
    _correction_extent = max(float(np.abs(_correction_values).max()), 1e-9)
    _correction_fig, _correction_ax = plt.subplots(figsize=(9, 4.2), constrained_layout=True)
    _correction_im = _correction_ax.imshow(_correction_values, aspect="auto", cmap="RdBu", vmin=-_correction_extent, vmax=_correction_extent)
    _correction_ax.set(xticks=range(4), xticklabels=["99–499", "500–1,999", "2,000–9,999", "10,000–19,999"], yticks=range(5), yticklabels=[c["candidate"].removeprefix("shrink_") for c in shrink_primary], xlabel="Position in sequence", title="Fixed shrink − parent by position (descriptive exposed-cell slices)")
    for _r in range(5):
        for _c in range(4):
            _correction_ax.text(_c, _r, f"{_correction_values[_r, _c]:+.2f}", ha="center", va="center", color="white" if abs(_correction_values[_r, _c]) > .65*_correction_extent else "black")
    _correction_fig.colorbar(_correction_im, ax=_correction_ax, label="Score difference ×10,000")
    if correction_target.value == "wp":
        _correction_fig.savefig(eda_dir / "correction_shrink_positions_r3.png", dpi=150)
        _correction_fig.savefig(eda_dir / "correction_shrink_positions_r3.svg")
    _correction_display = mo.mpl.interactive(_correction_fig)
    plt.close(_correction_fig)
    _correction_display
    return


@app.cell(hide_code=True)
def control_interpretation(mo):
    mo.md("""
    ## What does the control actually remove?

    **A still adapts online.** “Mean-only” describes its relation estimator, not its entire input. All three heads receive 211 raw/event features plus 208 state features. A keeps matured input movements, an evolving mean forecast and its error; only the slope channels are forced to zero. These statistics use past input changes, not target-label feedback.

    | Arm | Relation estimator | Status |
    |---|---|---|
    | Parent | No added readout correction | Evaluated reference |
    | A | Continuing mean updates; zero slope | Saved F0 and F1 head |
    | B | Slope/intercept updates through row98, then frozen | Saved F0 heads |
    | C | Continuing slope/intercept updates | Saved F0 heads |
    | Same-width head with208 state channels zeroed | Raw/event readout capacity only | Proposed control; not fitted |

    The A comparison can challenge the added value of slope/intercept relationships. It cannot by itself distinguish useful online statistics from generic readout capacity, nor can it reject online adaptation broadly. A matching raw/event-only head would answer that separate question.
    """)
    return


@app.cell(hide_code=True)
def f0_f1_fixed_shrink(eda_dir, json, mo, plt, repo, shrink_primary):
    _f1 = json.loads((repo / "research/campaign_2026-10-24/resume4h_r1/feasibility_scout/f1_mean_shrink_r1/F1_MEAN_SHRINK_GRADE_R2.json").read_text())
    _f0_a = next(c for c in shrink_primary if c["candidate"] == "shrink_A_lag1")
    _f1_a = next(c for c in _f1["contrasts"] if c["role"] == "primary")
    _fig, _ax = plt.subplots(figsize=(9, 3.2), layout="constrained")
    for _row, (_label, _contrast, _color) in enumerate([("F0 / seed 1001", _f0_a, "#2563eb"), ("F1 / seed 1002", _f1_a, "#d97706")]):
        _lo, _hi = _contrast["paired_component_bootstrap"]["ci95"]
        _ax.plot([1e4 * _lo, 1e4 * _hi], [_row, _row], color=_color, linewidth=3)
        _ax.plot(1e4 * _contrast["point_wp_delta"], _row, "o", color=_color)
    _ax.axvline(0, color="#6b7280", linewidth=1, linestyle="--")
    _ax.set(yticks=[0, 1], yticklabels=["F0 / seed 1001", "F1 / seed 1002"], xlabel="Fixed 0.1 mean-only correction − parent WP (×10⁻⁴)", title="F1 is essentially flat; F0's small point gain is not reproduced", ylim=(-0.6, 1.6))
    _ax.grid(axis="x", alpha=0.2)
    _fig.savefig(eda_dir / "f0_f1_fixed_shrink_r1.png", dpi=160)
    _fig.savefig(eda_dir / "f0_f1_fixed_shrink_r1.svg")
    _widget = mo.mpl.interactive(_fig)
    plt.close(_fig)
    mo.vstack([_widget, mo.md("Both panels are previously exposed. Intervals resample connected components conditional on saved models; they do not capture training or public-transfer uncertainty. F1 was checked after seeing F0. Different fold and seed prevent attributing the difference to either axis. No new predictor was fitted.")])
    return


@app.cell(hide_code=True)
def quota_package_comparison(eda_dir, json, mo, plt, repo):
    _quota = json.loads((repo / "research/campaign_2026-10-24/resume4h_r1/quota_submission_r1/QUOTA_GRADE_R1.json").read_text())
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained", gridspec_kw={"width_ratios": [2, 1]})
    _labels = []
    for _row, (_fold, _idx, _color) in enumerate([("0", 0, "#2563eb"), ("0", 1, "#16a34a"), ("2", 0, "#2563eb"), ("2", 1, "#16a34a")]):
        _contrast = _quota["folds"][_fold]["contrasts"][_idx]
        _lo, _hi = _contrast["component_bootstrap"]["ci95"]
        _axes[0].plot([1e3*_lo, 1e3*_hi], [_row, _row], color=_color, linewidth=2.5)
        _axes[0].plot(1e3*_contrast["point_deltas"]["wp"], _row, "o", color=_color)
        _labels.append("F"+_fold+" / "+("S3" if _idx == 0 else "S3+S4"))
    _axes[0].axvline(0, color="#6b7280", linestyle="--", linewidth=1)
    _axes[0].set(yticks=range(4), yticklabels=_labels, xlabel="Native package − TL3 WP (×10⁻³)", title="Fixed 32-component samples per stratum")
    _axes[0].grid(axis="x", alpha=0.2)
    _axes[1].bar(["TL3", "S3", "S3+S4"], [128.720300587, 133.058952649, 159.779462191], color=["#6b7280", "#2563eb", "#16a34a"])
    _axes[1].set(ylabel="Callback seconds / 1.28M rows", title="Same host, one CPU core")
    _fig.suptitle("Exploratory quota candidates: fitted diagnostics, not clean CV")
    _fig.savefig(eda_dir / "quota_package_comparison_r1.png", dpi=160)
    _fig.savefig(eda_dir / "quota_package_comparison_r1.svg")
    _widget = mo.mpl.interactive(_fig)
    plt.close(_fig)
    mo.vstack([_widget, mo.md("S3+S4 leads S3 in both point comparisons, with intervals spanning zero for that difference. H390 trained on supplied validation; TL3 teacher ancestry is unresolved. These intervals condition on the saved models. Runtime is a local sample measurement, not a competition-runtime guarantee. Both ZIPs are linked in the quota handoff report.")])
    return


@app.cell(hide_code=True)
def direction_geometry(eda_dir, json, mo, np, plt, repo):

    _direction = json.loads((repo / "research/campaign_2026-10-24/resume4h_r1/feasibility_scout/direction_probe_r1/DIRECTION_PROBE_RESULT_R2.json").read_text())
    _rows = [("F0 " + n.replace("_", " "), e["derivatives"]) for n, e in _direction["folds"]["F0"]["endpoints"].items()]
    _fig, _axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    _y = np.arange(len(_rows))
    _cal = [100 * np.mean([p["alpha0_projection"]["calibration"]["projection_norm_share"] for p in d["per_target"]]) for _, d in _rows]
    _score_share = [100 * np.mean([p["alpha0_projection"]["calibration"]["projected_derivative"] for p in d["per_target"]]) / d["D_at_0_wp_mean"] for _, d in _rows]
    _axes[0].barh(_y - .18, _cal, height=.32, label="Correction variance share", color="#4c78a8")
    _axes[0].barh(_y + .18, _score_share, height=.32, label="Score derivative share", color="#f58518")
    _axes[0].set_yticks(_y, [n for n, _ in _rows]); _axes[0].invert_yaxis()
    _axes[0].set_xlabel("Percent explained by calibration span")
    _axes[0].set_title("Large variance share, tiny score contribution")
    _axes[0].legend(loc="lower right", fontsize=8)
    _axes[0].text(.98, .98, "Orange bars: only 0.078–0.081% of score slope", transform=_axes[0].transAxes, ha="right", va="top", fontsize=8, color="#b76312")
    _bcs = [(n, d) for n, d in _rows if " A " not in n]
    _vals = [1e4 * np.mean([p["alpha0_projection"]["calibration_plus_A"]["remaining_derivative"] for p in d["per_target"]]) for _, d in _bcs]
    _ci = np.array([d["bootstrap"]["intervals"]["calibration_plus_A_remaining_derivative_wp_mean"]["ci95"] for _, d in _bcs]) * 1e4
    _axes[1].errorbar(_vals, np.arange(len(_bcs)), xerr=np.vstack([np.array(_vals) - _ci[:, 0], _ci[:, 1] - np.array(_vals)]), fmt="o", color="#54a24b", capsize=4)
    _axes[1].axvline(0, color="gray", linewidth=1)
    _axes[1].set_yticks(np.arange(len(_bcs)), [n for n, _ in _bcs]); _axes[1].invert_yaxis()
    _axes[1].set_xlabel("Remaining score derivative × 10⁻⁴ per unit alpha")
    _axes[1].set_title("Beyond calibration + common A: uncertain")
    _fig.suptitle("Existing F0 corrections: exposed diagnostic geometry", fontsize=13)
    for _ext in ("png", "svg"):
        _fig.savefig(eda_dir / f"direction_geometry_r2.{_ext}", dpi=160, bbox_inches="tight")
    _widget = mo.mpl.interactive(_fig)
    plt.close(_fig)
    mo.vstack([_widget, mo.md("Conditional component intervals; these slopes are not measured endpoint gains or public forecasts. F1 A has a much weaker slope and its fixed 0.1 correction remains essentially flat. No coefficients were selected or exported.")])

    return


@app.cell(hide_code=True)
def batch_weight_geometry(eda_dir, json, mo, np, plt, repo):

    _batch_audit = json.loads((repo / "research/campaign_2026-10-24/resume4h_r1/feasibility_scout/batch_mass_audit_r1/BATCH_MASS_RESULT_R1.json").read_text())
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4.1), constrained_layout=True)
    _labels = [f"F{f['fold']} / seed {f['seed']}" for f in _batch_audit["folds"]]
    for _i, _fold in enumerate(_batch_audit["folds"]):
        _d = _fold["measure"]["distortion_factor_mean_batch_mass_over_batch_mass"]
        _axes[0].plot([_d["min"], _d["max"]], [_i, _i], color="#a9c4dd", linewidth=3)
        _axes[0].plot([_d["p05"], _d["p95"]], [_i, _i], color="#4c78a8", linewidth=8, solid_capstyle="butt")
        _axes[0].plot(_d["median"], _i, "o", color="#f58518", markersize=8)
    _axes[0].set_xscale("log"); _axes[0].axvline(1, color="gray", linestyle="--")
    _axes[0].set_yticks([0, 1], _labels); _axes[0].set_ylim(-.55, 1.55)
    _axes[0].set_xlabel("Relative row weight: equal-batch / pooled measure")
    _axes[0].set_title("Wide batch reweighting in both cells")
    _axes[0].text(.02, .98, "Thin: min–max · Thick: 5–95% · Dot: median", transform=_axes[0].transAxes, va="top", fontsize=8)
    _pooled = [f["component_concentration"]["pooled_mass_measure"]["effective_component_count"] for f in _batch_audit["folds"]]
    _equal = [f["component_concentration"]["equal_batch_normalized_measure"]["effective_component_count"] for f in _batch_audit["folds"]]
    _x = np.arange(2)
    _axes[1].bar(_x - .18, _pooled, width=.34, label="Pooled", color="#4c78a8")
    _axes[1].bar(_x + .18, _equal, width=.34, label="Equal-batch", color="#54a24b")
    _axes[1].set_xticks(_x, _labels); _axes[1].set_ylabel("Inverse HHI of component weight shares")
    _axes[1].set_title("Equal-batch weights are less concentrated")
    _axes[1].legend()
    _fig.suptitle("Actual fitting weights: a measured difference, not proof of harm", fontsize=13)
    for _ext in ("png", "svg"):
        _fig.savefig(eda_dir / f"batch_weight_geometry_r1.{_ext}", dpi=160, bbox_inches="tight")
    _widget = mo.mpl.interactive(_fig)
    plt.close(_fig)
    mo.vstack([_widget, mo.md("All fitting rows from both pinned suffixes; no evaluation labels or fits. Inverse HHI measures concentration, not independent replications. Neither panel establishes which normalization improves official WP.")])

    return


if __name__ == "__main__":
    app.run()
