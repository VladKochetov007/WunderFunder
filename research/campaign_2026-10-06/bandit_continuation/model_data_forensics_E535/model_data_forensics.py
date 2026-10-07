import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    # Replot frozen 24-group summaries with the correct interpretation of p-column identifiers.
    import json as _json
    from pathlib import Path as _Path
    import matplotlib.pyplot as _plt
    _e535_root = _Path("research/campaign_2026-10-06/bandit_continuation/model_data_forensics_E535")
    _e535_data = _json.loads((_e535_root / "E535_SMALL_VALID_ROW_GROUP_STREAM_V1.json").read_text())
    _e535_records = _e535_data["records"]
    _e535_roles = [("H230-fit-valid", "H230 fit-valid (n=12)", "#2878B5"), ("H230-excluded-valid", "H230 excluded-eval (n=12)", "#D95F02")]
    _e535_fig, _e535_ax = _plt.subplots(1, 2, figsize=(13.5, 4.8), constrained_layout=True)
    for _e535_key, _e535_label, _e535_color in _e535_roles:
        _e535_rows = [r for r in _e535_records if r["role"] == _e535_key]
        _e535_x = [r["i0_best_price_events_after_warmup99"] / 19.901 for r in _e535_rows]
        _e535_y = [r["target_events_after_warmup99"] / 19.901 for r in _e535_rows]
        _e535_ax[0].scatter(_e535_x, _e535_y, color=_e535_color, label=_e535_label, alpha=.9)
    _e535_ax[0].set(xlabel="paired i0_p0 / i0_p11 price-code changes per 1,000 post-warmup rows", ylabel="t0/t1 target changes per 1,000 post-warmup rows", title="Input-code event proxy vs target-change clock")
    _e535_ax[0].grid(alpha=.25); _e535_ax[0].legend(frameon=False)
    _e535_box = [[r["median_target_run_rows_after_warmup99"] for r in _e535_records if r["role"] == key] for key, _, _ in _e535_roles]
    _e535_ax[1].boxplot(_e535_box, tick_labels=[label for _, label, _ in _e535_roles], showfliers=True)
    _e535_ax[1].set(ylabel="median target plateau length within sequence (rows)", title="Per-sequence target dwell time")
    _e535_ax[1].tick_params(axis="x", rotation=9); _e535_ax[1].grid(axis="y", alpha=.25)
    _e535_fig.suptitle("Fold0 valid data: deterministic 12 + 12 row-group sample, seed 535")
    _e535_out = _e535_root / "plots"
    _e535_out.mkdir(parents=True, exist_ok=True)
    _e535_fig.savefig(_e535_out / "E535_event_and_target_clocks_corrected_V1.png", dpi=180)
    _e535_fig.savefig(_e535_out / "E535_event_and_target_clocks_corrected_V1.svg")
    _e535_fig
    return


@app.cell(hide_code=True)
def bounded_valid_row_group_diagnostic():
    import hashlib
    import json
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pyarrow.parquet as pq

    project = Path('/home/vlad/development/wunder/wnn_connectome_starterpack')
    lineage = Path.cwd() / 'research/campaign_2026-10-06/bandit_continuation/model_data_forensics_E535'
    plots = lineage / 'plots'
    plots.mkdir(parents=True, exist_ok=True)
    split_path = project / 'research/campaign_2026-09-29/artifacts/splits/fold0.json'
    split = json.loads(split_path.read_text())
    fit_valid = split['fit']['valid']
    eval_valid = split['evaluation']['valid']
    seq_by_row_group = {int(row['row_group']): int(row['seq_ix']) for row in split['sequences']['valid']}
    rng = np.random.default_rng(535)
    sampled = {
        'H230-fit-valid': np.sort(rng.choice(fit_valid, size=12, replace=False)).astype(int).tolist(),
        'H230-excluded-valid': np.sort(rng.choice(eval_valid, size=12, replace=False)).astype(int).tolist(),
    }
    valid_path = project / 'datasets/valid.parquet'
    pf = pq.ParquetFile(valid_path)
    assert pf.metadata.num_row_groups == len(split['sequences']['valid']) == 1873
    assert pf.metadata.num_rows == 1873 * 20000
    columns = ['seq_ix', 'i0_p0', 'i0_p11', 't0', 't1', 'is_scored']
    target_quanta = np.array([0.12543, 0.24133, 0.46854, 0.65687], dtype=np.float64)
    records = []
    for role, groups in sampled.items():
        for group in groups:
            table = pf.read_row_group(int(group), columns=columns, use_threads=False)
            arrays = {name: table.column(name).to_numpy(zero_copy_only=False) for name in columns}
            assert len(arrays['seq_ix']) == 20000
            observed_seq_ix = int(arrays['seq_ix'][0])
            expected_seq_ix = seq_by_row_group[int(group)]
            targets = np.column_stack((arrays['t0'], arrays['t1'])).astype(np.float32, copy=False)
            scored = np.asarray(arrays['is_scored'], dtype=bool).copy()
            scored[:99] = False
            # Row-level target events and the project's i0 best-price event definition.
            target_changed = np.zeros(20000, dtype=bool)
            target_changed[1:] = np.any(targets[1:] != targets[:-1], axis=1)
            px_changed = np.zeros(20000, dtype=bool)
            px_changed[1:] = ((arrays['i0_p0'][1:] != arrays['i0_p0'][:-1]) &
                              (arrays['i0_p11'][1:] != arrays['i0_p11'][:-1]))
            target_events = np.flatnonzero(target_changed[99:])
            target_run_lengths = np.diff(np.r_[0, target_events, len(target_changed) - 99]).astype(np.int32)
            abs_levels = np.unique(np.abs(targets[99:].astype(np.float64)))
            gaps = np.diff(abs_levels)
            gaps = gaps[gaps > 1e-6]
            quantum = float(gaps.min()) if gaps.size else float('nan')
            market_class = int(np.argmin(np.abs(target_quanta - quantum))) if np.isfinite(quantum) else -1
            if market_class >= 0 and abs(target_quanta[market_class] - quantum) >= 1e-3:
                market_class = -1
            sample_hash = hashlib.sha256()
            for name in columns:
                sample_hash.update(np.ascontiguousarray(arrays[name]).tobytes())
            records.append({
                'role': role,
                'row_group': int(group),
                'expected_seq_ix': expected_seq_ix,
                'observed_seq_ix': observed_seq_ix,
                'identity_match': observed_seq_ix == expected_seq_ix,
                'sampled_rows': len(arrays['seq_ix']),
                'scored_rows_after_warmup99': int(scored.sum()),
                'target_events_after_warmup99': int(target_changed[99:].sum()),
                'i0_best_price_events_after_warmup99': int(px_changed[99:].sum()),
                'median_target_run_rows_after_warmup99': float(np.median(target_run_lengths)),
                'p90_target_run_rows_after_warmup99': float(np.quantile(target_run_lengths, 0.9)),
                'target_quantum': quantum,
                'market_class': market_class,
                'finite_targets': bool(np.isfinite(targets).all()),
                'sample_columns_sha256': sample_hash.hexdigest(),
            })
            del table, arrays, targets, scored, target_changed, px_changed, target_run_lengths

    assert all(row['identity_match'] for row in records)
    summary = {
        'id': 'E535_SMALL_VALID_ROW_GROUP_STREAM_V1',
        'selection_seed': 535,
        'split_manifest': str(split_path.relative_to(project)),
        'split_manifest_sha256': hashlib.sha256(split_path.read_bytes()).hexdigest(),
        'data_path': str(valid_path.relative_to(project)),
        'data_read_mode': 'ParquetFile.read_row_group, only six columns, one 20,000-row group at a time; no full table/prediction arrays',
        'data_metadata': {'row_groups': pf.metadata.num_row_groups, 'rows': pf.metadata.num_rows},
        'fold0_membership': {'fit_valid_row_groups': len(fit_valid), 'excluded_evaluation_valid_row_groups': len(eval_valid)},
        'sampled_groups': sampled,
        'sampled_sequences': len(records),
        'all_sampled_seq_ix_match_manifest': True,
        'records': records,
        'interpretation_limit': 'Exploratory sample from previously exposed fold0 validation; not a model score, population estimate, or fresh confirmation.',
    }
    summary_path = lineage / 'E535_SMALL_VALID_ROW_GROUP_STREAM_V1.json'
    summary_path.write_text(json.dumps(summary, indent=2) + '\n')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = {'H230-fit-valid': '#2878B5', 'H230-excluded-valid': '#D95F02'}
    for role, label in [('H230-fit-valid', 'H230 fit valid (n=12)'), ('H230-excluded-valid', 'H230 excluded eval (n=12)')]:
        rows = [r for r in records if r['role'] == role]
        x = np.array([r['i0_best_price_events_after_warmup99'] / 19.901 for r in rows])
        y = np.array([r['target_events_after_warmup99'] / 19.901 for r in rows])
        axes[0].scatter(x, y, color=colors[role], label=label, alpha=0.85, edgecolor='white', linewidth=0.5)
    axes[0].set_xlabel('i0 best-price events per 1,000 rows')
    axes[0].set_ylabel('label-pair changes per 1,000 rows')
    axes[0].set_title('Input event clock vs target-change clock')
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(alpha=0.2)
    axes[1].boxplot(
        [[r['median_target_run_rows_after_warmup99'] for r in records if r['role'] == role]
         for role in ['H230-fit-valid', 'H230-excluded-valid']],
        tick_labels=['H230 fit valid', 'H230 excluded eval'],
        showfliers=False,
    )
    axes[1].set_ylabel('median target plateau length (rows)')
    axes[1].set_title('Per-sequence target dwell time')
    axes[1].grid(axis='y', alpha=0.2)
    fig.suptitle('Fold0 valid sequences: deterministic 12 + 12 row-group sample, seed 535', fontsize=10)
    fig.tight_layout()
    fig.savefig(plots / 'E535_event_and_target_clocks.png', dpi=160, bbox_inches='tight')
    fig.savefig(plots / 'E535_event_and_target_clocks.svg', bbox_inches='tight')
    plt.close(fig)

    role_summary = []
    for role in sampled:
        rows = [r for r in records if r['role'] == role]
        role_summary.append({
            'role': role,
            'n': len(rows),
            'scored_rows_median': int(np.median([r['scored_rows_after_warmup99'] for r in rows])),
            'label_events_median': float(np.median([r['target_events_after_warmup99'] for r in rows])),
            'i0_events_median': float(np.median([r['i0_best_price_events_after_warmup99'] for r in rows])),
            'target_run_median_rows': float(np.median([r['median_target_run_rows_after_warmup99'] for r in rows])),
            'sampled_class_counts': {str(k): sum(r['market_class'] == k for r in rows) for k in range(4)},
        })
    mo.md('''### Bounded E535 data diagnostic

    Read 24 of the 1,873 validation row groups, six columns at a time, one complete 20,000-row sequence per read. Every sampled `seq_ix` matched the frozen fold0 manifest. This is an exposed exploratory sample, not a population estimate.

    | Role | n | median scored rows | median label changes | median i0 price events | median target run length |
    |---|---:|---:|---:|---:|---:|
    ''' + '\n'.join(f"| {r['role']} | {r['n']} | {r['scored_rows_median']} | {r['label_events_median']:.1f} | {r['i0_events_median']:.1f} | {r['target_run_median_rows']:.1f} |" for r in role_summary) + '''

    The plot records two causal data clocks available before any fit. Use it to choose a matched follow-up; do not read the sample difference as an effect estimate.''')
    return


@app.cell(hide_code=True)
def _():
    # One-shot bounded descriptive diagnostic; this cell launches no model fits.
    import runpy as _e535_runpy
    _e535_namespace = _e535_runpy.run_path("research/campaign_2026-10-06/bandit_continuation/model_data_forensics_E535/E535_FIXED468_SCAN_V1.py", run_name="__main__")
    _e535_namespace["FIGURE"]

    return


if __name__ == "__main__":
    app.run()
