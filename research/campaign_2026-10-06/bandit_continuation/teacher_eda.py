import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    import marimo as mo

    repo = Path('/home/vlad/development/wunder/wnn_connectome_starterpack')
    eda_dir = repo / 'research/campaign_2026-10-06/bandit_continuation'
    plot_dir = eda_dir / 'plots'
    plot_dir.mkdir(exist_ok=True)
    mo.md("## Teacher disagreement audit\nExposed fold 0; 64 reproducibly sampled evaluation sequences, every 40th row after warm-up. These are historical reference caches, not new student results or a public forecast. No training runs from this notebook.")
    return eda_dir, json, mo, np, plot_dir, plt, repo


@app.cell(hide_code=True)
def _(eda_dir, json, mo, np, repo):
    model_names = ['c20261003_combo_s3_f0', 'c20261003_combo_s4_f0', 'c20260929_since_s3_f0', 'c20260929_ft_since_s3_f0']
    model_labels = ['warm GRU256 s3', 'warm GRU256 s4', 'since GRU128', 'fine-tuned GRU128']
    split_data = json.loads((repo / 'research/campaign_2026-09-29/artifacts/splits/fold0.json').read_text())
    with np.load(repo / 'cache/valid_targets.npz') as _cache:
        _groups = _cache['row_group']
        _eval_groups = set(split_data['evaluation']['valid'])
        _positions = np.array([i for i, g in enumerate(_groups) if int(g) in _eval_groups])
        sample_positions = np.sort(np.random.default_rng(20261006).choice(_positions, size=min(64, len(_positions)), replace=False))
        sample_rows = np.arange(99, 20000, 40)
        sample_y = _cache['y'][sample_positions][:, sample_rows, :].copy()
        sample_mask = _cache['mask'][sample_positions][:, sample_rows].copy()
        sample_groups = _groups[sample_positions].tolist()
    sample_predictions = []
    cache_audit = []
    for _name in model_names:
        _path = repo / 'cache' / f'pred_{_name}_valid.npy'
        _pred = np.load(_path, mmap_mode='r')
        _side = json.loads(_path.with_suffix('.json').read_text())
        assert _side['row_groups'] == _groups.tolist(), f'Row-order mismatch: {_name}'
        assert _pred.shape == (len(_groups), 20000, 2)
        _values = _pred[sample_positions][:, sample_rows, :].copy()
        assert np.isfinite(_values).all()
        sample_predictions.append(_values)
        cache_audit.append({'run': _name, 'shape': list(_pred.shape), 'dtype': str(_pred.dtype), 'checkpoint_sha256': _side['checkpoint_sha256'], 'declared_prediction_sha256': _side['prediction_sha256'], 'split_manifest_sha256': _side['split_manifest_sha256'], 'epoch': _side['epoch'], 'row_alignment_checked': True, 'full_file_hash_verified_in_this_eda': False})
    sample_predictions = np.stack(sample_predictions)
    eda_manifest = {'kind': 'historical_teacher_disagreement_eda', 'exposure': 'exposed discovery', 'sample_seed': 20261006, 'sample_groups': sample_groups, 'sample_rows': sample_rows.tolist(), 'scored_rows': int(sample_mask.sum()), 'caches': cache_audit, 'new_scientific_fits': 0}
    (eda_dir / 'TEACHER_EDA.json').write_text(json.dumps(eda_manifest, indent=2) + '\n')
    mo.md(f"Audited row alignment for **{len(model_names)} caches**; sampled **{len(sample_groups)} groups** and **{sample_mask.sum():,} scored rows**. Cached file hashes are recorded from sidecars; full-file hash verification belongs to experiment admission.")
    return model_labels, sample_mask, sample_predictions, sample_y


@app.cell(hide_code=True)
def _(
    mo,
    model_labels,
    np,
    plot_dir,
    plt,
    sample_mask,
    sample_predictions,
    sample_y,
):
    plot_residuals = np.clip(sample_predictions, -2, 2) - np.clip(sample_y, -2, 2)[None, ...]
    plot_weights = np.abs(np.clip(sample_y, -2, 2)) * sample_mask[..., None]
    correlations = []
    fig_corr, axes_corr = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    for _target, _ax in enumerate(axes_corr):
        _w = plot_weights[..., _target].reshape(-1)
        _r = plot_residuals[..., _target].reshape(len(model_labels), -1)
        _centered = _r - ((_r * _w).sum(axis=1) / _w.sum())[:, None]
        _cov = (_centered * _w) @ _centered.T / _w.sum()
        _corr = _cov / np.sqrt(np.outer(np.diag(_cov), np.diag(_cov)))
        correlations.append(_corr.tolist())
        _im = _ax.imshow(_corr, vmin=0, vmax=1, cmap='viridis')
        _ax.set_xticks(range(len(model_labels)), model_labels, rotation=35, ha='right')
        _ax.set_yticks(range(len(model_labels)), model_labels)
        _ax.set_title(f'Target {_target}: weighted residual correlation')
        for _i in range(len(model_labels)):
            for _j in range(len(model_labels)):
                _ax.text(_j, _i, f'{_corr[_i, _j]:.3f}', ha='center', va='center', color='black' if _corr[_i, _j] > .65 else 'white', fontsize=9)
    fig_corr.colorbar(_im, ax=list(axes_corr), shrink=.8, label='Correlation')
    fig_corr.suptitle('Historical caches on exposed fold 0: 64 groups, sparse row sample', fontsize=11)
    fig_corr.savefig(plot_dir / 'teacher_residual_correlation.png', dpi=160)
    fig_corr.savefig(plot_dir / 'teacher_residual_correlation.svg')
    mo.vstack([mo.md('### Do these teachers make different errors?\nWeights are absolute clipped labels, using scored sampled rows. Correlation describes residual overlap; it does not establish a distillation gain.'), fig_corr])
    return (correlations,)


@app.cell(hide_code=True)
def _(
    correlations,
    eda_dir,
    json,
    mo,
    model_labels,
    np,
    plot_dir,
    plt,
    sample_mask,
    sample_predictions,
    sample_y,
):
    fig_disagreement, axes_disagreement = plt.subplots(1, 2, figsize=(10, 4.3), constrained_layout=True)
    correlation_summary = {'weighted_residual_correlations': correlations, 'model_labels': model_labels, 'disagreement_bins': []}
    for _target, _ax in enumerate(axes_disagreement):
        _select = sample_mask.reshape(-1)
        _disagreement = np.std(sample_predictions[..., _target], axis=0).reshape(-1)[_select]
        _error = np.abs(np.mean(np.clip(sample_predictions[..., _target], -2, 2), axis=0) - np.clip(sample_y[..., _target], -2, 2)).reshape(-1)[_select]
        _indices = np.argsort(_disagreement)
        _bins = np.array_split(_indices, 8)
        _x = [float(_disagreement[_b].mean()) for _b in _bins]
        _y = [float(_error[_b].mean()) for _b in _bins]
        _ax.plot(_x, _y, 'o-', color=['#147d92', '#b1572c'][_target])
        _ax.set_xlabel('Mean cross-model prediction standard deviation')
        _ax.set_ylabel('Mean absolute clipped-label error')
        _ax.set_title(f'Target {_target}: equal-count disagreement bins')
        _ax.grid(alpha=.2)
        correlation_summary['disagreement_bins'].append({'target': _target, 'mean_prediction_sd': _x, 'mean_absolute_error': _y, 'bin_counts': [len(_b) for _b in _bins]})
    fig_disagreement.suptitle('Descriptive sampled-row association; no causal or public-score claim', fontsize=11)
    fig_disagreement.savefig(plot_dir / 'teacher_disagreement_vs_error.png', dpi=160)
    fig_disagreement.savefig(plot_dir / 'teacher_disagreement_vs_error.svg')
    (eda_dir / 'TEACHER_EDA_STATISTICS.json').write_text(json.dumps(correlation_summary, indent=2) + '\n')
    mo.vstack([mo.md('### Is teacher disagreement associated with harder rows?\nEach bin has roughly the same number of scored sampled rows. Labels and errors can contain irreducible noise; this is a diagnostic for choosing later probes.'), fig_disagreement])
    return


if __name__ == "__main__":
    app.run()
