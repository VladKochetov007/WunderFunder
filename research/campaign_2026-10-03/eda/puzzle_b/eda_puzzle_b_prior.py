import marimo

__generated_with = "0.16.0"
app = marimo.App(width="full")


@app.cell
def _():
    import sys
    sys.path.insert(0, '/home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-03/eda/puzzle_b/scripts')
    import marimo as mo
    import numpy as np, pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    from common import seq, cols, nrg, OUT, TRAIN, VALID
    scan_tr = pd.read_csv(f'{OUT}/seqscan_train.csv'); scan_tr['cls'] = scan_tr.t0_scale.round(4)
    scan_va = pd.read_csv(f'{OUT}/seqscan_valid.csv'); scan_va['cls'] = scan_va.t0_scale.round(4)
    return OUT, TRAIN, VALID, mo, norm, np, pd, plt, scan_tr, scan_va, seq


@app.cell
def _(mo):
    mo.md("""
    # Puzzle B: counts, exact values, coincidences
    Every figure is also saved as PNG next to this notebook. Heavy scans live in `scripts/` and write CSV/NPZ caches
    (`seqscan_*.csv`, `hist_train.npz`, `topvals_train.csv`, `feat_corr.csv`, `ridge_probe.csv`).
    """)
    return


@app.cell
def _(OUT, norm, np, plt, seq):
    # Fingerprint 1: inputs sit on the grid Phi^-1(k/198) (sklearn QuantileTransformer, n_quantiles=100, normal output)
    _d = seq(0)
    _fig, _ax = plt.subplots(1, 3, figsize=(16, 4))
    for _a, _cs, _ttl in [(_ax[0], [f'i0_p{k}' for k in range(22)], 'i0 prices'),
                         (_ax[1], [f'i0_v{k}' for k in range(22)], 'i0 volumes'),
                         (_ax[2], ['a7', 'i0_dv0', 'i1_dv0', 'i0_dp0'], 'a7 + dv + dp')]:
        _k = norm.cdf(_d[_cs].values.astype(np.float64).ravel()) * 198
        _a.hist(_k - np.round(_k), bins=200, color='C0')
        _a.set_title(f'{_ttl}: frac part of Phi(x)*198'); _a.set_yscale('log')
    plt.tight_layout(); plt.savefig(f'{OUT}/fig02_quantile_grid.png', dpi=70)
    _fig
    return


@app.cell
def _(OUT, np, plt, scan_tr, scan_va):
    # Fingerprint 2: targets = integer ticks x one of four class scales
    _fig, _ax = plt.subplots(1, 3, figsize=(16, 4))
    _ct = scan_tr.cls.value_counts().sort_index(); _cv = scan_va.cls.value_counts().sort_index()
    _x = np.arange(len(_ct))
    _ax[0].bar(_x - .2, _ct / _ct.sum(), .4, label='train'); _ax[0].bar(_x + .2, _cv / _cv.sum(), .4, label='valid')
    _ax[0].set_xticks(_x, [str(c) for c in _ct.index]); _ax[0].legend(); _ax[0].set_title('target scale class share (sequences)')
    for _c in sorted(scan_tr.cls.unique()):
        _s = scan_tr[scan_tr.cls == _c]
        _ax[1].hist(_s.t0_kstd, bins=np.linspace(0, 20, 60), alpha=.5, label=str(_c))
        _ax[2].hist(_s.t0_kmean, bins=np.linspace(-3, 3, 60), alpha=.5, label=str(_c))
    _ax[1].set_title('per-seq std of t0 in ticks'); _ax[1].legend()
    _ax[2].set_title('per-seq mean of t0 in ticks (negative bias)'); _ax[2].legend()
    plt.tight_layout(); plt.savefig(f'{OUT}/fig03_target_ticks.png', dpi=70)
    _fig
    return


@app.cell
def _(OUT, np, pd, plt, seq):
    # Clocks: which input blocks change on the rows where a target changes
    def _chg(x):
        return np.r_[False, (x[1:] != x[:-1]).any(1)] if x.ndim == 2 else np.r_[False, x[1:] != x[:-1]]
    _rows = []
    for _rg in [0, 1, 2, 3, 500, 5000, 10000]:
        _d = seq(_rg)
        _g = {}
        for _i in ['i0', 'i1']:
            _g[_i + '_pA'] = _chg(_d[[f'{_i}_p{k}' for k in range(11)]].values)
            _g[_i + '_pB'] = _chg(_d[[f'{_i}_p{k}' for k in range(11, 22)]].values)
            _g[_i + '_v'] = _chg(_d[[f'{_i}_v{k}' for k in range(22)]].values)
            _g[_i + '_dp'] = _chg(_d[[f'{_i}_dp{k}' for k in range(4)]].values)
            _g[_i + '_dv'] = _chg(_d[[f'{_i}_dv{k}' for k in range(4)]].values)
        for _k in range(8):
            _g[f'a{_k}'] = _chg(_d[f'a{_k}'].values)
        _E = pd.DataFrame(_g); _tc = _chg(_d.t0.values) | _chg(_d.t1.values)
        _rows.append(pd.concat([_E.mean().rename('base'), _E[_tc].mean().rename('given_tchg')], axis=1).assign(rg=_rg))
    clock = pd.concat(_rows).reset_index().rename(columns={'index': 'block'})
    _p = clock.groupby('block')[['base', 'given_tchg']].mean()
    _fig, _ax = plt.subplots(figsize=(14, 4))
    _x = np.arange(len(_p)); _ax.bar(_x - .2, _p.base, .4, label='P(block changes)'); _ax.bar(_x + .2, _p.given_tchg, .4, label='P(block changes | target changes)')
    _ax.set_xticks(_x, _p.index, rotation=60); _ax.legend(); _ax.set_title('targets change only on i0 price/dp events')
    plt.tight_layout(); plt.savefig(f'{OUT}/fig04_clocks.png', dpi=70)
    _fig
    return


@app.cell
def _(OUT, np, plt, seq):
    # Coincidence: a period-6 flicker in i0 dp, and t1 follows it row by row (seq rg=10000, class 0.6569)
    _d = seq(10000)
    _s = slice(255, 330); _x = np.arange(255, 330)
    _fig, _ax = plt.subplots(3, 1, figsize=(14, 7), sharex=True)
    for _k in range(4):
        _ax[0].step(_x, _d[f'i0_dp{_k}'].values[_s], where='post', label=f'i0_dp{_k}')
    _ax[0].legend(ncol=4); _ax[0].set_title('i0 dp: rest state + 3-step excursions with period 6')
    for _k in [0, 3, 16, 19]:
        _ax[1].step(_x, _d[f'i0_p{_k}'].values[_s], where='post', label=f'i0_p{_k}')
    _ax[1].legend(ncol=4)
    _ax[2].step(_x, _d.t0.values[_s] / 0.656868, where='post', label='t0 ticks'); _ax[2].step(_x, _d.t1.values[_s] / 0.656868, where='post', label='t1 ticks')
    _ax[2].legend(); _ax[2].set_title('targets flip with the current state: target = future - current')
    plt.tight_layout(); plt.savefig(f'{OUT}/fig05_period6_flicker.png', dpi=70)
    _fig
    return


@app.cell
def _(OUT, VALID, np, plt, seq):
    # is_scored: clustered rows, ~9% of each valid sequence
    _fig, _ax = plt.subplots(2, 1, figsize=(14, 6))
    for _j, _rg in enumerate([0, 100, 1500]):
        _d = seq(_rg, ['is_scored', 'a3', 't0'], path=VALID)
        _ax[0].plot(_d.is_scored.values * 1.0 + 1.2 * _j, lw=.5, label=f'valid rg {_rg}')
    _ax[0].set_title('is_scored over the sequence (offset per seq)'); _ax[0].legend()
    _d = seq(100, ['is_scored', 'a3', 'a2'], path=VALID)
    _ax[1].plot(_d.a3.values, lw=.5, label='a3'); _ax[1].fill_between(np.arange(20000), -3, 3, where=_d.is_scored.values, color='C3', alpha=.2, label='scored')
    _ax[1].legend(); _ax[1].set_title('valid rg 100: a3 with scored rows shaded (AUC of a3 for is_scored ~0.69)')
    plt.tight_layout(); plt.savefig(f'{OUT}/fig06_is_scored.png', dpi=70)
    _fig
    return


@app.cell
def _(OUT, pd, plt):
    # Univariate score-like correlation of each input with targets, per class
    _R = pd.read_csv(f'{OUT}/feat_corr.csv')
    _p = _R[_R.kind == 'lvl'].pivot_table(index='col', columns=['target', 'cls'], values='wcorr')
    _order = [c for c in _p.index if c.startswith('i0_p')] + [c for c in _p.index if c.startswith('i0_dp')] + [c for c in _p.index if c.startswith('i1_p')] + [c for c in _p.index if c.startswith('i1_dp')] + [c for c in _p.index if c.startswith('a')]
    _p = _p.loc[[c for c in _order if c in _p.index]]
    _fig, _ax = plt.subplots(figsize=(8, 18))
    _im = _ax.imshow(_p.values, cmap='RdBu_r', vmin=-.5, vmax=.5, aspect='auto')
    _ax.set_yticks(range(len(_p)), _p.index, fontsize=7); _ax.set_xticks(range(_p.shape[1]), [f'{a}\n{b}' for a, b in _p.columns], fontsize=7)
    plt.colorbar(_im); _ax.set_title('weighted corr(level, target) by class')
    plt.tight_layout(); plt.savefig(f'{OUT}/fig07_feat_corr.png', dpi=70)
    _fig
    return


if __name__ == "__main__":
    app.run()
