import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")

with app.setup(hide_code=True):

    import json, sys
    from pathlib import Path
    import numpy as np
    import marimo as mo
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ROOT = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
    sys.path.insert(0, str(ROOT / "src"))
    from wnn.splits import load_campaign_split
    PLOTS = ROOT / "research/campaign_2026-10-01/plots/eda"
    PLOTS.mkdir(parents=True, exist_ok=True)
    CLASS_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#e34948"]
    plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
                         "axes.spines.right": False, "font.size": 9})
    def save(fig, name):
        """Persist every figure as PNG in the research folder (notebook state is not the record)."""
        fig.savefig(PLOTS / f"{name}.png", bbox_inches="tight")
        return fig



@app.cell(hide_code=True)
def data():

    _z = np.load(ROOT / "cache/valid_targets.npz")
    Y = _z["y"]; MASK = _z["mask"].copy(); MASK[:, :99] = False
    CLS = _z["market_class"].astype(int); ROWG = _z["row_group"].astype(int)
    _pos = {int(g): i for i, g in enumerate(ROWG)}
    FOLD = np.full(len(ROWG), -1); COMP = np.full(len(ROWG), -1)
    for _f in range(4):
        _sp = load_campaign_split(ROOT / f"research/campaign_2026-09-29/artifacts/splits/fold{_f}.json", ROOT)
        _cm = {g: c for s, g, c in _sp.components if s == "valid"}
        for _g in _sp.evaluation_groups:
            FOLD[_pos[_g]] = _f; COMP[_pos[_g]] = _cm[_g]
    def _load(name): return np.load(ROOT / f"cache/pred_{name}_valid.npy", mmap_mode="r")
    SOTA = np.zeros(Y.shape, np.float32); REF = np.zeros(Y.shape, np.float32)
    for _f in range(4):
        _rows = np.flatnonzero(FOLD == _f)
        _ft4 = _load(f"c20260929_ft_since_s4_f{_f}"); _ft3 = _load(f"c20260929_ft_since_s3_f{_f}")
        _p4 = _load(f"c20260929_since_s4_f{_f}"); _h4 = _load(f"c20260929_ft_since_s4_f{_f}h")
        _dp = _load(f"c20261001_d256_s3_f{_f}"); _dh = _load(f"c20261001_ft_d256_s3_f{_f}h")
        for _i in _rows:
            _d = 0.5 * (np.asarray(_dp[_i], np.float32) + np.asarray(_dh[_i], np.float32))
            if CLS[_i] == 3:
                _s = 0.5 * (np.asarray(_p4[_i], np.float32) + np.asarray(_h4[_i], np.float32))
            else:
                _s = np.asarray(_ft4[_i], np.float32)
            SOTA[_i] = 0.5 * (_s + _d)
            REF[_i] = 0.5 * (np.asarray(_ft3[_i], np.float32) + np.asarray(_ft4[_i], np.float32))
    import pyarrow.parquet as _pq
    _pf = _pq.ParquetFile(ROOT / "datasets/valid.parquet")
    A6 = _pf.read(columns=["a6"]).column(0).to_numpy().reshape(-1, 20000).astype(np.float32)
    A6_END = float(A6.min())
    EVENT = np.isclose(A6, A6_END, atol=1e-6)   # quote-move marker: targets change exactly on these rows
    print(Y.shape, SOTA.dtype, "a6 endpoint", A6_END, "event share", EVENT.mean().round(4), "folds", np.bincount(FOLD + 1))

    return CLS, EVENT, FOLD, MASK, REF, SOTA, Y


@app.cell(hide_code=True)
def metric(MASK, REF, SOTA, Y):
    def wsums(y, p, m):
        """|y|-weighted raw sums per target over selected rows (clip ±2): [w, wy, wp, wyy, wpp, wyp]."""
        yy = np.clip(y[m].astype(np.float64), -2, 2); pp = np.clip(p[m].astype(np.float64), -2, 2); w = np.abs(yy)
        return np.stack([w.sum(0), (w*yy).sum(0), (w*pp).sum(0), (w*yy*yy).sum(0), (w*pp*pp).sum(0), (w*yy*pp).sum(0)], -1)
    def wp_from(s):
        n, sy, sp, syy, spp, syp = np.moveaxis(np.asarray(s), -1, 0)
        r = (syp/n - sy*sp/n**2) / np.sqrt((syy/n - (sy/n)**2) * (spp/n - (sp/n)**2))
        return float(np.mean(r))
    SEQ_SOTA = np.stack([wsums(Y[i], SOTA[i], MASK[i]) for i in range(len(Y))])
    SEQ_REF = np.stack([wsums(Y[i], REF[i], MASK[i]) for i in range(len(Y))])

    return SEQ_REF, SEQ_SOTA, wp_from, wsums


@app.cell(hide_code=True)
def summary(CLS, SEQ_REF, SEQ_SOTA, Y, wp_from):
    _rows = []
    for _c in [0, 1, 2, 3, "all"]:
        _sel = np.ones(len(Y), bool) if _c == "all" else CLS == _c
        _w = SEQ_SOTA[_sel][..., 0].sum() / SEQ_SOTA[..., 0].sum()
        _rows.append({"class": _c, "sequences": int(_sel.sum()), "metric weight share": round(float(_w), 3),
                      "WP reference (incumbent recipe)": round(wp_from(SEQ_REF[_sel].sum(0)), 5),
                      "WP SOTA (R4D analog)": round(wp_from(SEQ_SOTA[_sel].sum(0)), 5),
                      "delta": round(wp_from(SEQ_SOTA[_sel].sum(0)) - wp_from(SEQ_REF[_sel].sum(0)), 5)})
    SUMMARY = _rows
    mo.vstack([mo.md("## Summary — clean out-of-fold, 4 folds, Global WP (pooled, |y|-weighted, clip ±2)"), mo.ui.table(_rows, selection=None)])

    return (SUMMARY,)


@app.cell(hide_code=True)
def target_dist(CLS, MASK, Y):
    _fig, _ax = plt.subplots(1, 3, figsize=(13, 3.4))
    for _c in range(4):
        _sel = CLS == _c
        _y = np.clip(Y[_sel][MASK[_sel]], -2, 2)
        _ax[0].hist(_y[:, 0], bins=81, range=(-2, 2), histtype="step", density=True, color=CLASS_COLORS[_c], label=f"class {_c}")
        _ax[1].hist(_y[:, 1], bins=81, range=(-2, 2), histtype="step", density=True, color=CLASS_COLORS[_c])
    _ax[0].set_title("t0 on scored rows (clipped)"); _ax[1].set_title("t1 on scored rows (clipped)"); _ax[0].legend(fontsize=8)
    _y3 = np.clip(Y[CLS == 3][MASK[CLS == 3]], -2, 2)
    _ax[2].hexbin(_y3[:, 0], _y3[:, 1], gridsize=40, bins="log", cmap="Blues"); _ax[2].set_title("class 3: t0 vs t1 (log count)")
    _ax[2].set_xlabel("t0"); _ax[2].set_ylabel("t1")
    save(_fig, "01_target_distributions")

    return


@app.cell(hide_code=True)
def run_structure(CLS, EVENT, Y):
    def run_bounds(i):
        """Event rows (quote-move markers) from row 99 on; runs are [e_k, e_{k+1})."""
        return np.flatnonzero(EVENT[i, 99:]) + 99
    _fig, _ax = plt.subplots(1, 2, figsize=(12, 3.4))
    for _c in range(4):
        _seqs = np.flatnonzero(CLS == _c)[:150]
        _rl = np.concatenate([np.diff(run_bounds(i)) for i in _seqs])
        _ax[0].hist(np.log10(_rl), bins=60, histtype="step", density=True, color=CLASS_COLORS[_c], label=f"class {_c}: median {int(np.median(_rl))} rows")
    _ax[0].set_title("run length between quote-move events (log10 rows)"); _ax[0].legend(fontsize=8)
    _chg = (Y[:, 1:] != Y[:, :-1]).any(-1)
    _on = (_chg & EVENT[:, 1:]).sum() / max(_chg.sum(), 1)
    _ax[1].bar(range(4), [EVENT[CLS == c].mean() for c in range(4)], color=CLASS_COLORS); _ax[1].set_xticks(range(4))
    _ax[1].set_xlabel("class"); _ax[1].set_title(f"event-row share (target changes on event rows: {_on:.4f})")
    save(_fig, "02_run_structure")

    return


@app.cell(hide_code=True)
def event_series(CLS, EVENT, SEQ_SOTA, SOTA, Y):
    # Run-level (event-time) series per sequence. Run k = rows [e_k, e_{k+1}); the target is constant on a run.
    # D = (t0 - t1)/2 is the future direction (~80% of metric variance), S = t0 + t1 the spread-like level.
    # Residual uses a per-class |y|-weighted OLS scale so it is orthogonal to the prediction in the metric's geometry.
    def _beta(cls):
        _s = SEQ_SOTA[CLS == cls].sum(0)
        n, sy, sp, syy, spp, syp = np.moveaxis(_s, -1, 0)
        return (syp/n - sy*sp/n**2) / (spp/n - (sp/n)**2)
    BETA = np.stack([_beta(c) for c in range(4)])      # (class, target)
    RUNS = []
    for _i in range(len(Y)):
        _e = np.flatnonzero(EVENT[_i, 99:]) + 99
        if len(_e) < 30: continue
        _y = np.clip(Y[_i, _e].astype(np.float64), -2, 2)
        _pend = np.clip(SOTA[_i, np.r_[_e[1:] - 1, 19999]].astype(np.float64), -2, 2)
        _pstart = np.clip(SOTA[_i, _e].astype(np.float64), -2, 2)
        _b = BETA[CLS[_i]]
        RUNS.append({"i": _i, "cls": int(CLS[_i]), "len": np.diff(np.r_[_e, 20000]),
                     "D": (_y[:, 0] - _y[:, 1]) / 2, "S": _y.sum(1),
                     "rD_start": (_y[:, 0] - _b[0]*_pstart[:, 0] - (_y[:, 1] - _b[1]*_pstart[:, 1])) / 2,
                     "rD_end": (_y[:, 0] - _b[0]*_pend[:, 0] - (_y[:, 1] - _b[1]*_pend[:, 1])) / 2})
    mo.md(f"Event-time series built for **{len(RUNS)}** sequences; per-class residual scale β (t0, t1): {np.round(BETA, 3).tolist()}")

    return BETA, RUNS


@app.cell(hide_code=True)
def target_acf(RUNS):
    from statsmodels.tsa.stattools import acf as sm_acf, pacf as sm_pacf
    def mean_acf(key, cls, nlags=20, fn="acf", max_seqs=250):
        """Length-weighted average of per-sequence ACF/PACF of an event-time series (sequence boundaries respected)."""
        _acc, _w = np.zeros(nlags + 1), 0.0
        for _r in [r for r in RUNS if r["cls"] == cls][:max_seqs]:
            _x = _r[key]
            if len(_x) < 4 * nlags or np.std(_x) < 1e-9: continue
            _v = sm_acf(_x, nlags=nlags, fft=True) if fn == "acf" else sm_pacf(_x, nlags=nlags, method="ywm")
            _acc += len(_x) * _v; _w += len(_x)
        return _acc / max(_w, 1)
    LAGS = np.arange(21)
    _fig, _ax = plt.subplots(2, 2, figsize=(13, 6.4), sharex=True)
    TARGET_ACF = {}
    for _c in range(4):
        for _j, _key in enumerate(["D", "S"]):
            _a = mean_acf(_key, _c); _p = mean_acf(_key, _c, fn="pacf")
            TARGET_ACF[(_c, _key)] = (_a, _p)
            _ax[0, _j].plot(LAGS[1:], _a[1:], "o-", ms=3, color=CLASS_COLORS[_c], label=f"class {_c}")
            _ax[1, _j].plot(LAGS[1:], _p[1:], "o-", ms=3, color=CLASS_COLORS[_c])
    for _j, _key in enumerate(["D (direction)", "S (t0+t1, spread-like)"]):
        _ax[0, _j].set_title(f"target ACF in event time — {_key}"); _ax[1, _j].set_title(f"target PACF in event time — {_key}")
        _ax[1, _j].set_xlabel("lag (quote-move runs)")
        for _r in range(2): _ax[_r, _j].axhline(0, color="k", lw=0.6)
    _ax[0, 0].legend(fontsize=8)
    save(_fig, "03_target_acf_pacf_event_time")

    return LAGS, mean_acf, sm_acf


@app.cell(hide_code=True)
def residual_acf(BETA, CLS, LAGS, SOTA, Y, mean_acf):
    _fig, _ax = plt.subplots(2, 3, figsize=(15, 6.4))
    RESID_ACF = {}
    for _c in range(4):
        for _j, _key in enumerate(["rD_start", "rD_end"]):
            _a = mean_acf(_key, _c); _p = mean_acf(_key, _c, fn="pacf")
            RESID_ACF[(_c, _key)] = (_a, _p)
            _ax[0, _j].plot(LAGS[1:], _a[1:], "o-", ms=3, color=CLASS_COLORS[_c], label=f"class {_c}")
            _ax[1, _j].plot(LAGS[1:], _p[1:], "o-", ms=3, color=CLASS_COLORS[_c])
    _ax[0, 0].set_title("residual ACF, p at run START (event time)"); _ax[1, 0].set_title("residual PACF, p at run START")
    _ax[0, 1].set_title("residual ACF, p at run END (event time)"); _ax[1, 1].set_title("residual PACF, p at run END")
    # row-time ACF of the scored-row residual (log-spaced lags), class means over sequences
    _lags = np.unique(np.logspace(0, 3.3, 28).astype(int))
    for _c in range(4):
        _acc = np.zeros(len(_lags)); _n = 0
        for _i in np.flatnonzero(CLS == _c)[:200]:
            _p = SOTA[_i, 99:].astype(np.float64); _y = np.clip(Y[_i, 99:].astype(np.float64), -2, 2)
            _r = ((_y[:, 0] - BETA[_c, 0]*_p[:, 0]) - (_y[:, 1] - BETA[_c, 1]*_p[:, 1])) / 2
            _r = _r - _r.mean(); _v = (_r * _r).mean()
            if _v < 1e-12: continue
            _acc += np.array([(_r[:-L] * _r[L:]).mean() / _v for L in _lags]); _n += 1
        _ax[0, 2].semilogx(_lags, _acc / max(_n, 1), "-", color=CLASS_COLORS[_c], label=f"class {_c}")
        _ys = np.clip(Y[np.flatnonzero(CLS == _c)[:200], 99:], -2, 2).astype(np.float64)
        _d = (_ys[..., 0] - _ys[..., 1]) / 2; _d = _d - _d.mean(1, keepdims=True); _vv = (_d*_d).mean(1, keepdims=True) + 1e-12
        _ax[1, 2].semilogx(_lags, [((_d[:, :-L]*_d[:, L:]).mean(1, keepdims=True) / _vv).mean() for L in _lags], "-", color=CLASS_COLORS[_c])
    _ax[0, 2].set_title("row-time ACF of residual D (all rows ≥ 99)"); _ax[1, 2].set_title("row-time ACF of target D")
    for _a in _ax.ravel(): _a.axhline(0, color="k", lw=0.6)
    _ax[0, 0].legend(fontsize=8); _ax[1, 2].set_xlabel("lag (rows)"); _ax[1, 0].set_xlabel("lag (runs)"); _ax[1, 1].set_xlabel("lag (runs)")
    save(_fig, "04_residual_acf_pacf")

    return


@app.cell(hide_code=True)
def arch_garch(LAGS, RUNS, sm_acf):
    # Volatility clustering in event time: ACF of squared (demeaned) series = ARCH effect; Engle ARCH-LM per sequence;
    # GARCH(1,1) fits on run-level D and on the residual for a sample of sequences per class.
    from statsmodels.stats.diagnostic import het_arch as _het_arch
    from arch import arch_model as _arch_model
    import warnings as _warnings
    _fig, _ax = plt.subplots(1, 3, figsize=(15, 3.6))
    GARCH = {}
    for _c in range(4):
        _sq = []
        for _key, _ls in (("D", "-"), ("rD_start", "--")):
            _acc, _w = np.zeros(21), 0
            for _r in [r for r in RUNS if r["cls"] == _c][:250]:
                _x = _r[_key] - _r[_key].mean(); _x2 = _x * _x - (_x * _x).mean()
                if _x2.std() < 1e-9 or len(_x2) < 100: continue
                _acc += len(_x2) * sm_acf(_x2, nlags=20, fft=True); _w += len(_x2)
            _ax[0].plot(LAGS[1:], (_acc / max(_w, 1))[1:], _ls, color=CLASS_COLORS[_c], label=f"class {_c} {'target D' if _key=='D' else 'residual'}")
        _pv, _ab = [], []
        for _r in [r for r in RUNS if r["cls"] == _c][:60]:
            _x = _r["rD_start"]
            if len(_x) < 300 or _x.std() < 1e-6: continue
            _pv.append(_het_arch(_x - _x.mean(), nlags=10)[1])
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore")
                try:
                    _res = _arch_model(100 * (_x - _x.mean()), mean="Zero", vol="GARCH", p=1, q=1).fit(disp="off")
                    _ab.append((_res.params["alpha[1]"], _res.params["beta[1]"]))
                except Exception:
                    pass
        _ab = np.array(_ab)
        GARCH[_c] = {"arch_lm_share_p<0.01": float(np.mean(np.array(_pv) < 0.01)), "median_alpha": float(np.median(_ab[:, 0])),
                     "median_beta": float(np.median(_ab[:, 1])), "median_persistence": float(np.median(_ab.sum(1))), "n": len(_ab)}
        _ax[1].hist(_ab.sum(1), bins=20, range=(0, 1.05), histtype="step", color=CLASS_COLORS[_c], label=f"class {_c}")
        _ax[2].bar(_c, GARCH[_c]["arch_lm_share_p<0.01"], color=CLASS_COLORS[_c])
    _ax[0].set_title("ACF of squared series (ARCH effect), event time"); _ax[0].legend(fontsize=7); _ax[0].set_xlabel("lag (runs)")
    _ax[1].set_title("GARCH(1,1) persistence α+β of residual (per sequence)"); _ax[1].legend(fontsize=8)
    _ax[2].set_title("share of sequences with ARCH-LM p<0.01 (residual)"); _ax[2].set_xticks(range(4))
    save(_fig, "05_arch_garch")
    mo.ui.table([{"class": c, **v} for c, v in GARCH.items()], selection=None)

    return (GARCH,)


@app.cell(hide_code=True)
def calibration(CLS, MASK, SOTA, Y):
    # Reliability: |y|-weighted mean target vs prediction bins, per class and target (scored rows, OOF).
    _fig, _ax = plt.subplots(1, 2, figsize=(12, 3.8))
    for _t in range(2):
        for _c in range(4):
            _sel = CLS == _c
            _p = np.clip(SOTA[_sel][MASK[_sel]][:, _t], -2, 2); _y = np.clip(Y[_sel][MASK[_sel]][:, _t], -2, 2); _w = np.abs(_y)
            _q = np.quantile(_p, np.linspace(0, 1, 26)); _b = np.clip(np.searchsorted(_q, _p) - 1, 0, 24)
            _pm = np.bincount(_b, _w * _p, 25) / np.maximum(np.bincount(_b, _w, 25), 1e-9)
            _ym = np.bincount(_b, _w * _y, 25) / np.maximum(np.bincount(_b, _w, 25), 1e-9)
            _ax[_t].plot(_pm, _ym, "o-", ms=3, color=CLASS_COLORS[_c], label=f"class {_c}")
        _ax[_t].plot([-2, 2], [-2, 2], "k:", lw=0.8); _ax[_t].set_title(f"t{_t}: |y|-weighted mean target per prediction bin")
        _ax[_t].set_xlabel("prediction (bin mean)"); _ax[_t].set_ylabel("target (bin mean)")
    _ax[0].legend(fontsize=8)
    save(_fig, "06_calibration")

    return


@app.cell(hide_code=True)
def event_position(CLS, EVENT, MASK, SOTA, Y, wp_from, wsums):
    # Where inside a run is the metric lost? WP by rows since the last quote move (causal) and rows until the next one (not causal).
    _bins = np.array([0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 20001])
    _since = np.zeros(Y.shape[:2], np.int32); _until = np.zeros(Y.shape[:2], np.int32)
    for _i in range(len(Y)):
        _e = np.flatnonzero(EVENT[_i]); _idx = np.arange(20000)
        _k = np.searchsorted(_e, _idx, side="right") - 1
        _since[_i] = np.where(_k >= 0, _idx - _e[np.maximum(_k, 0)], 9999)
        _k2 = np.searchsorted(_e, _idx, side="right")
        _until[_i] = np.where(_k2 < len(_e), _e[np.minimum(_k2, len(_e) - 1)] - _idx, 9999)
    _fig, _ax = plt.subplots(1, 3, figsize=(15, 3.6))
    for _j, (_arr, _lab) in enumerate(((_since, "rows since last quote move (causal)"), (_until, "rows until next quote move (non-causal)"))):
        for _c in range(4):
            _sel = CLS == _c
            _bb = np.digitize(_arr[_sel], _bins) - 1
            _vals, _wts = [], []
            for _k in range(len(_bins) - 1):
                _m = MASK[_sel] & (_bb == _k)
                if _m.sum() < 500: _vals.append(np.nan); _wts.append(0); continue
                _vals.append(wp_from(wsums(Y[_sel], SOTA[_sel], _m))); _wts.append(np.abs(np.clip(Y[_sel][_m], -2, 2)).sum())
            _ax[_j].plot(range(len(_vals)), _vals, "o-", ms=3, color=CLASS_COLORS[_c], label=f"class {_c}")
            if _j == 1:
                _ax[2].plot(range(len(_wts)), np.array(_wts) / max(sum(_wts), 1), "o-", ms=3, color=CLASS_COLORS[_c])
        _ax[_j].set_xticks(range(len(_bins) - 1)); _ax[_j].set_xticklabels([f"{a}" for a in _bins[:-1]], rotation=45, fontsize=7)
        _ax[_j].set_title(f"WP by {_lab}"); _ax[_j].set_xlabel("bin lower edge (rows)")
    _ax[2].set_xticks(range(len(_bins) - 1)); _ax[2].set_xticklabels([f"{a}" for a in _bins[:-1]], rotation=45, fontsize=7)
    _ax[2].set_title("metric weight share by rows until next move"); _ax[0].legend(fontsize=8)
    save(_fig, "07_wp_by_event_position")

    return


@app.cell(hide_code=True)
def per_sequence(CLS, FOLD, SEQ_REF, SEQ_SOTA, wp_from):
    # Per-sequence picture: within-sequence WP distribution, SOTA vs reference gain per sequence, worst sequences.
    _seq_wp = np.array([wp_from(s) if s[..., 0].min() > 0 else np.nan for s in SEQ_SOTA])
    _seq_ref = np.array([wp_from(s) if s[..., 0].min() > 0 else np.nan for s in SEQ_REF])
    _fig, _ax = plt.subplots(1, 3, figsize=(15, 3.6))
    for _c in range(4):
        _v = _seq_wp[(CLS == _c) & np.isfinite(_seq_wp)]
        _ax[0].hist(_v, bins=40, range=(-0.5, 1), histtype="step", density=True, color=CLASS_COLORS[_c], label=f"class {_c} median {np.median(_v):.2f}")
        _d = (_seq_wp - _seq_ref)[(CLS == _c) & np.isfinite(_seq_wp)]
        _ax[1].hist(_d, bins=40, range=(-0.05, 0.05), histtype="step", color=CLASS_COLORS[_c])
    _ax[0].set_title("within-sequence WP (SOTA OOF)"); _ax[0].legend(fontsize=7)
    _ax[1].set_title("per-sequence WP gain: SOTA − reference"); _ax[1].axvline(0, color="k", lw=0.6)
    _w = SEQ_SOTA[..., 0, 0]
    _ax[2].scatter(np.log10(_w + 1), _seq_wp, s=4, c=[CLASS_COLORS[c] for c in CLS], alpha=0.5)
    _ax[2].set_title("sequence WP vs log10 metric weight"); _ax[2].set_xlabel("log10(Σ|y| on scored rows)")
    save(_fig, "08_per_sequence")
    _worst = np.argsort(np.nan_to_num(_seq_wp, nan=9))[:12]
    mo.ui.table([{"seq_row": int(i), "class": int(CLS[i]), "fold": int(FOLD[i]), "WP": round(float(_seq_wp[i]), 3),
                  "ref WP": round(float(_seq_ref[i]), 3), "weight": round(float(_w[i]), 1)} for i in _worst], selection=None)

    return


@app.cell(hide_code=True)
def between_within(CLS, SEQ_SOTA):
    # Pooled covariance decomposition per class: between-sequence (sequence means) vs within-sequence parts.
    _rows = []
    for _c in range(4):
        for _t in range(2):
            _s = SEQ_SOTA[CLS == _c][:, _t]                       # per-sequence raw sums [w, wy, wp, wyy, wpp, wyp]
            _n = _s[:, 0]; _my = _s[:, 1] / np.maximum(_n, 1e-9); _mp = _s[:, 2] / np.maximum(_n, 1e-9)
            _N = _n.sum(); _My = _s[:, 1].sum() / _N; _Mp = _s[:, 2].sum() / _N
            _cov_b = (_n * (_my - _My) * (_mp - _Mp)).sum() / _N
            _cov_w = (_s[:, 5] - _n * _my * _mp).sum() / _N
            _vy_b = (_n * (_my - _My) ** 2).sum() / _N; _vp_b = (_n * (_mp - _Mp) ** 2).sum() / _N
            _vy = _s[:, 3].sum() / _N - _My ** 2; _vp = _s[:, 4].sum() / _N - _Mp ** 2
            _rows.append({"class": _c, "target": f"t{_t}", "cov share between": round(_cov_b / (_cov_b + _cov_w), 3),
                          "y var share between": round(_vy_b / _vy, 3), "p var share between": round(_vp_b / _vp, 3),
                          "corr between": round(_cov_b / np.sqrt(_vy_b * _vp_b), 3),
                          "corr within": round(_cov_w / np.sqrt((_vy - _vy_b) * (_vp - _vp_b)), 3)})
    BETWEEN_WITHIN = _rows
    mo.ui.table(_rows, selection=None)

    return


@app.cell(hide_code=True)
def findings(GARCH, SUMMARY):
    _g = {c: round(v["median_persistence"], 3) for c, v in GARCH.items()}
    mo.md(f"""
    ## Findings (clean OOF, 4 folds; SOTA = R4D analog, reference = incumbent-recipe pair)

    **Where the metric lives.** Class 3 carries ~48% of the metric weight at WP ≈ {SUMMARY[3]['WP SOTA (R4D analog)']:.3f}; classes 0–2 sit at 0.74–0.81.
    SOTA gains are largest in class 3 (+{SUMMARY[3]['delta']:.4f}), then classes 2/1/0. The public test is estimated to hold ~46% class-3 sequences, so class 3 dominates public score.

    **Target dynamics (event time).** Classes 0–2: target direction D is a persistent AR-like process (ACF 0.65–0.84 at lag 1, PACF cuts after ~2 runs) — consecutive runs share most of their future path.
    Class 3 is a different regime: S alternates sign every run (ACF ≈ −0.9 at lag 1) and D oscillates with period ≈ 4–6 runs — quote **flicker cycles** between two levels.

    **Residuals.** In event time the SOTA residual stays autocorrelated for 10+ runs (lag-1 0.46–0.75); class 3 keeps periodic peaks at lags 4/8/12/16/20.
    This is overlapping-horizon structure: a run's label depends on many later quote moves (maturity ≥ 30 events, E-317), so these correlations are **not causally exploitable** (feature-reconstructed label feedback ≤ 0).
    In row time the residual ACF decays to ~0 by 1,000 rows.

    **Volatility (ARCH/GARCH).** ARCH-LM rejects homoscedasticity in 100% of sampled sequences. Median GARCH(1,1) persistence α+β of the event-time residual: {_g}.
    Class 3 is near-integrated (β ≈ 0.82): residual magnitude regimes last long. Earlier per-sequence scale/level oracles turned out to be overfit when cross-fitted (E-314), so a causal scale model must come from observable activity, not past residuals.

    **Calibration.** |y|-weighted reliability is close to the diagonal for every class and target (slight shrink at the extremes in class 2), so recalibration has no headroom (consistent with E-288).

    **Event position.** WP falls steeply with rows until the next quote move (non-causal): ~0.88 within 8 rows of a move, ≤0.3 beyond 64 rows. In class 3 most metric weight sits 16–256 rows before the next move, where WP is 0.1–0.5 — the long queue-race runs.
    WP by rows since the last move (causal) is comparatively flat in class 3, so the model does not know *when* it is in a long run.

    **Between vs within sequences.** >90% of the pooled covariance is within sequences. Class 3 has both weak between-sequence correlation (≈0.52) and weak within (≈0.43); y's between-sequence variance share (≈0.11–0.12) is twice the prediction's (≈0.06) — the model under-disperses sequence-level means in class 3.

    ### Where to dig (creative-research readout)
    1. **Distillation strength** is the only lever with a large recent effect (α 0.7 vs 0.5: +0.0023 single model, fold 0) — denoising class-3 labels whose realized values are dominated by long overlapping futures. Confirm and push α / teacher quality.
    2. **Class-3 flicker regime as a representation**: explicit cycle phase/period features (event-time position within the flicker cycle) may let the GRU separate flicker runs from genuine queue races; cheap to test as inputs.
    3. **Under-dispersed class-3 sequence level**: a causal, activity-based class-3 level/scale state (not past residuals) — test only with cross-fitting; prior oracle results warn of overfit.
    """)

    return


if __name__ == "__main__":
    app.run()
