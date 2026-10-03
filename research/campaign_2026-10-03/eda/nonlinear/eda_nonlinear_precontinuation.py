import marimo

__generated_with = "0.25.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import sys
    from pathlib import Path
    ROOT = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
    OUT = ROOT / "research/campaign_2026-10-03/eda/nonlinear"
    sys.path.insert(0, str(OUT))
    import nlcommon as nc

    return OUT, nc, np, plt


@app.cell(hide_code=True)
def _(OUT, nc, plt):
    # Fig 01: one raw validation sequence -- price-like, volume-like, trades, aux, targets
    _d = nc.load("valid", 7)
    _fig, _ax = plt.subplots(7, 1, figsize=(16, 18), sharex=True)
    for _i in range(11): _ax[0].plot(_d[f"i0_p{_i}"], lw=.5, label=f"p{_i}")
    _ax[0].set_title("i0 bid price-like p0..p10"); _ax[0].legend(ncol=11, fontsize=6)
    for _i in range(11, 22): _ax[1].plot(_d[f"i0_p{_i}"], lw=.5)
    _ax[1].set_title("i0 ask price-like p11..p21")
    for _i in range(4): _ax[2].plot(_d[f"i0_dp{_i}"], lw=.4)
    _ax[2].set_title("i0 trade price dp0..3")
    for _i in range(8): _ax[3].plot(_d[f"a{_i}"], lw=.5, label=f"a{_i}")
    _ax[3].legend(ncol=8, fontsize=6); _ax[3].set_title("a0..a7")
    for _i in range(11): _ax[4].plot(_d[f"i1_p{_i}"], lw=.5)
    _ax[4].set_title("i1 bid price-like")
    _ax[5].plot(_d["t0"], lw=.5, label="t0"); _ax[5].plot(_d["t1"], lw=.5, label="t1"); _ax[5].legend()
    _ax[6].plot(_d["is_scored"].astype(float), lw=.5); _ax[6].set_title("is_scored")
    _fig.tight_layout(); _fig.savefig(OUT / "fig01_raw_seq.png", dpi=70)
    _fig
    return


@app.cell(hide_code=True)
def _(OUT, nc, np, plt):
    # Fig 02: zoom on a window around scored rows
    _d = nc.load("valid", 7)
    _s = np.where(_d["is_scored"])[0]
    _lo = max(_s[0] - 150, 0); _hi = _lo + 400
    _r = np.arange(_lo, _hi)
    _fig, _ax = plt.subplots(6, 1, figsize=(16, 16), sharex=True)
    for _i in range(11): _ax[0].step(_r, _d[f"i0_p{_i}"][_r], lw=.7, label=f"p{_i}", where="post")
    _ax[0].legend(ncol=11, fontsize=6); _ax[0].set_title("i0 bid price-like")
    for _i in range(11): _ax[1].step(_r, _d[f"i0_v{_i}"][_r], lw=.7, where="post")
    _ax[1].set_title("i0 bid volume-like")
    for _i in range(4): _ax[2].step(_r, _d[f"i0_dp{_i}"][_r], lw=.7, where="post"); _ax[2].step(_r, _d[f"i0_dv{_i}"][_r] - 3, lw=.7, where="post")
    _ax[2].set_title("i0 dp (top) and dv-3 (bottom)")
    for _i in range(8): _ax[3].step(_r, _d[f"a{_i}"][_r], lw=.7, label=f"a{_i}", where="post")
    _ax[3].legend(ncol=8, fontsize=6); _ax[3].set_ylim(-6, 4)
    _ax[4].step(_r, _d["t0"][_r], label="t0", where="post"); _ax[4].step(_r, _d["t1"][_r], label="t1", where="post"); _ax[4].legend()
    _ax[5].step(_r, _d["is_scored"][_r].astype(float), where="post"); _ax[5].set_title("is_scored")
    for _a in _ax: _a.grid(alpha=.3)
    _fig.tight_layout(); _fig.savefig(OUT / "fig02_zoom.png", dpi=70)
    _fig
    return


@app.cell(hide_code=True)
def _(OUT, nc, np):
    # Load cached valid state table (built by build_state.py); train table stays on disk (shared-slice memory is tight)
    VA = dict(np.load(OUT / "cache_valid40.npz"))
    SN = list(VA["SN"]); FEATS = nc.FEATS
    def col(D, name):
        if name in SN: return D["S"][:, SN.index(name)]
        return D["X"][:, FEATS.index(name)]
    def wp(y, p, w=None):
        y = np.clip(y, -2, 2); p = np.clip(p, -2, 2); w = np.abs(y) if w is None else w
        a = y - np.average(y, weights=w); b = p - np.average(p, weights=w)
        return np.sum(w * a * b) / np.sqrt(np.sum(w * a * a) * np.sum(w * b * b))

    return VA, col, wp


@app.cell(hide_code=True)
def _(OUT, VA, col, np, plt):
    # Fig 03: P(scored | state) and E|t| | state -- conditional curves on valid (unscored rows reweighted x8)
    def qbin(v, nb=20):
        q = np.unique(np.quantile(v, np.linspace(0, 1, nb + 1)[1:-1])); return np.searchsorted(q, v), q
    _vars = ["a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7", "act0_h10", "act0_h100", "act0_h1000", "trd0_h100", "tsince_chg0", "tsince_trd0", "act1_h100", "step"]
    _w = np.where(VA["SC"], 1.0, 8.0)
    _fig, _axs = plt.subplots(4, 4, figsize=(18, 14)); _axs = _axs.ravel()
    for _a, _v in zip(_axs, _vars):
        _x = col(VA, _v); _b, _q = qbin(_x)
        _xm = np.array([np.median(_x[_b == i]) for i in range(_b.max() + 1)])
        _ps = np.array([np.average(VA["SC"][_b == i], weights=_w[_b == i]) for i in range(_b.max() + 1)])
        _ey = np.array([np.abs(np.clip(VA["Y"][(_b == i), 0], -2, 2)).mean() for i in range(_b.max() + 1)])
        _eys = np.array([np.abs(np.clip(VA["Y"][(_b == i) & VA["SC"], 0], -2, 2)).mean() for i in range(_b.max() + 1)])
        _a.plot(_xm, _ps, "o-", c="C0", label="P(scored)"); _a.set_title(_v); _a.set_ylabel("P(scored)", color="C0")
        _a2 = _a.twinx(); _a2.plot(_xm, _ey, "s--", c="C3", ms=3, label="E|t0| all"); _a2.plot(_xm, _eys, "^-", c="C1", ms=3, label="E|t0| scored")
        if _v == "a0": _a2.legend(fontsize=7)
    _fig.tight_layout(); _fig.savefig(OUT / "fig03_scored_vs_state.png", dpi=65)
    _fig
    return


@app.cell(hide_code=True)
def _(OUT, VA, col, np, plt):
    # Fig 04: within tick-unit class -- P(scored), E|k0| (ticks) and corr(t0,t1) vs a3 / a2 / act0_h100 deciles
    _units = np.unique(VA["U"])
    _w = np.where(VA["SC"], 1.0, 8.0)
    _fig, _axs = plt.subplots(3, 3, figsize=(17, 12))
    for _j, _v in enumerate(["a3", "a2", "act0_h100"]):
        _x = col(VA, _v)
        for _ci, _u in enumerate(_units):
            _m = VA["U"] == _u
            _q = np.quantile(_x[_m], np.linspace(0, 1, 11)); _b = np.clip(np.searchsorted(_q, _x[_m], side="right") - 1, 0, 9)
            _xm = [np.median(_x[_m][_b == i]) for i in range(10)]
            _ps = [np.average(VA["SC"][_m][_b == i], weights=_w[_m][_b == i]) for i in range(10)]
            _ek = [np.abs(VA["K"][_m][_b == i, 0]).mean() for i in range(10)]
            _cc = [np.corrcoef(VA["K"][_m][_b == i, 0], VA["K"][_m][_b == i, 1])[0, 1] for i in range(10)]
            _axs[0, _j].plot(_xm, _ps, "o-", label=f"unit {_u:.3f}"); _axs[1, _j].plot(_xm, _ek, "o-"); _axs[2, _j].plot(_xm, _cc, "o-")
        _axs[0, _j].set_title(f"P(scored) vs {_v}"); _axs[1, _j].set_title(f"E|k0| ticks vs {_v}"); _axs[2, _j].set_title(f"corr(k0,k1) vs {_v}")
    _axs[0, 0].legend()
    _fig.tight_layout(); _fig.savefig(OUT / "fig04_class_conditional.png", dpi=65)
    _fig
    return


@app.cell(hide_code=True)
def _(OUT, VA, col, np, plt, wp):
    # Fig 05: state-dependent calibration of the linear (ridge) signal on scored valid rows.
    # For each state variable: within-bin WP, within-bin weighted slope of y on pred, and the share of total |y| weight.
    PR = dict(np.load(OUT / "preds_valid40.npz"))
    _sc = VA["SC"]
    def wslope(y, p, w):
        pm = np.average(p, weights=w); ym = np.average(y, weights=w)
        return np.sum(w * (p - pm) * (y - ym)) / np.sum(w * (p - pm) ** 2)
    _states = {"unit": VA["U"], "a3": col(VA, "a3"), "a2": col(VA, "a2"), "act0_h100": col(VA, "act0_h100"),
               "tsince_chg0": col(VA, "tsince_chg0"), "trd0_h100": col(VA, "trd0_h100"), "a6": col(VA, "a6"), "step": col(VA, "step")}
    _fig, _axs = plt.subplots(3, len(_states), figsize=(24, 10))
    STATE_TAB = {}
    for _j, (_nm, _x) in enumerate(_states.items()):
        _xs = _x[_sc]
        if _nm == "unit": _b = np.searchsorted(np.unique(_xs), _xs); _nb = len(np.unique(_xs))
        else:
            _q = np.unique(np.quantile(_xs, np.linspace(0, 1, 7)[1:-1])); _b = np.searchsorted(_q, _xs); _nb = len(_q) + 1
        _xm = np.array([np.median(_xs[_b == i]) for i in range(_nb)])
        for _k in range(2):
            _y = np.clip(VA["Y"][_sc, _k], -2, 2); _p = PR[f"ridge_t{_k}"][_sc]; _w = np.abs(_y)
            _r = [wp(_y[_b == i], _p[_b == i]) for i in range(_nb)]
            _s = [wslope(_y[_b == i], _p[_b == i], _w[_b == i]) for i in range(_nb)]
            _sh = [_w[_b == i].sum() / _w.sum() for i in range(_nb)]
            _axs[0, _j].plot(_xm, _r, "o-", label=f"t{_k}"); _axs[1, _j].plot(_xm, _s, "o-"); _axs[2, _j].plot(_xm, _sh, "o-")
            STATE_TAB[(_nm, _k)] = (_xm, _r, _s, _sh)
        _axs[0, _j].set_title(f"WP within bin | {_nm}"); _axs[1, _j].set_title(f"slope y~pred | {_nm}"); _axs[2, _j].set_title(f"|y| weight share | {_nm}")
    _axs[0, 0].legend()
    _fig.tight_layout(); _fig.savefig(OUT / "fig05_state_calibration.png", dpi=60)
    _fig
    return (PR,)


@app.cell(hide_code=True)
def _(OUT, VA, np, plt):
    # Fig 06: joint law of (k0, k1) in tick units per class (scored rows), log colour. Mirror + floor structure.
    _fig, _axs = plt.subplots(1, 4, figsize=(22, 5.5))
    for _a, _u in zip(_axs, np.unique(VA["U"])):
        _m = (VA["U"] == _u) & VA["SC"]
        _L = 8 if _u > 0.4 else 16
        _h, _, _ = np.histogram2d(np.clip(VA["K"][_m, 0], -_L, _L), np.clip(VA["K"][_m, 1], -_L, _L), bins=np.arange(-_L - .5, _L + 1.5))
        _im = _a.imshow(np.log10(_h.T / _h.sum() + 1e-5), origin="lower", extent=[-_L - .5, _L + .5, -_L - .5, _L + .5], cmap="viridis", vmin=-5, vmax=-0.4)
        _a.set_xlabel("k0 = t0/unit"); _a.set_ylabel("k1 = t1/unit"); _a.set_title(f"unit {_u:.3f}: n={_m.sum()} (clip +-{_L})")
        _a.axhline(-3, c="w", lw=.4, ls=":"); _a.axvline(-3, c="w", lw=.4, ls=":")
    _fig.colorbar(_im, ax=_axs, label="log10 P"); _fig.savefig(OUT / "fig06_joint_k0k1.png", dpi=65)
    _fig
    return


@app.cell(hide_code=True)
def _(PR, VA, col, np, wp):
    # Test A: state-dependent gain/offset recalibration of the ridge signal, 2-fold cross-fit by sequence on scored rows.
    def recal_cv(pcol, k, states, inter=True, seed=0):
        sc = VA["SC"]; g = VA["G"][sc]; y = np.clip(VA["Y"][sc, k], -2, 2); p = PR[pcol][sc]; w = np.abs(y)
        S = np.stack(states, 1)[sc] if states else np.zeros((sc.sum(), 0))
        S = (S - S.mean(0)) / (S.std(0) + 1e-9)
        F = [np.ones_like(p), p] + ([S[:, j] for j in range(S.shape[1])]) + ([p * S[:, j] for j in range(S.shape[1])] if inter else [])
        F = np.stack(F, 1)
        ug = np.unique(g); rng = np.random.default_rng(seed); fa = set(rng.choice(ug, len(ug) // 2, replace=False))
        ina = np.array([gg in fa for gg in g]); out = np.zeros_like(p)
        for tr, te in [(ina, ~ina), (~ina, ina)]:
            A = F[tr] * np.sqrt(w[tr])[:, None]; b = y[tr] * np.sqrt(w[tr])
            beta = np.linalg.lstsq(A.T @ A + 1e-3 * np.eye(A.shape[1]), A.T @ b, rcond=None)[0]
            out[te] = F[te] @ beta
        return wp(y, p), wp(y, out)
    _cls = [(VA["U"] == u).astype(float) for u in np.unique(VA["U"])[:-1]]
    _sv = {n: col(VA, n) for n in ["a3", "a2", "act0_h100", "tsince_chg0", "trd0_h100", "act0_h10", "a6"]}
    for _k in range(2):
        print(f"t{_k}: raw/affine-only", np.round(recal_cv(f"ridge_t{_k}", _k, [], inter=False), 4))
        print(f"t{_k}: class offsets+gains", np.round(recal_cv(f"ridge_t{_k}", _k, _cls), 4))
        print(f"t{_k}: class + a3,a2,act,tsince,trd gains", np.round(recal_cv(f"ridge_t{_k}", _k, _cls + [_sv[n] for n in ["a3", "a2", "act0_h100", "tsince_chg0", "trd0_h100"]]), 4))
        print(f"t{_k}: same, offsets only (no gain)", np.round(recal_cv(f"ridge_t{_k}", _k, _cls + [_sv[n] for n in ["a3", "a2", "act0_h100", "tsince_chg0", "trd0_h100"]], inter=False), 4))

    return


@app.cell(hide_code=True)
def _(OUT, PR, VA, np, plt, wp):
    # Fig 07: link shape -- E[y | pred bin] and metric-optimal E[y|y|]/E[|y|] per class, scored rows (40-seq cache)
    _fig, _axs = plt.subplots(2, 4, figsize=(20, 9))
    for _k in range(2):
        for _a, _u in zip(_axs[_k], np.unique(VA["U"])):
            _m = VA["SC"] & (VA["U"] == _u)
            _p = PR[f"ridge_t{_k}"][_m]; _y = np.clip(VA["Y"][_m, _k], -2, 2)
            _q = np.quantile(_p, np.linspace(0, 1, 21)); _b = np.clip(np.searchsorted(_q, _p, side="right") - 1, 0, 19)
            _pm = [np.median(_p[_b == i]) for i in range(20)]
            _ey = [_y[_b == i].mean() for i in range(20)]
            _eo = [np.sum(_y[_b == i] * np.abs(_y[_b == i])) / np.sum(np.abs(_y[_b == i])) for i in range(20)]
            _pf = [np.mean(_y[_b == i] <= -1.3) for i in range(20)]
            _a.plot(_pm, _ey, "o-", label="E[y|p]"); _a.plot(_pm, _eo, "s-", label="E[y|y|]/E|y|")
            _a.plot(_pm, _pm, "k:", lw=.8)
            _a2 = _a.twinx(); _a2.plot(_pm, _pf, "x--", c="C3", label="P(y<=-1.3)"); _a2.set_ylim(0, 1)
            _a.set_title(f"t{_k} unit {_u:.3f}: WP in-class {wp(_y, _p):.3f}"); _a.set_xlabel("ridge pred")
            if _u == np.unique(VA["U"])[0]: _a.legend(fontsize=8, loc="upper left")
    _fig.tight_layout(); _fig.savefig(OUT / "fig07_link_shape.png", dpi=65)
    _fig
    return


@app.cell(hide_code=True)
def _(OUT, VA, nc, np, plt):
    # Fig 08: regime persistence -- ACF of state variables and |k0| (14 valid seqs), and a3/scored over one sequence
    _R = dict(np.load(OUT / "regime_speed.npz"))
    _fig, _axs = plt.subplots(1, 2, figsize=(18, 5.5))
    for _n in ["a2", "a3", "a6", "act0_h100", "abs_k0", "chg0", "trd0", "absd1_p"]:
        _axs[0].semilogx(_R["lags"], np.nanmean(_R[_n], 0), "o-", ms=3, label=_n)
    _axs[0].axhline(0.5, c="k", lw=.5, ls=":"); _axs[0].axhline(0, c="k", lw=.5); _axs[0].legend(ncol=2); _axs[0].set_xlabel("lag (rows)"); _axs[0].set_title(f"ACF (mean over seqs); CV of inter-change gaps: median {np.median(_R['cv']):.1f}")
    _d = nc.load("valid", int(np.unique(VA["G"])[0]))
    _axs[1].plot(_d["a3"], lw=.5, label="a3"); _axs[1].plot(_d["a2"] - 3, lw=.5, label="a2 - 3")
    _axs[1].plot(np.abs(_d["t0"]) + 2, lw=.5, label="|t0| + 2")
    _s = np.where(_d["is_scored"])[0]; _axs[1].plot(_s, np.full(len(_s), -4.5), "|", ms=8, c="k", label="scored")
    _axs[1].legend(ncol=4); _axs[1].set_title("one sequence: a3 / a2 / |t0| / scored rows")
    _fig.tight_layout(); _fig.savefig(OUT / "fig08_regime_persistence.png", dpi=65)
    _fig
    return


if __name__ == "__main__":
    app.run()
