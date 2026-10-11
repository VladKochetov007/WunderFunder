import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    from pathlib import Path

    import marimo as mo
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    ROOT = Path(__file__).resolve().parents[5] if "__file__" in globals() else Path.cwd()
    HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    PLOTS = HERE / "plots"
    PLOTS.mkdir(exist_ok=True)
    return HERE, PLOTS, ROOT, json, mo, np, plt


@app.cell
def _(ROOT, json, np):
    # Fold-1 valid _rows (457 groups), scored _rows only (mask & step>=99). Read-only, CPU, ~1 core.
    CACHE = ROOT / "cache"
    split = json.loads((ROOT / "research/campaign_2026-09-29/artifacts/splits/fold1.json").read_text())
    groups = np.asarray([int(g) for g in split["evaluation"]["valid"]])
    comp_of = {int(r["row_group"]): int(r["component"]) for r in split["sequences"]["valid"]}
    comp = np.asarray([comp_of[int(g)] for g in groups])
    with np.load(CACHE / "valid_targets.npz") as _z:
        pos = {int(g): i for i, g in enumerate(_z["row_group"])}
        idx = np.asarray([pos[int(g)] for g in groups])
        Yfull = np.clip(_z["y"][idx].astype(np.float32), -2, 2)
        M = _z["mask"][idx].copy()
        mclass = _z["market_class"][idx]
    M &= (np.arange(M.shape[1]) >= 99)[None]
    MODELS = {
        "student": "pred_h230_f1_s3_ep3_ep2_valid",
        "teacher_ctrl": "pred_vt_f1_ctrl_s3_valid",
        "gru384_fixmix": "pred_vt_f1_gru384_fixmix_valid",
    }
    Pfull = {}
    for name, stem in MODELS.items():
        side = json.loads((CACHE / f"{stem}.json").read_text())
        order = {int(g): i for i, g in enumerate(side["row_groups"])}
        arr = np.load(CACHE / f"{stem}.npy", mmap_mode="r")
        sel = np.asarray([order[int(g)] for g in groups])
        srt = np.argsort(sel)
        out = np.empty((len(groups), arr.shape[1], 2), np.float32)
        out[srt] = np.asarray(arr[sel[srt]])
        Pfull[name] = np.clip(out, -2, 2)
    return CACHE, M, MODELS, Pfull, Yfull, comp, groups, mclass


@app.cell
def _(M, Pfull, Yfull, comp, groups, mclass, np):
    G, T, _ = Yfull.shape
    # label run position (_rows since the label last changed), per target; run start = 0
    chg = np.zeros((G, T, 2), bool)
    chg[:, 1:] = Yfull[:, 1:] != Yfull[:, :-1]
    chg[:, 0] = True
    tt = np.broadcast_to(np.arange(T, dtype=np.int32)[None, :, None], (G, T, 2))
    last = np.maximum.accumulate(np.where(chg, tt, np.int32(0)), axis=1)
    runpos_full = (tt - last).astype(np.int32)
    # run length (for the run containing each row): next change
    nxt = np.minimum.accumulate(np.where(chg, tt, np.int32(T))[:, ::-1], axis=1)[:, ::-1]  # next change at or after t
    nxt_after = np.full((G, T, 2), T, np.int32)
    nxt_after[:, :-1] = nxt[:, 1:]
    runlen_full = (nxt_after - last).astype(np.int32)
    del nxt, nxt_after, last, tt
    gi, ti = np.nonzero(M)
    Y = Yfull[gi, ti]  # (n,2)
    P = {_k: _v[gi, ti] for _k, _v in Pfull.items()}
    RP = runpos_full[gi, ti]
    RL = runlen_full[gi, ti]
    CLS = mclass[gi]
    COMP = comp[gi]
    n = len(gi)
    del runlen_full, runpos_full
    return CLS, COMP, G, P, RL, RP, T, Y, chg, gi, n, ti


@app.cell
def _(CLS, G, M, Y, gi, groups, n, np, ti, CACHE):
    # join causal event-clock features from heur_valid.npz via (valid-position, step)
    import zipfile

    from numpy.lib import format as _npfmt

    with np.load(CACHE / "heur_valid.npz") as _z:
        names = [str(_s) for _s in _z["names"]]
        hs, hst = _z["seq"], _z["step"]
    keep = np.isin(hs, groups)
    hs, hst = hs[keep], hst[keep]
    # stream X.npy in row chunks (the full array is ~2 GB; the RAM budget is 3 GB)
    with zipfile.ZipFile(CACHE / "heur_valid.npz") as _zf, _zf.open("X.npy") as _fh:
        _npfmt.read_magic(_fh)
        _shape, _fortran, _dt = _npfmt.read_array_header_1_0(_fh)
        _parts, _row = [], 0
        while _row < _shape[0]:
            _m = min(200_000, _shape[0] - _row)
            _buf = np.frombuffer(_fh.read(_m * _shape[1] * 4), np.float32).reshape(_m, _shape[1])
            _parts.append(_buf[keep[_row:_row + _m]].copy())
            _row += _m
    X = np.concatenate(_parts)
    del _parts
    glob = {int(g): i for i, g in enumerate(groups)}
    hkey = np.fromiter((glob[int(_s)] for _s in hs), np.int64, len(hs)) * 20000 + hst
    key = gi.astype(np.int64) * 20000 + ti
    o = np.argsort(hkey)
    _p = np.clip(np.searchsorted(hkey[o], key), 0, len(o) - 1)
    ok = hkey[o][_p] == key
    FE = np.full((n, X.shape[1]), np.nan, np.float32)
    FE[ok] = X[o[_p[ok]]]
    FEAT = {_nm: FE[:, i] for i, _nm in enumerate(names)}
    join_rate = float(ok.mean())
    return FEAT, join_rate, names


@app.cell
def _(P, Y, np):
    W = np.abs(Y).astype(np.float64)
    mu_y = (W * Y).sum(0) / W.sum(0)
    stats = {}
    for _k, _p in P.items():
        mu_p = (W * _p).sum(0) / W.sum(0)
        syy = (W * (Y - mu_y) ** 2).sum(0)
        spp = (W * (_p - mu_p) ** 2).sum(0)
        # per-row additive contribution to weighted Pearson (sums to corr per target)
        contrib = W * (Y - mu_y) * (_p - mu_p) / np.sqrt(syy * spp)
        stats[_k] = dict(mu_p=mu_p, contrib=contrib, wp=contrib.sum(0), spp=spp)
    resid = {_k: (Y - p) for _k, p in P.items()}
    rss = {_k: W * resid[_k] ** 2 for _k in P}
    wp_report = {_k: dict(t0=float(_v["wp"][0]), t1=float(_v["wp"][1]), wp=float(_v["wp"].mean())) for _k, _v in stats.items()}
    wp_report
    return W, mu_y, resid, rss, stats, wp_report


@app.cell
def _(W, Y, P, np, rss, stats):
    def table(labels, order=None, model="student"):
        """Per-slice: weight share, share of WP numerator (avg over targets), local weighted corr, calibration slope, residual-SS share."""
        c = stats[model]["contrib"]
        p = P[model]
        _rows = []
        _cats = np.unique(labels) if order is None else order
        _tot_w, _tot_r = W.sum(0), rss[model].sum(0)
        for _cat in _cats:
            _s = labels == _cat
            if _s.sum() == 0:
                continue
            _w = W[_s]
            _ws = _w.sum(0)
            _yy, _pp = Y[_s], p[_s]
            _my = (_w * _yy).sum(0) / _ws
            _mp = (_w * _pp).sum(0) / _ws
            _cov = (_w * (_yy - _my) * (_pp - _mp)).sum(0)
            _vy = (_w * (_yy - _my) ** 2).sum(0)
            _vp = (_w * (_pp - _mp) ** 2).sum(0)
            corr = _cov / np.sqrt(_vy * _vp)
            slope = (_w * _yy * _pp).sum(0) / (_w * _pp * _pp).sum(0)
            _rows.append(dict(
                slice=str(_cat), n=int(_s.sum()), w_share=float((_ws / _tot_w).mean()),
                wp_share=float((c[_s].sum(0) / stats[model]["wp"]).mean()),
                local_corr=float(corr.mean()), slope=float(slope.mean()),
                rss_share=float((rss[model][_s].sum(0) / _tot_r).mean()),
            ))
        return _rows

    def qbins(_x, _q=5):
        e = np.nanquantile(_x, np.linspace(0, 1, _q + 1)[1:-1])
        e = np.unique(e)
        return np.digitize(_x, e), e
    return qbins, table


@app.cell
def _(CLS, FEAT, P, RL, RP, W, Y, COMP, np, qbins, table):
    SL = {}
    SL["class"] = table(CLS)
    ti_all = None
    return SL, ti_all


@app.cell
def _(SL, ti, FEAT, CLS, RL, RP, W, Y, COMP, np, qbins, table, gi, P, mu_y, rss, stats):
    # row age
    age = np.digitize(ti, [300, 1000])
    SL["age"] = table(np.array(["99-299", "300-999", ">=1000"])[age])
    # |y| bucket (target-wise weights => use max |y| of the two targets for bucket)
    ay = np.abs(Y).max(1)
    ya = np.digitize(ay, [0.05, 0.25, 0.75, 1.5])
    SL["|y|max bucket"] = table(np.array(["0-.05", ".05-.25", ".25-.75", ".75-1.5", "1.5-2"])[ya])
    # label run position (target 0 position; targets share runs mostly)
    rp = RP.min(1)
    rpb = np.digitize(rp, [1, 2, 4, 8, 16])
    SL["run position"] = table(np.array(["start(0)", "1", "2-3", "4-7", "8-15", "16+"])[rpb])
    # volatility regime: causal EMA abs price move
    for _nm in ("absd_p0_hl64", "rate_book_hl64", "since_book", "since_trade", "rate_trade_hl64", "tradevol_hl128"):
        _b, _e = qbins(FEAT[_nm], 5)
        _lab = np.array([f"Q{i+1}" for i in range(len(_e) + 1)])[_b]
        SL[_nm] = table(_lab)
    return age, ay, rp, rpb, ya


@app.cell
def _(SL):
    import pandas as pd
    frames = {_k: pd.DataFrame(_v) for _k, _v in SL.items()}
    frames
    return frames, pd


@app.cell
def _(PLOTS, frames, np, plt):
    keys = list(frames)
    ncol = 3
    nrow = int(np.ceil(len(keys) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(16, 3.2 * nrow))
    for _ax, _k in zip(axs.ravel(), keys):
        _d = frames[_k]
        _x = np.arange(len(_d))
        _ax.bar(_x - 0.27, _d.w_share, 0.27, label="weight share", color="#999")
        _ax.bar(_x, _d.wp_share, 0.27, label="WP numerator share", color="#1f77b4")
        _ax.bar(_x + 0.27, _d.rss_share, 0.27, label="resid SS share", color="#d62728")
        _ax.set_xticks(_x)
        _ax.set_xticklabels(_d.slice, rotation=30, fontsize=7)
        _ax.set_title(_k, fontsize=9)
    axs.ravel()[0].legend(fontsize=7)
    for _ax in axs.ravel()[len(keys):]:
        _ax.axis("off")
    fig.tight_layout()
    fig.savefig(PLOTS / "slice_shares.png", dpi=110)
    fig
    return


@app.cell
def _(COMP, PLOTS, W, gi, np, plt, stats, rss):
    # component / sequence concentration of the WP numerator and residual SS
    cn = stats["student"]["contrib"].mean(1)
    comp_ids, inv = np.unique(COMP, return_inverse=True)
    comp_c = np.bincount(inv, cn)
    comp_w = np.bincount(inv, W.mean(1))
    comp_r = np.bincount(inv, rss["student"].mean(1))
    seq_c = np.bincount(gi, cn)
    def topk(_v, fracs=(0.5, 0.8, 0.95)):
        _s = np.sort(_v)[::-1]
        _cs = np.cumsum(_s) / _s.sum()
        return {f: int(np.searchsorted(_cs, f) + 1) for f in fracs}
    conc = dict(n_components=len(comp_ids), n_sequences=len(seq_c),
                comp_numerator=topk(comp_c), comp_weight=topk(comp_w), comp_rss=topk(comp_r),
                seq_numerator=topk(seq_c))
    fig2, ax2 = plt.subplots(1, 2, figsize=(11, 3.5))
    for _v, _nm in ((comp_c, "WP numerator"), (comp_w, "|y| weight"), (comp_r, "resid SS")):
        _s = np.sort(_v)[::-1]
        ax2[0].plot(np.arange(1, len(_s) + 1) / len(_s), np.cumsum(_s) / _s.sum(), label=_nm)
    ax2[0].set_xlabel("fraction of components (sorted desc)")
    ax2[0].set_ylabel("cumulative share")
    ax2[0].legend()
    # per-component local quality: contribution / weight
    eff = comp_c / comp_w
    ax2[1].hist(eff / (cn.sum() / W.mean(1).sum()), bins=40)
    ax2[1].set_xlabel("component efficiency relative to global (numerator share / weight share)")
    fig2.tight_layout()
    fig2.savefig(PLOTS / "component_concentration.png", dpi=110)
    conc, np.sum(comp_c)
    return comp_c, comp_w, conc, eff, seq_c


@app.cell
def _(P, W, Y, mu_y, np, stats, rss, resid, PLOTS, plt):
    # disagreement vs residual; model blend oracle; residual correlation
    dis_ts = np.abs(P["student"] - P["teacher_ctrl"]).max(1)
    dis_sg = np.abs(P["student"] - P["gru384_fixmix"]).max(1)
    res_corr = {}
    for _a in P:
        for _b in P:
            if _a < _b:
                _ra, _rb = resid[_a], resid[_b]
                res_corr[f"{_a}|{_b}"] = [float(np.corrcoef(_ra[:, _k], _rb[:, _k])[0, 1]) for _k in range(2)]
    def wp_of(_q):
        _mp = (W * _q).sum(0) / W.sum(0)
        _c = (W * (Y - mu_y) * (_q - _mp)).sum(0) / np.sqrt((W * (Y - mu_y) ** 2).sum(0) * (W * (_q - _mp) ** 2).sum(0))
        return _c
    blends = {"student": wp_of(P["student"]), "mean3": wp_of(sum(P.values()) / 3),
              "student+teacher": wp_of((P["student"] + P["teacher_ctrl"]) / 2),
              "student+gru": wp_of((P["student"] + P["gru384_fixmix"]) / 2)}
    blends = {_k: float(_v.mean()) for _k, _v in blends.items()}
    # disagreement deciles
    db = np.digitize(dis_ts, np.quantile(dis_ts, np.linspace(0, 1, 11)[1:-1]))
    dtab = []
    for _q in range(10):
        _s = db == _q
        _w = W[_s]
        dtab.append(dict(decile=_q + 1, mean_dis=float(dis_ts[_s].mean()), w_share=float((_w.sum(0) / W.sum(0)).mean()),
                         rss_share=float((rss["student"][_s].sum(0) / rss["student"].sum(0)).mean()),
                         rss_per_w=float((rss["student"][_s].sum(0) / _w.sum(0)).mean())))
    fig3, ax3 = plt.subplots(figsize=(6, 3.5))
    ax3.bar([_d["decile"] for _d in dtab], [_d["rss_share"] for _d in dtab], color="#d62728", label="resid SS share")
    ax3.plot([_d["decile"] for _d in dtab], [_d["w_share"] for _d in dtab], "k.-", label="weight share")
    ax3.set_xlabel("|student - teacher| disagreement decile")
    ax3.legend()
    fig3.tight_layout()
    fig3.savefig(PLOTS / "disagreement_deciles.png", dpi=110)
    return blends, dis_sg, dis_ts, dtab, res_corr, wp_of


@app.cell
def _(Pfull, Yfull, M, PLOTS, np, plt):
    # lead/lag: does the prediction at t+_k correlate better with y_t? (_k>0: prediction lags the label)
    lag_out = {}
    ks = list(range(-6, 25))
    for model in ("student",):
        P3 = Pfull[model]
        _res = []
        for _k in ks:
            if _k >= 0:
                _ys, _ps = Yfull[:, : Yfull.shape[1] - _k], P3[:, _k:]
                _ms = M[:, : M.shape[1] - _k] & M[:, _k:]
            else:
                _ys, _ps = Yfull[:, -_k:], P3[:, : P3.shape[1] + _k]
                _ms = M[:, -_k:] & M[:, : M.shape[1] + _k]
            _w = np.abs(_ys[_ms]).astype(np.float64)
            _y_, _p_ = _ys[_ms].astype(np.float64), _ps[_ms].astype(np.float64)
            _my, _mp = (_w * _y_).sum(0) / _w.sum(0), (_w * _p_).sum(0) / _w.sum(0)
            _c = (_w * (_y_ - _my) * (_p_ - _mp)).sum(0) / np.sqrt((_w * (_y_ - _my) ** 2).sum(0) * (_w * (_p_ - _mp) ** 2).sum(0))
            _res.append(float(_c.mean()))
        lag_out[model] = _res
    figl, axl = plt.subplots(figsize=(6, 3.5))
    axl.plot(ks, lag_out["student"], "o-")
    axl.set_xlabel("k: corr(y_t, pred_{t+_k})")
    axl.set_ylabel("weighted corr")
    figl.tight_layout()
    figl.savefig(PLOTS / "leadlag.png", dpi=110)
    lead = dict(zip(ks, lag_out["student"]))
    return ks, lead, lag_out


@app.cell
def _(CLS, FEAT, M, P, Pfull, W, Y, Yfull, gi, mu_y, names, np, rss, table, qbins, wp_of, chg, ti, RP):
    from scipy.signal import lfilter

    extra = {}
    # (a) saturation and sign accuracy
    sat = np.abs(Y) >= 2 - 1e-6
    extra["sat_row_frac"] = [float(_x) for _x in sat.mean(0)]
    extra["sat_weight_share"] = [float((W[:, _k] * sat[:, _k]).sum() / W[:, _k].sum()) for _k in range(2)]
    _sgn = (np.sign(P["student"]) == np.sign(Y)).astype(float)
    extra["sign_acc_w"] = [float((W[:, _k] * _sgn[:, _k]).sum() / W[:, _k].sum()) for _k in range(2)]
    _ay = np.abs(Y).max(1)
    for _nm, _sel in (("sat", sat.all(1)), ("nonsat", ~sat.any(1))):
        extra[f"sign_acc_{_nm}"] = float((_sgn[_sel] * W[_sel]).sum() / W[_sel].sum())
        extra[f"w_share_{_nm}"] = float(W[_sel].sum() / W.sum())
    # (b) class x staleness interaction
    _b, _e = qbins(FEAT["since_book"], 4)
    _stale = np.where(_b >= len(_e), "stale", "fresh")
    _cls = np.array([f"c{_c}" for _c in CLS])
    extra["class_x_stale"] = table(np.char.add(np.char.add(_cls, "_"), _stale))
    _rpb = np.where(RP.min(1) >= 16, "long", "short")
    extra["class_x_runlong"] = table(np.char.add(np.char.add(_cls, "_"), _rpb))
    # (c) between-run vs within-run decomposition of residual SS (per target, weights |y|)
    _G, _T = Yfull.shape[:2]
    _between = _tot = 0.0
    for _k in range(2):
        _R = (Yfull[:, :, _k] - Pfull["student"][:, :, _k]).astype(np.float64)
        _rid = (np.cumsum(chg[:, :, _k], axis=1) + np.arange(_G)[:, None] * _T).ravel()
        _wf = (np.abs(Yfull[:, :, _k]) * M).astype(np.float64)
        _bs = np.bincount(_rid, (_wf * _R).ravel())
        _ws = np.bincount(_rid, _wf.ravel())
        _between += (_bs[_ws > 0] ** 2 / _ws[_ws > 0]).sum()
        _tot += (_wf * _R ** 2).sum()
    extra["resid_ss_between_run_share"] = float(_between / _tot)
    del _R, _rid, _wf, _bs, _ws
    # (d) residual persistence: lag-k autocorr of residual within scored pairs
    _Rf = (Yfull - Pfull["student"])
    _ac = {}
    for _lag in (1, 4, 16, 64):
        _m2 = M[:, _lag:] & M[:, :-_lag]
        _a, _bb = _Rf[:, _lag:][_m2], _Rf[:, :-_lag][_m2]
        _ac[_lag] = float(np.corrcoef(_a[:, 0], _bb[:, 0])[0, 1]), float(np.corrcoef(_a[:, 1], _bb[:, 1])[0, 1])
    extra["resid_autocorr"] = _ac
    del _Rf
    # (e) causal EMA smoothing of the student prediction (reset per sequence)
    _sm = {}
    for _hl in (1, 2, 4, 8, 16):
        _a = 1 - 0.5 ** (1 / _hl)
        _f = lfilter([_a], [1, -(1 - _a)], Pfull["student"].astype(np.float64), axis=1)
        _pf = _f[gi, ti]
        _sm[f"ema{_hl}"] = float(wp_of(_pf.astype(np.float32)).mean())
        _sm[f"mix_p_ema{_hl}"] = float(wp_of(((_pf + P["student"]) / 2).astype(np.float32)).mean())
    del _f, _pf
    extra["causal_ema_smoothing"] = _sm
    # (f) linear headroom: weighted ridge of student residual on causal clocks/features (+ pred), train even / test odd sequences
    _nf = len(names)
    _Z = np.empty((len(gi), _nf + 3), np.float32)
    for _i, _n in enumerate(names):
        _col = np.nan_to_num(FEAT[_n], nan=0.0)
        _lo, _hi = np.percentile(_col[::10], [0.5, 99.5])
        _col = np.clip(_col, _lo, _hi)
        _Z[:, _i] = (_col - _col.mean()) / (_col.std() + 1e-6)
    _Z[:, _nf:_nf + 2] = P["student"]
    _Z[:, -1] = 1.0
    _tr = (gi % 2 == 0)
    _res = Y - P["student"]
    _gain = []
    for _k in range(2):
        for _nm, _sel in (("all", np.ones(len(gi), bool)), ("class3", CLS == 3)):
            _trs, _tes = _tr & _sel, ~_tr & _sel
            _A = 50.0 * np.eye(_Z.shape[1])
            _bv = np.zeros(_Z.shape[1])
            _ids = np.flatnonzero(_trs)
            for _c0 in range(0, len(_ids), 100_000):
                _ii = _ids[_c0:_c0 + 100_000]
                _zz = _Z[_ii] * np.sqrt(W[_ii, _k])[:, None]
                _A += (_zz.T @ _zz).astype(np.float64)
                _bv += _zz.T @ (_res[_ii, _k] * np.sqrt(W[_ii, _k])).astype(np.float64)
            _beta = np.linalg.solve(_A, _bv)
            _fit = (_Z[_tes] @ _beta.astype(np.float32))
            _r0 = (W[_tes, _k] * _res[_tes, _k] ** 2).sum()
            _r1 = (W[_tes, _k] * (_res[_tes, _k] - _fit) ** 2).sum()
            _gain.append(dict(target=_k, subset=_nm, heldout_resid_ss_reduction=float(1 - _r1 / _r0)))
    extra["linear_residual_headroom"] = _gain
    del _Z
    return extra, lfilter


@app.cell
def _(CLS, COMP, FEAT, G, M, P, Pfull, ROOT, RP, W, Y, Yfull, chg, groups, gi, json, mu_y, np, stats, ti, wp_of):
    more = {}
    # (i) are label changes tied to book events?  since_book==0 (book event on this row) at run start vs inside runs
    _sb = FEAT["since_book"]
    _st = RP.min(1) == 0
    more["since_book_quantiles_runstart"] = [float(_q) for _q in np.nanquantile(_sb[_st], [0.1, 0.5, 0.9])]
    more["since_book_quantiles_inside"] = [float(_q) for _q in np.nanquantile(_sb[~_st], [0.1, 0.5, 0.9])]
    more["since_book_min"] = float(np.nanmin(_sb))
    # (ii) causal one-step "advance" of the prediction: p_t + a (p_t - p_{t-1}), within sequence
    _adv = {}
    _d1 = Pfull["student"][:, 1:] - Pfull["student"][:, :-1]
    _dd = np.zeros_like(Pfull["student"])
    _dd[:, 1:] = _d1
    _dgrid = _dd[gi, ti]
    for _a in (0.25, 0.5, 1.0, 2.0):
        _adv[str(_a)] = float(wp_of((P["student"] + _a * _dgrid).astype(np.float32)).mean())
    more["advance_filter"] = _adv
    del _dd, _d1
    # (iii) per-component local correlation inside class 3 and heterogeneity
    _cid, _inv = np.unique(COMP, return_inverse=True)
    _rows = []
    for _i, _c in enumerate(_cid):
        _s = _inv == _i
        _w = W[_s].astype(np.float64)
        _y, _p = Y[_s], P["student"][_s]
        _wy = _w.sum(0)
        _my, _mp = (_w * _y).sum(0) / _wy, (_w * _p).sum(0) / _wy
        _cv = (_w * (_y - _my) * (_p - _mp)).sum(0) / np.sqrt((_w * (_y - _my) ** 2).sum(0) * (_w * (_p - _mp) ** 2).sum(0))
        _rows.append((int(_c), int(np.bincount(CLS[_s]).argmax()), float(_wy.mean()), float(_cv.mean()), int(_s.sum())))
    _rows = np.array(_rows)
    more["component_corr_by_class"] = {
        int(_k): dict(n=int((_rows[:, 1] == _k).sum()), median_corr=float(np.median(_rows[_rows[:, 1] == _k, 3])),
                      p10=float(np.percentile(_rows[_rows[:, 1] == _k, 3], 10)), p90=float(np.percentile(_rows[_rows[:, 1] == _k, 3], 90)),
                      weight_share=float(_rows[_rows[:, 1] == _k, 2].sum() / _rows[:, 2].sum()))
        for _k in range(4)}
    more["class_purity"] = float(np.mean([(CLS[COMP == _c] == int(_rows[_i, 1])).mean() for _i, _c in enumerate(_cid)]))
    more["comp_n_big_low"] = int(((_rows[:, 3] < 0.3) & (_rows[:, 2] > np.median(_rows[:, 2]))).sum())
    # (iv) residual correlation between targets
    _res = Y - P["student"]
    more["resid_corr_t0_t1"] = float(np.corrcoef(_res[:, 0], _res[:, 1])[0, 1])
    more["pred_corr_t0_t1"] = float(np.corrcoef(P["student"][:, 0], P["student"][:, 1])[0, 1])
    more["y_corr_t0_t1"] = float(np.corrcoef(Y[:, 0], Y[:, 1])[0, 1])
    # (iii-b) NON-CAUSAL oracle: replace the prediction by its mean over each label run (bounds within-run jitter)
    _orc = np.empty_like(P["student"])
    for _k in range(2):
        _rid = (np.cumsum(chg[:, :, _k], axis=1) + np.arange(G)[:, None] * 20000)[gi, ti]
        _u_, _inv2 = np.unique(_rid, return_inverse=True)
        _sm_ = np.bincount(_inv2, P["student"][:, _k]) / np.bincount(_inv2)
        _orc[:, _k] = _sm_[_inv2]
    more["oracle_runmean_wp"] = float(wp_of(_orc).mean())
    more["oracle_mix_half_wp"] = float(wp_of((_orc + P["student"]) / 2).mean())
    # (v) bank-wide: class-wise best member (stitched) vs best single, exploratory/in-sample
    _bank = [json.loads(_l) for _l in (ROOT / "research/campaign_2026-10-06/bandit_continuation/prediction_bank/BANK_FOLD1.jsonl").read_text().splitlines() if _l.strip()]
    _grade = json.loads((ROOT / "research/campaign_2026-10-06/bandit_continuation/prediction_bank/LATEST_GRADE_FOLD1.json").read_text())["singles"]
    _cand = sorted([_b for _b in _bank if "lead" not in _b["name"] and _b["name"] in _grade and _grade[_b["name"]]["wp"] > 0.6], key=lambda _b: -_grade[_b["name"]]["wp"])
    _cand = _cand[::3][:40]
    _flat, _names = [], []
    for _b in _cand:
        _path = ROOT / _b["prediction"]
        _side = json.loads(_path.with_suffix(".json").read_text())
        _order = {int(_g): _i for _i, _g in enumerate(_side["row_groups"])}
        _sel = np.asarray([_order[int(_g)] for _g in groups])
        _arr = np.load(_path, mmap_mode="r")
        _sorted = np.argsort(_sel)
        _tmp = np.empty((len(groups), _arr.shape[1], 2), np.float32)
        _tmp[_sorted] = np.asarray(_arr[_sel[_sorted]])
        _flat.append(np.clip(_tmp[gi, ti], -2, 2))
        _names.append(_b["name"])
        del _tmp
    _cls_corr = np.zeros((len(_flat), 4))
    for _j, _p in enumerate(_flat):
        for _k in range(4):
            _s = CLS == _k
            _w = W[_s].astype(np.float64)
            _y, _q = Y[_s], _p[_s]
            _wy = _w.sum(0)
            _my, _mp = (_w * _y).sum(0) / _wy, (_w * _q).sum(0) / _wy
            _cls_corr[_j, _k] = ((_w * (_y - _my) * (_q - _mp)).sum(0) / np.sqrt((_w * (_y - _my) ** 2).sum(0) * (_w * (_q - _mp) ** 2).sum(0))).mean()
    _wp_single = np.array([float(wp_of(_p).mean()) for _p in _flat])
    _best_per_class = _cls_corr.argmax(0)
    _stitch = np.zeros_like(_flat[0])
    for _k in range(4):
        _s = CLS == _k
        _stitch[_s] = _flat[_best_per_class[_k]][_s]
    _res_s = Y - P["student"]
    _rho, _gain = [], []
    for _j, _p in enumerate(_flat):
        if _names[_j] == "h230_f1_s3_ep3":
            continue
        _rj = Y - _p
        _rho.append(float(np.mean([np.corrcoef(_res_s[:, _k], _rj[:, _k])[0, 1] for _k in range(2)])))
        _gain.append(float(wp_of((P["student"] + _p) / 2).mean() - (0.6692300927903485 + _wp_single[_j]) / 2))
    more["bank_pair_resid_rho_range"] = [min(_rho), max(_rho)]
    more["bank_pair_gain_vs_rho_corr"] = float(np.corrcoef(_rho, _gain)[0, 1])
    more["bank_pair_gain_range"] = [min(_gain), max(_gain)]
    more["bank_n"] = len(_flat)
    more["bank_best_single_wp"] = float(_wp_single.max())
    more["bank_classwise_best_member"] = [_names[_i] for _i in _best_per_class]
    more["bank_classwise_corr_range"] = [[float(_cls_corr[:, _k].min()), float(_cls_corr[:, _k].max())] for _k in range(4)]
    more["bank_stitched_wp"] = float(wp_of(_stitch).mean())
    more["bank_mean_all_wp"] = float(wp_of(sum(_flat) / len(_flat)).mean())
    more["bank_rank_corr_class3_vs_overall"] = float(np.corrcoef(np.argsort(np.argsort(_cls_corr[:, 3])), np.argsort(np.argsort(_wp_single)))[0, 1])
    return (more,)


@app.cell
def _(CLS, FEAT, P, PLOTS, RP, ROOT, SL, W, Y, gi, lead, np, plt, wp_of):
    ev_ = {}
    _ev = FEAT["since_book"] == 0
    _start = RP.min(1) == 0
    ev_["P_runstart_given_book_event"] = float((_start & _ev).sum() / _ev.sum())
    ev_["P_book_event_given_runstart"] = float((_start & _ev).sum() / _start.sum())
    ev_["book_event_row_frac"] = float(_ev.mean())
    ev_["weight_share_event_rows"] = float(W[_ev].sum() / W.sum())
    # causal expanding mean of the prediction since the last book-event scored row (reset per sequence); valid but conservative
    _newseq = np.r_[True, gi[1:] != gi[:-1]]
    _segstart = _newseq | _ev
    _segid = np.cumsum(_segstart) - 1
    _startidx = np.flatnonzero(_segstart)[_segid]
    _cnt = (np.arange(len(gi)) - _startidx + 1)[:, None]
    _cs = np.cumsum(P["student"].astype(np.float64), axis=0)
    _base = np.where((_startidx > 0)[:, None], _cs[np.maximum(_startidx - 1, 0)], 0.0)
    _cm = ((_cs - _base) / _cnt).astype(np.float32)
    ev_["causal_segmean_wp"] = float(wp_of(_cm).mean())
    ev_["causal_segmean_mix_wp"] = float(wp_of((_cm + P["student"]) / 2).mean())
    # NOTE: a sample-and-hold test was dropped: heur_valid covers only ~12% of grid steps, so events at other steps are invisible.
    # weighted share of residual SS on event vs non-event rows
    _r = W * (Y - P["student"]) ** 2
    ev_["resid_ss_share_event_rows"] = float(_r[_ev].sum() / _r.sum())
    # sum/difference decomposition of the two targets
    _u = lambda a: (a[:, 0] - a[:, 1]) / 2
    _v = lambda a: (a[:, 0] + a[:, 1]) / 2
    _wm = W.mean(1)
    def _corr(a, b):
        ma, mb = (_wm * a).sum() / _wm.sum(), (_wm * b).sum() / _wm.sum()
        return float((_wm * (a - ma) * (b - mb)).sum() / np.sqrt((_wm * (a - ma) ** 2).sum() * (_wm * (b - mb) ** 2).sum()))
    ev_["uv"] = dict(
        var_y_u=float((_wm * _u(Y) ** 2).sum() / _wm.sum()), var_y_v=float((_wm * _v(Y) ** 2).sum() / _wm.sum()),
        var_p_u=float((_wm * _u(P["student"]) ** 2).sum() / _wm.sum()), var_p_v=float((_wm * _v(P["student"]) ** 2).sum() / _wm.sum()),
        corr_u=_corr(_u(Y), _u(P["student"])), corr_v=_corr(_v(Y), _v(P["student"])))
    _shr = {}
    for _lam in (1.5, 1.0, 0.75, 0.5, 0.25, 0.0):
        _pu, _pv = _u(P["student"]), _v(P["student"])
        _p2 = np.stack([_pu + _lam * _pv, -_pu + _lam * _pv], 1).astype(np.float32)
        _shr[str(_lam)] = float(wp_of(_p2).mean())
    ev_["v_shrink_wp"] = _shr
    _grid = {}
    for _au in (0.8, 1.0, 1.25, 1.5, 2.0):
        for _lam in (0.5, 0.75, 1.0):
            _pu, _pv = _u(P["student"]), _v(P["student"])
            _p2 = np.stack([_au * _pu + _lam * _pv, -_au * _pu + _lam * _pv], 1).astype(np.float32)
            _grid[f"u{_au}_v{_lam}"] = float(wp_of(_p2).mean())
    ev_["uv_scale_grid_wp"] = _grid
    # class descriptors from meta_valid
    with np.load(ROOT / "cache/meta_valid.npz") as _m:
        _mc = _m["market_class"]
        ev_["class_meta_means"] = {int(_c): {_k: float(_m[_k][_mc == _c].mean()) for _k in ("t0_std", "t1_std", "t0_zero", "t01_corr", "px_event_rate", "trade_rate", "i1_px_event_rate", "scored_frac")} for _c in range(4)}
    # key figure
    fig4, ax4 = plt.subplots(1, 4, figsize=(17, 3.4))
    _d = SL["class"]
    ax4[0].bar([r["slice"] for r in _d], [r["local_corr"] for r in _d], color="#1f77b4")
    ax4[0].set_title("local weighted corr by market class")
    _d = [r for r in SL["run position"]]
    _o = ["start(0)", "1", "2-3", "4-7", "8-15", "16+"]
    _d = sorted(_d, key=lambda r: _o.index(r["slice"]))
    ax4[1].plot(_o, [r["local_corr"] for r in _d], "o-")
    ax4[1].set_title("local corr vs rows since label change")
    ax4[2].plot(list(lead.keys()), list(lead.values()), "o-")
    ax4[2].set_title("corr(y_t, pred_{t+k})")
    ax4[3].plot(list(ev_["v_shrink_wp"].keys()), list(ev_["v_shrink_wp"].values()), "o-")
    ax4[3].set_title("WP vs shrink of common mode (t0+t1)/2")
    for _a in ax4[2:3]:
        _a.set_xticks(list(lead.keys())[::5])
    fig4.tight_layout()
    fig4.savefig(PLOTS / "key_surprises.png", dpi=110)
    return (ev_,)


@app.cell
def _(HERE, SL, blends, conc, dtab, extra, ev_, more, join_rate, json, lead, res_corr, wp_report):
    (HERE / "results.json").write_text(json.dumps(dict(
        wp=wp_report, join_rate=join_rate, slices=SL, concentration={_k: _v for _k, _v in conc.items()},
        blends=blends, resid_corr=res_corr, disagreement=dtab, leadlag=lead, extra=extra, more=more, events=ev_), indent=1, default=str))
    return


if __name__ == "__main__":
    app.run()
