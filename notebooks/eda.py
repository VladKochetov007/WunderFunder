import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Wunder Connectome: EDA under the global metric

    Metric (current): clipped **global** weighted Pearson, pooled over all scored rows of all
    sequences, weight $|y|$, averaged over $t_0, t_1$. Old local `METRIC.md` averaged per sequence.

    Data in memory: validation targets, scoring mask, market class (1873 x 20000),
    organizer-baseline predictions, per-sequence summary table. Features are read per sequence on demand.
    """)
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import polars as pl
    import matplotlib.pyplot as plt
    import pickle
    from wnn.gmetric import global_wp, wp_sums, wp_from_sums
    from wnn.metric import block_scores
    from wnn.data import SequenceFile, FEATURE_COLUMNS
    import importlib
    import wnn.ticks as ticks
    importlib.reload(ticks)
    from wnn.ticks import reconstruct, learn_moves, learn_spreads, consistent_exits, CLASS3_QUANTUM

    ROOT = mo.notebook_dir().parent
    QUANTA = np.array([0.12543, 0.24133, 0.46854, 0.65687])
    BLUE, ORANGE, AQUA, VIOLET, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#0b0b0b", "#8a8984"
    CLASS_COLORS = [BLUE, ORANGE, AQUA, VIOLET]
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6})
    valid_file = SequenceFile(ROOT / "datasets" / "valid.parquet")
    _d = np.load(ROOT / "cache" / "valid_targets.npz")
    y_valid, mask_valid, cls_valid, rg_valid = _d["y"], _d["mask"], _d["market_class"], _d["row_group"]
    pred_base = np.load(ROOT / "cache" / "pred_baseline_valid.npy")
    seqstats = pl.read_parquet(ROOT / "cache" / "seqstats_valid.parquet").sort("row_group")

    return (
        BLUE,
        CLASS3_QUANTUM,
        CLASS_COLORS,
        FEATURE_COLUMNS,
        INK,
        MUTED,
        ORANGE,
        QUANTA,
        ROOT,
        SequenceFile,
        block_scores,
        cls_valid,
        consistent_exits,
        learn_moves,
        learn_spreads,
        mask_valid,
        mo,
        np,
        pickle,
        pl,
        plt,
        pred_base,
        reconstruct,
        rg_valid,
        seqstats,
        valid_file,
        wp_from_sums,
        wp_sums,
        y_valid,
    )


@app.cell
def _(
    block_scores,
    cls_valid,
    mask_valid,
    mo,
    np,
    pred_base,
    wp_from_sums,
    wp_sums,
    y_valid,
):
    _m = mask_valid
    y_sc = y_valid[_m]
    p_sc = pred_base[_m]
    cls_sc = np.broadcast_to(cls_valid[:, None], _m.shape)[_m]


    def wp_rows(y, p):
        """Global WP on already-selected rows (N, 2)."""
        r = [float(wp_from_sums(wp_sums(y[:, k], p[:, k]))) for k in range(2)]
        return {"t0": round(r[0], 4), "t1": round(r[1], 4), "wp": round(0.5 * (r[0] + r[1]), 4)}


    mo.md(f"## 1. Metric anatomy\n\nScored rows: **{len(y_sc):,}** ({mask_valid.mean():.1%} of rows). "
          f"Baseline global WP **{wp_rows(y_sc, p_sc)['wp']}**; per-sequence WP "
          f"**{np.nanmean(block_scores(y_valid, pred_base, mask_valid).mean(1)):.4f}** (organizer number 0.589595).")
    return cls_sc, p_sc, wp_rows, y_sc


@app.cell
def _(mo):
    scale = mo.ui.slider(0.05, 1.5, step=0.05, value=0.25, label="baseline multiplier", show_value=True)
    scale
    return (scale,)


@app.cell
def _(BLUE, ORANGE, mo, np, p_sc, plt, scale, wp_rows, y_sc):
    _grid = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.25, 1.5])
    _curve = [wp_rows(y_sc, g * p_sc)["wp"] for g in _grid]
    _cur = wp_rows(y_sc, scale.value * p_sc)
    _fig, _ax = plt.subplots(figsize=(6, 3.2))
    _ax.plot(_grid, _curve, color=BLUE, lw=2, marker="o", ms=5)
    _ax.scatter([scale.value], [_cur["wp"]], s=80, color=ORANGE, zorder=3)
    _ax.set_xscale("log"); _ax.set_xlabel("multiplier on baseline predictions"); _ax.set_ylabel("global WP")
    _ax.set_title("Clip at +-2 punishes large outputs", fontsize=10)
    _clip = (np.abs(scale.value * p_sc) > 2).mean(0)
    mo.hstack([_fig, mo.md(f"**scaled x{scale.value:.2f}**: WP {_cur['wp']} (t0 {_cur['t0']}, t1 {_cur['t1']})  \n"
                           f"share of predictions clipped: t0 {_clip[0]:.1%}, t1 {_clip[1]:.1%}")])
    return


@app.cell
def _(mo, np, p_sc, pl, wp_rows, y_sc):
    _S, _D = y_sc.sum(1), y_sc[:, 0] - y_sc[:, 1]
    _Sh, _Dh = p_sc.sum(1), p_sc[:, 0] - p_sc[:, 1]


    def _comb(s, dd):
        return np.stack([(s + dd) / 2, (s - dd) / 2], -1)


    _rows = [
        ("baseline x0.25", 0.25 * p_sc),
        ("oracle S only (D = 0)", _comb(_S, 0 * _D)),
        ("oracle D only (S = 0)", _comb(0 * _S, _D)),
        ("oracle S + baseline D", _comb(_S, _Dh)),
        ("oracle S + 0.5 baseline D", _comb(_S, 0.5 * _Dh)),
        ("baseline S only", _comb(_Sh, 0 * _D)),
        ("baseline D only", _comb(0 * _S, _Dh)),
    ]
    sd_table = pl.DataFrame([{"predictor": n, **wp_rows(y_sc, q.astype(np.float32))} for n, q in _rows])
    mo.vstack([mo.md("### Sum/difference decomposition\n$S = t_0 + t_1$ (level, spread-like), $D = t_0 - t_1$ (direction). "
                     f"Scored var(S) = {_S.var():.3f}, var(D) = {_D.var():.3f}, corr(S, D) = {np.corrcoef(_S, _D)[0,1]:.3f}."), sd_table])
    return


@app.cell
def _(QUANTA, cls_sc, cls_valid, mo, np, p_sc, pl, wp_rows, y_sc):
    _rows = []
    _p = 0.25 * p_sc
    for _k in range(4):
        _s = cls_sc == _k
        _w = np.abs(np.clip(y_sc[_s], -2, 2))
        _rows.append({"class": _k, "quantum": QUANTA[_k], "seqs": int((cls_valid == _k).sum()),
                      "scored share": round(_s.mean(), 3),
                      "weight share t0": round(_w[:, 0].sum() / np.abs(np.clip(y_sc[:, 0], -2, 2)).sum(), 3),
                      "WP within class (x0.25)": wp_rows(y_sc[_s], _p[_s])["wp"],
                      "target mean t0": round(y_sc[_s, 0].mean(), 3), "target mean t1": round(y_sc[_s, 1].mean(), 3),
                      "pred mean t0": round(_p[_s, 0].mean(), 3), "pred mean t1": round(_p[_s, 1].mean(), 3)})
    class_table = pl.DataFrame(_rows)
    mo.vstack([mo.md("### Per class\nLevel is class-driven; baseline level is miscalibrated across classes."), class_table])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Target copula: the L-shape

    Targets are exact integer multiples of a per-class quantum $q$ (1 tick). In tick units the joint
    support is L-shaped: when one target is positive the other sits at or below $-3$ ticks, and
    $t_0 + t_1 \approx -\text{spread}$ near the knee. Reading: $t_0$ behaves like the PnL of a long
    entered at the current ask, $t_1$ like a short entered at the current bid, both floored near a 3-tick stop.
    """)
    return


@app.cell
def _(QUANTA, mo):
    copula_class = mo.ui.dropdown({f"class {k} (q={QUANTA[k]})": k for k in range(4)}, value="class 3 (q=0.65687)", label="market class")
    copula_rows = mo.ui.radio(["all rows", "scored rows"], value="scored rows", label="rows")
    mo.hstack([copula_class, copula_rows], justify="start")
    return copula_class, copula_rows


@app.cell
def _(
    BLUE,
    MUTED,
    ORANGE,
    QUANTA,
    cls_valid,
    copula_class,
    copula_rows,
    mask_valid,
    mo,
    np,
    pl,
    plt,
    y_valid,
):
    _k = copula_class.value
    _s = cls_valid == _k
    _yy = np.rint(y_valid[_s][:, 99:] / QUANTA[_k]).astype(np.int64).reshape(-1, 2)
    if copula_rows.value == "scored rows":
        _yy = _yy[mask_valid[_s][:, 99:].reshape(-1)]
    _h, _, _ = np.histogram2d(_yy[:, 0], _yy[:, 1], bins=[np.arange(-8.5, 9), np.arange(-8.5, 9)])
    _h = _h / _h.sum()
    fig_copula, _ax = plt.subplots(1, 2, figsize=(11, 4.3))
    _im = _ax[0].imshow(np.log10(_h.T + 1e-6), origin="lower", extent=[-8.5, 8.5, -8.5, 8.5], cmap="Blues", vmin=-5, vmax=-0.3)
    _r = np.arange(-8, 9)
    _ax[0].plot(_r, -_r - 1, color=ORANGE, lw=0.8, ls="--")
    _ax[0].axvline(-3, color=MUTED, lw=0.6); _ax[0].axhline(-3, color=MUTED, lw=0.6)
    _ax[0].set_xlabel("t0 / q"); _ax[0].set_ylabel("t1 / q"); _ax[0].grid(False)
    _ax[0].set_title("log10 joint share; dashed t0+t1=-1; grey -3 ticks", fontsize=9)
    fig_copula.colorbar(_im, ax=_ax[0], shrink=0.8)
    _S = _yy.sum(1)
    _vals, _cnt = np.unique(np.clip(_S, -8, 6), return_counts=True)
    _ax[1].bar(_vals, _cnt / _cnt.sum(), color=BLUE, width=0.8)
    _ax[1].set_xlabel("S = t0 + t1 (ticks, clipped to [-8, 6])"); _ax[1].set_ylabel("share")
    _ax[1].set_title("sum distribution", fontsize=9)
    fig_copula.tight_layout()
    _pos0, _pos1 = _yy[:, 0] > 0, _yy[:, 1] > 0
    _rules = pl.DataFrame([{
        "P(t0>0 and t1>0)": round((_pos0 & _pos1).mean(), 4),
        "P(t1<=-3 | t0>0)": round((_yy[_pos0, 1] <= -3).mean(), 4),
        "P(t0<=-3 | t1>0)": round((_yy[_pos1, 0] <= -3).mean(), 4),
        "P(t0 = -3 | t0<=-3)": round((_yy[_yy[:, 0] <= -3, 0] == -3).mean(), 4),
        "min t0": int(_yy[:, 0].min()), "max t0": int(_yy[:, 0].max()),
        "P(S = -1)": round((_S == -1).mean(), 4), "P(S > 0)": round((_S > 0).mean(), 4),
    }])
    mo.vstack([fig_copula, _rules])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Per-sequence moments: two-state arcs

    A feature that spends share $f$ of a sequence at value $a$ and the rest at $b$ has mean
    $\mu = fa + (1-f)b$ and variance $(\mu-a)(b-\mu)$, an arc in the mean-std plane.
    Dashed arc: pooled top-2 values. Facets are classes.
    """)
    return


@app.cell
def _(FEATURE_COLUMNS, mo):
    _cols = [c for c in FEATURE_COLUMNS]
    arc_col = mo.ui.dropdown(_cols, value="a7", label="feature")
    arc_col
    return (arc_col,)


@app.cell
def _(CLASS_COLORS, INK, arc_col, mo, np, pl, plt, seqstats):
    _c = arc_col.value
    _mu, _sd = seqstats[f"{_c}_mean"].to_numpy(), seqstats[f"{_c}_std"].to_numpy()
    _a, _b = seqstats[f"{_c}_a"].to_numpy(), seqstats[f"{_c}_b"].to_numpy()
    _fa = seqstats[f"{_c}_fa"].to_numpy()
    _cls = seqstats["cls"].to_numpy()
    _vals, _cnt = np.unique(np.r_[_a, _b], return_counts=True)
    _top = _vals[np.argsort(-_cnt)[:2]]
    _lo, _hi = _top.min(), _top.max()
    _g = np.linspace(_lo, _hi, 200)
    fig_arc, _ax = plt.subplots(1, 4, figsize=(14, 3.2), sharex=True, sharey=True)
    for _k in range(4):
        _s = _cls == _k
        _ax[_k].scatter(_mu[_s], _sd[_s], s=5, lw=0, alpha=0.6, color=CLASS_COLORS[_k])
        _ax[_k].plot(_g, np.sqrt(np.maximum((_g - _lo) * (_hi - _g), 0)), color=INK, lw=1, ls="--")
        _ax[_k].set_title(f"{_c} class {_k}", fontsize=9)
        _ax[_k].set_xlabel("sequence mean")
    _ax[0].set_ylabel("sequence std")
    fig_arc.tight_layout()
    _stats = pl.DataFrame({"class": list(range(4)),
                           "median top value share": [round(float(np.median(_fa[_cls == k])), 3) for k in range(4)],
                           "median distinct values": [int(np.median(seqstats[f"{_c}_nuniq"].to_numpy()[_cls == k])) for k in range(4)],
                           "median change rate": [round(float(np.median(seqstats[f"{_c}_chg"].to_numpy()[_cls == k])), 3) for k in range(4)]})
    mo.vstack([fig_arc, mo.md(f"pooled top-2 values: {_top.round(4).tolist()}"), _stats])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Class 3: exact tick paths and label inversion

    Book-side codes are global and deterministic, so each code transition means a fixed quote move.
    The move table is learned on **train** class-3 sequences (dominant target jump per transition),
    then applied to validation from features alone. Inversion: for each event row, which future rows
    $j$ reproduce **both** targets if the long (entry at ask) and short (entry at bid) exit together at $j$,
    each with a 3-tick stop?
    """)
    return


@app.cell
def _(
    CLASS3_QUANTUM,
    ROOT,
    SequenceFile,
    learn_moves,
    learn_spreads,
    mo,
    np,
    pickle,
):
    _path = ROOT / "cache" / "class3_moves_full.pkl"
    if _path.exists():
        moves3, spreads3 = pickle.load(open(_path, "rb"))
    else:
        _train = SequenceFile(ROOT / "datasets" / "train.parquet")
        _seqs = []
        for _g in range(0, 600):
            _sq = _train.read(_g)
            if np.abs(np.rint(_sq.targets / CLASS3_QUANTUM) * CLASS3_QUANTUM - _sq.targets).max() < 1e-3:
                _seqs.append((_sq.features, _sq.targets))
        moves3, spreads3 = learn_moves(_seqs), learn_spreads(_seqs)
        pickle.dump((moves3, spreads3), open(_path, "wb"))
    mo.md(f"learned on train class-3 sequences: **{len(moves3)}** code transitions, **{len(spreads3)}** code spreads")
    return moves3, spreads3


@app.cell
def _(
    CLASS3_QUANTUM,
    cls_valid,
    mo,
    moves3,
    np,
    pl,
    reconstruct,
    rg_valid,
    spreads3,
    valid_file,
):
    rgs3 = np.flatnonzero(cls_valid == 3)[:120]
    recon3 = {}
    _rows = []
    for _i in rgs3:
        _sq = valid_file.read(int(rg_valid[_i]))
        _a, _b, _unk = reconstruct(_sq.features, moves3, spreads3)
        _q = np.rint(_sq.targets / CLASS3_QUANTUM).astype(np.int64)
        recon3[int(_i)] = (_a, _b, _q, _sq.features, _sq.scored)
        _ev = np.flatnonzero((np.diff(_a) != 0) | (np.diff(_b) != 0)) + 1
        _dq = _q[_ev] - _q[_ev - 1]
        _rows.append({"seq": int(_i), "unknown": _unk, "spread<1": int(((_a - _b) < 1).sum()),
                      "spread>=3": int(((_a - _b) >= 3).sum()),
                      "dt0==-dA": float((_dq[:, 0] == -np.diff(_a)[_ev - 1]).mean()),
                      "dt1==+dB": float((_dq[:, 1] == np.diff(_b)[_ev - 1]).mean())})
    recon_table = pl.DataFrame(_rows)
    mo.vstack([mo.md(f"sequences with unknown transitions: {(recon_table['unknown'] > 0).sum()} / {len(recon_table)}; "
                     f"rows with impossible spread<1: {recon_table['spread<1'].sum()}; "
                     f"mean event exactness t0 {recon_table['dt0==-dA'].mean():.3f}, t1 {recon_table['dt1==+dB'].mean():.3f}"),
               recon_table.describe()])
    return (recon3,)


@app.cell
def _(BLUE, ORANGE, consistent_exits, mo, np, pl, plt, recon3):
    inv_rows = []
    for _i, (_a, _b, _q, _x, _m) in recon3.items():
        _ev = np.flatnonzero((np.diff(_a) != 0) | (np.diff(_b) != 0)) + 1
        _ev = _ev[(_ev >= 99) & (_ev < 19000)]
        _f, _l, _n = consistent_exits(_a, _b, _q[_ev, 0], _q[_ev, 1], _ev, 6000, 3)
        inv_rows.append(pl.DataFrame({"seq": np.full(len(_ev), _i), "row": _ev, "t0": _q[_ev, 0], "t1": _q[_ev, 1],
                                      "first": _f, "last": _l, "n": _n, "scored": _m[_ev]}))
    inversion = pl.concat(inv_rows).with_columns(lag_first=pl.col("first") - pl.col("row"), lag_last=pl.col("last") - pl.col("row"))
    _ok = inversion.filter(pl.col("first") >= 0)
    fig_inv, _ax = plt.subplots(1, 2, figsize=(12, 3.4))
    _ax[0].hist(np.log10(_ok["lag_first"].to_numpy()), bins=80, color=BLUE, alpha=0.8, label="first consistent exit")
    _ax[0].hist(np.log10(_ok["lag_last"].to_numpy()), bins=80, color=ORANGE, alpha=0.5, label="last consistent exit")
    _ax[0].set_xlabel("log10 rows after entry"); _ax[0].legend(frameon=False)
    _ax[1].hist(np.log10(1 + _ok["n"].to_numpy()), bins=60, color=BLUE)
    _ax[1].set_xlabel("log10(1 + number of consistent exit rows)")
    fig_inv.tight_layout()
    mo.vstack([mo.md(f"event rows: {len(inversion):,}; with at least one consistent joint exit within 6000 rows: "
                     f"**{(inversion['first'] >= 0).mean():.3f}**; stopped both (t0=t1=-3) excluded? no."), fig_inv,
               _ok.select(pl.col("lag_first").quantile(q).alias(f"first q{q}") for q in (0.1, 0.5, 0.9))])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. The target law: one integer drives both targets

    Class-3 inversion showed $t_0 + t_1 = -\text{spread}$ exactly whenever no side is at the loss value,
    and the losing side is always exactly $-(\text{sp}+2)$ ticks. With $U = F - \text{bid}$ for a common
    future price $F$ and current spread $\text{sp}$ (ticks):

    | region | $t_0$ | $t_1$ |
    |---|---|---|
    | $U < 0$ (below bid) | $-(\text{sp}+2)$ | $-U$ |
    | $0 \le U \le \text{sp}$ | $U - \text{sp}$ | $-U$ |
    | $U > \text{sp}$ (above ask) | $U - \text{sp}$ | $-(\text{sp}+2)$ |

    The map is invertible, so $(\text{sp}, U)$ follow from targets alone. Test on every class: the
    implied spread must be positive and change only on rows where the i0 book changes.
    """)
    return


@app.cell
def _(QUANTA, cls_valid, mo, np, pl, rg_valid, valid_file):
    def invert_targets(q):
        """(T, 2) targets in ticks -> implied spread and U (ticks)."""
        t0, t1 = q[:, 0], q[:, 1]
        up, dn = t0 >= 1, t1 >= 1
        sp = np.where(up, -t1 - 2, np.where(dn, -t0 - 2, -(t0 + t1)))
        u = np.where(up, t0 + sp, -t1)
        return sp, u


    _rows = []
    for _k in range(4):
        for _i in np.flatnonzero(cls_valid == _k)[:40]:
            _sq = valid_file.read(int(rg_valid[_i]))
            _q = np.rint(_sq.targets / QUANTA[_k]).astype(np.int64)
            _sp, _u = invert_targets(_q)
            _book = np.r_[True, (_sq.features[1:, 0:22] != _sq.features[:-1, 0:22]).any(1)]
            _chg = np.r_[False, np.diff(_sp) != 0]
            _rows.append({"class": _k, "seq": int(_i), "sp<1": float((_sp[99:] < 1).mean()),
                          "sp changes": int(_chg[99:].sum()),
                          "sp changes off book events": int((_chg & ~_book)[99:].sum()),
                          "median sp": float(np.median(_sp[99:])), "max sp": int(_sp[99:].max()),
                          "U range": f"{int(np.percentile(_u[99:], 1))}..{int(np.percentile(_u[99:], 99))}"})
    law_check = pl.DataFrame(_rows)
    law_summary = law_check.group_by("class").agg(
        pl.col("sp<1").mean().alias("share rows sp<1"),
        (pl.col("sp changes off book events").sum() / pl.col("sp changes").sum()).alias("share sp changes off book events"),
        pl.col("median sp").median(), pl.col("max sp").max()).sort("class")
    mo.vstack([law_summary, law_check.head(12)])
    return (invert_targets,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. What the structure is worth

    Spread is a function of the current book, so a decoder can know it causally (assumed exact here;
    decodability is tested separately). Tables are fit on validation half A (even sequence index) and
    scored on half B, global WP on scored rows. Predictors:
    the tilted-mean table $E_Q[t \mid \text{class}, \text{sp}]$ (no direction information), and that table
    plus a per-class multiple of the baseline prediction.
    """)
    return


@app.cell
def _(
    QUANTA,
    cls_sc,
    cls_valid,
    invert_targets,
    mask_valid,
    np,
    p_sc,
    pl,
    wp_rows,
    y_sc,
    y_valid,
):
    _q_all = np.rint(y_valid / QUANTA[cls_valid][:, None, None]).astype(np.int64)
    sp_all = np.empty(y_valid.shape[:2], np.int64)
    u_all = np.empty(y_valid.shape[:2], np.int64)
    for _i in range(len(y_valid)):
        sp_all[_i], u_all[_i] = invert_targets(_q_all[_i])
    sp_sc = np.minimum(sp_all[mask_valid], 12)
    half_b = (np.arange(len(y_valid)) % 2 == 1)
    half_sc = np.broadcast_to(half_b[:, None], mask_valid.shape)[mask_valid]
    _yc = np.clip(y_sc, -2, 2).astype(np.float64)
    table = {}
    for _k in range(4):
        for _s in np.unique(sp_sc[cls_sc == _k]):
            _sel = (cls_sc == _k) & (sp_sc == _s) & ~half_sc
            if _sel.sum() < 50:
                continue
            _w = np.abs(_yc[_sel])
            table[(_k, _s)] = (_w * _yc[_sel]).sum(0) / np.maximum(_w.sum(0), 1e-9)
    prior_sc = np.zeros_like(y_sc)
    for (_k, _s), _v in table.items():
        prior_sc[(cls_sc == _k) & (sp_sc == _s)] = _v
    _bs = 0.25 * p_sc
    _res = [{"predictor": "baseline x0.25", **wp_rows(y_sc[half_sc], _bs[half_sc])},
            {"predictor": "table E_Q[t|class, sp]", **wp_rows(y_sc[half_sc], prior_sc[half_sc])}]
    _best = None
    for _c in [0.1, 0.2, 0.3, 0.5, 0.75, 1.0]:
        _p = prior_sc + _c * _bs
        _r = wp_rows(y_sc[~half_sc], _p[~half_sc])["wp"]
        if _best is None or _r > _best[0]:
            _best = (_r, _c)
    _p = prior_sc + _best[1] * _bs
    _res.append({"predictor": f"table + {_best[1]} x scaled baseline (c fit on A)", **wp_rows(y_sc[half_sc], _p[half_sc])})
    structure_table = pl.DataFrame(_res)
    structure_table
    return half_sc, prior_sc, sp_all, sp_sc, u_all


@app.cell
def _(cls_sc, half_sc, np, p_sc, pl, prior_sc, wp_rows, y_sc):
    def wls_stack(cols_fit, y_fit, cols_eval):
        """Per-target |y|-weighted least squares on design columns; returns eval predictions."""
        out = np.zeros((len(cols_eval[0]), 2))
        for _k in range(2):
            _yk = np.clip(y_fit[:, _k], -2, 2).astype(np.float64)
            _X = np.column_stack([np.ones(len(_yk))] + [c[:, _k] if c.ndim == 2 else c for c in cols_fit])
            _w = np.abs(_yk)
            _beta = np.linalg.lstsq(_X * np.sqrt(_w)[:, None], _yk * np.sqrt(_w), rcond=None)[0]
            _Xe = np.column_stack([np.ones(len(out))] + [c[:, _k] if c.ndim == 2 else c for c in cols_eval])
            out[:, _k] = _Xe @ _beta
        return out


    _A, _B = ~half_sc, half_sc
    _onehot = lambda sel: [(cls_sc[sel] == _k).astype(float) for _k in range(4)]
    _designs = {
        "baseline only (WLS)": lambda s: [p_sc[s]],
        "baseline + class dummies": lambda s: [p_sc[s]] + _onehot(s)[1:],
        "baseline + table prior": lambda s: [p_sc[s], prior_sc[s]],
        "baseline x class + table prior": lambda s: [p_sc[s] * (cls_sc[s] == _k)[:, None] for _k in range(4)] + [prior_sc[s]],
    }
    stack_table = pl.DataFrame([{"design": n, **wp_rows(y_sc[_B], wls_stack(f(_A), y_sc[_A], f(_B)).astype(np.float32))}
                                for n, f in _designs.items()])
    stack_table
    return (stack_table,)


@app.cell
def _(
    cls_sc,
    half_sc,
    mask_valid,
    mo,
    np,
    pl,
    sp_all,
    sp_sc,
    u_all,
    wp_rows,
    y_sc,
):
    u_sc = u_all[mask_valid]
    branch_sc = np.where(u_sc < 0, 0, np.where(u_sc > sp_all[mask_valid], 2, 1))


    def cell_table(keys, fit):
        """Tilted-mean prediction per discrete key, fit on rows `fit`, applied everywhere."""
        _pred = np.zeros_like(y_sc, dtype=np.float64)
        _yc = np.clip(y_sc, -2, 2).astype(np.float64)
        _uk, _inv = np.unique(keys, axis=0, return_inverse=True)
        _inv = _inv.ravel()
        for _j in range(len(_uk)):
            _sel = _inv == _j
            _f = _sel & fit
            if _f.sum() == 0:
                continue
            _w = np.abs(_yc[_f])
            _pred[_sel] = (_w * _yc[_f]).sum(0) / np.maximum(_w.sum(0), 1e-9)
        return _pred


    _A, _B = ~half_sc, half_sc
    _branch_share = np.bincount(branch_sc[_B], minlength=3) / _B.sum()
    oracle_table = pl.DataFrame([
        {"oracle": "class, sp", **wp_rows(y_sc[_B], cell_table(np.c_[cls_sc, sp_sc], _A)[_B])},
        {"oracle": "class, sp, branch", **wp_rows(y_sc[_B], cell_table(np.c_[cls_sc, sp_sc, branch_sc], _A)[_B])},
        {"oracle": "class, sp, sign(U - sp/2)", **wp_rows(y_sc[_B], cell_table(np.c_[cls_sc, sp_sc, (2 * u_sc > sp_sc)], _A)[_B])},
        {"oracle": "class, sp, U clipped [-4, sp+4]", **wp_rows(y_sc[_B], cell_table(np.c_[cls_sc, sp_sc, np.clip(u_sc, -4, sp_sc + 4)], _A)[_B])},
    ])
    mo.vstack([mo.md(f"branch shares on scored rows (below / inside / above): {np.round(_branch_share, 3).tolist()}"), oracle_table])
    return (branch_sc,)


@app.cell
def _(
    branch_sc,
    cls_sc,
    half_sc,
    mo,
    np,
    p_sc,
    pl,
    sp_sc,
    stack_table,
    wp_rows,
    y_sc,
):
    from sklearn.linear_model import LogisticRegression


    def branch_design(sel, sp=None):
        sp = sp_sc if sp is None else sp
        _p = p_sc[sel].astype(np.float64)
        _oh = np.stack([(cls_sc[sel] == _k) for _k in range(4)], 1).astype(float)
        _spf = np.log(sp[sel].astype(float))
        _cols = [_p, _p[:, :1] * _p[:, 1:], np.tanh(_p / 3), _oh[:, 1:], _spf[:, None]]
        _cols += [_p * _oh[:, [_k]] for _k in range(4)] + [_p * _spf[:, None]]
        return np.column_stack(_cols)


    def moment_tables(fit, sp=None):
        """M1, M0 per (class, sp, branch) from rows `fit`: E[|t| t], E[|t|] for both targets."""
        sp = sp_sc if sp is None else sp
        _yc = np.clip(y_sc, -2, 2).astype(np.float64)
        _m1, _m0 = {}, {}
        for _key in set(zip(cls_sc[fit].tolist(), sp[fit].tolist())):
            for _b in range(3):
                _s = fit & (cls_sc == _key[0]) & (sp == _key[1]) & (branch_sc == _b)
                if _s.sum() == 0:
                    continue
                _w = np.abs(_yc[_s])
                _m1[(*_key, _b)] = (_w * _yc[_s]).mean(0)
                _m0[(*_key, _b)] = _w.mean(0)
        return _m1, _m0


    def rational_readout(probs, sel, m1, m0, sp=None):
        sp = sp_sc if sp is None else sp
        _num = np.zeros((sel.sum(), 2))
        _den = np.full((sel.sum(), 2), 1e-9)
        _c, _s = cls_sc[sel], sp[sel]
        for (_k, _sp, _b), _v in m1.items():
            _r = (_c == _k) & (_s == _sp)
            _num[_r] += probs[_r, _b:_b + 1] * _v
            _den[_r] += probs[_r, _b:_b + 1] * m0[(_k, _sp, _b)]
        return _num / _den


    _A, _B = ~half_sc, half_sc
    _rng = np.random.default_rng(0)
    _fit_idx = np.flatnonzero(_A)
    _fit_idx = _rng.choice(_fit_idx, min(len(_fit_idx), 600_000), replace=False)
    _fit = np.zeros_like(_A); _fit[_fit_idx] = True
    branch_clf = LogisticRegression(max_iter=300, C=1.0).fit(branch_design(_fit), branch_sc[_fit])
    _probs_B = branch_clf.predict_proba(branch_design(_B))
    _m1, _m0 = moment_tables(_A)
    readout_B = rational_readout(_probs_B, _B, _m1, _m0)
    from sklearn.metrics import roc_auc_score
    _auc_up = roc_auc_score(branch_sc[_B] == 2, _probs_B[:, 2])
    _auc_dn = roc_auc_score(branch_sc[_B] == 0, _probs_B[:, 0])
    readout_table = pl.DataFrame([{"predictor": "baseline x class + table prior (linear WLS)", **stack_table.row(3, named=True) | {}},
                                  {"predictor": "logit branch probs -> rational readout", **wp_rows(y_sc[_B], readout_B.astype(np.float32))}]).drop("design", strict=False)
    mo.vstack([mo.md(f"branch AUC on half B from baseline outputs: above-ask {_auc_up:.3f}, below-bid {_auc_dn:.3f}"), readout_table])
    return LogisticRegression, branch_design, moment_tables, rational_readout


@app.cell
def _(
    QUANTA,
    ROOT,
    SequenceFile,
    cls_valid,
    invert_targets,
    mask_valid,
    mo,
    np,
    pl,
    rg_valid,
    sp_all,
    valid_file,
    y_valid,
):
    from sklearn.ensemble import HistGradientBoostingClassifier

    _train = SequenceFile(ROOT / "datasets" / "train.parquet")
    _X, _Y, _C = [], [], []
    _rng = np.random.default_rng(1)
    for _g in _rng.choice(3000, 160, replace=False):
        _sq = _train.read(int(_g))
        _q = _sq.targets / QUANTA[None, :, None].repeat(1, 0)[0] if False else None
        _k = int(np.argmin([np.abs(np.rint(_sq.targets / qq) * qq - _sq.targets).max() for qq in QUANTA]))
        _sp, _ = invert_targets(np.rint(_sq.targets / QUANTA[_k]).astype(np.int64))
        _r = _rng.choice(np.arange(99, 20000), 1500, replace=False)
        _X.append(_sq.features[_r, 0:22]); _Y.append(np.minimum(_sp[_r], 8)); _C.append(np.full(len(_r), _k))
    _Xtr, _Ytr, _Ctr = np.concatenate(_X), np.concatenate(_Y), np.concatenate(_C)
    sp_decoder = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.2).fit(np.c_[_Xtr, _Ctr], _Ytr)
    _Xv, _Yv, _Cv = [], [], []
    for _i in np.random.default_rng(2).choice(len(y_valid), 120, replace=False):
        _sq = valid_file.read(int(rg_valid[_i]))
        _r = np.flatnonzero(mask_valid[_i])
        _Xv.append(_sq.features[_r, 0:22]); _Yv.append(np.minimum(sp_all[_i, _r], 8)); _Cv.append(np.full(len(_r), cls_valid[_i]))
    _Xv, _Yv, _Cv = np.concatenate(_Xv), np.concatenate(_Yv), np.concatenate(_Cv)
    _hat = sp_decoder.predict(np.c_[_Xv, _Cv])
    sp_decode_table = pl.DataFrame([{"class": k, "rows": int((_Cv == k).sum()), "exact accuracy": round(float((_hat == _Yv)[_Cv == k].mean()), 4),
                                     "majority baseline": round(float(np.bincount(_Yv[_Cv == k]).max() / (_Cv == k).sum()), 4)} for k in range(4)])
    mo.vstack([mo.md("### Is spread decodable from the current book?\nGBM on the 22 i0 price columns + class, fit on 160 train sequences, scored on scored rows of 120 validation sequences (spread capped at 8)."), sp_decode_table])
    return (sp_decoder,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Modeling plan implied by the law

    **Output head: distribution over U, not two regressions.** Bins relative to the spread:
    below the bid ($-U \in \{1,\dots,5^+\}$), inside ($U \in \{0,\dots,\text{sp}\}$, as position), above the ask
    ($U-\text{sp} \in \{1,\dots,5^+\}$). Targets map to bins exactly through the inverse law.

    **Loss.** Cross-entropy on U-bins, sample weight = scoring propensity$^\alpha$ (estimates $P(u\mid x)$
    under the scored-row distribution). No $|y|$ weight in the loss: the metric tilt is applied at readout.
    Optional small wMSE on the readout. Huber is dropped: it pulls toward the median while the metric
    weights the tails.

    **Readout (exact, analytic).** $g_k(u,\text{sp})$ from the law, times the class quantum, clipped to $\pm2$:
    $p_k = \sum_u P(u\mid x)\,|g_k|\,g_k \big/ \sum_u P(u\mid x)\,|g_k|$. Bounded by construction, so no clip loss.

    **Features to add.** Decoded spread in ticks (exact lookup for classes 2-3, learned for 0-1); class posterior
    online; class-3 reconstructed tick path (distance to recent high/low, run length, last move direction,
    up/down counts in windows, time since last tick move); a-column clocks as real-time proxies.

    **Open lever.** If $F$'s horizon is found (H-045), past $U$ become observable online, giving each sequence
    its own supervised signal (level, volatility, momentum sign).

    **Next experiment (H-046).** Same small GRU on 2k train sequences, losses: old corr mix, wMSE, Huber,
    CE-U + readout. Score global WP on validation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. What is F? Horizon probe (H-045)

    $F(s) = \text{bid}(s) + U(s)$ is piecewise constant and changes only at book events. When $F$ switches
    at event $s$ from $F^-$ to $F^+$, the horizon end has just crossed a price change, so the quote reaches
    $F^+$ at some later row $c$. The lag $c - s$ is the realized horizon in rows. A fixed row horizon gives a
    constant lag; a fixed real-time horizon gives lags that stretch in quiet periods and shrink in bursts.
    """)
    return


@app.cell
def _(BLUE, ORANGE, invert_targets, mo, np, pl, plt, recon3):
    _rows = []
    for _i, (_a, _b, _q, _x, _m) in recon3.items():
        _sp, _u = invert_targets(_q)
        _F = _b + _u
        _ch = np.flatnonzero(np.diff(_F) != 0) + 1
        _ch = _ch[(_ch > 99) & (_ch < 19000)]
        _book_ev = np.r_[False, (np.diff(_a) != 0) | (np.diff(_b) != 0)]
        _evc = np.cumsum(_book_ev)
        for _s in _ch:
            _f = _F[_s]
            _j = np.flatnonzero(((_b[_s:] == _f) | (_a[_s:] == _f)))
            _j = _j[_j > 0]
            _hit = -1
            if len(_j):
                # first row after s where the touch moves onto F (entering, not already there)
                for _jj in _j:
                    _r = _s + _jj
                    if not ((_b[_r - 1] == _f) or (_a[_r - 1] == _f)):
                        _hit = _r
                        break
            _rows.append({"seq": _i, "row": int(_s), "dF": int(_F[_s] - _F[_s - 1]), "hit": _hit,
                          "lag_rows": _hit - _s if _hit > 0 else -1,
                          "lag_events": int(_evc[_hit] - _evc[_s]) if _hit > 0 else -1,
                          "a2": float(_x[_s, 106]), "a3": float(_x[_s, 107]), "a5": float(_x[_s, 109]),
                          "a6": float(_x[_s, 110]), "a7": float(_x[_s, 111])})
    fchange = pl.DataFrame(_rows)
    _ok = fchange.filter(pl.col("lag_rows") > 0)
    fig_lag, _ax = plt.subplots(1, 3, figsize=(14, 3.4))
    _ax[0].hist(np.log10(_ok["lag_rows"].to_numpy()), bins=60, color=BLUE)
    _ax[0].set_xlabel("log10 lag in rows (F switch -> quote reaches F)")
    _ax[1].hist(np.log10(1 + _ok["lag_events"].to_numpy()), bins=60, color=ORANGE)
    _ax[1].set_xlabel("log10(1 + lag in book events)")
    _ax[2].scatter(_ok["a3"].to_numpy(), np.log10(_ok["lag_rows"].to_numpy()), s=3, lw=0, alpha=0.4, color=BLUE)
    _ax[2].set_xlabel("a3 at switch"); _ax[2].set_ylabel("log10 lag rows")
    fig_lag.tight_layout()
    mo.vstack([mo.md(f"F switches: {len(fchange):,}; reached later: {(fchange['lag_rows'] > 0).mean():.3f}"), fig_lag,
               _ok.select([pl.col("lag_rows").quantile(q).alias(f"rows q{q}") for q in (0.1, 0.25, 0.5, 0.75, 0.9)]
                          + [pl.col("lag_events").quantile(q).alias(f"events q{q}") for q in (0.1, 0.5, 0.9)])])
    return


@app.cell
def _(
    LogisticRegression,
    branch_design,
    branch_sc,
    cls_valid,
    half_sc,
    mask_valid,
    mo,
    moment_tables,
    np,
    pl,
    rational_readout,
    rg_valid,
    sp_decoder,
    sp_sc,
    valid_file,
    wp_rows,
    y_sc,
    y_valid,
):
    # decoded spread on every scored row (single-row GBM fit on train), same readout pipeline keyed by decoded spread
    _parts = []
    for _i in range(len(y_valid)):
        _r = np.flatnonzero(mask_valid[_i])
        if len(_r) == 0:
            continue
        _x = valid_file.read(int(rg_valid[_i]), targets=False, mask=False).features[_r, 0:22]
        _parts.append(np.c_[_x, np.full(len(_r), cls_valid[_i])])
    sp_hat_sc = sp_decoder.predict(np.concatenate(_parts)).astype(np.int64)
    _A, _B = ~half_sc, half_sc


    def _pipeline(sp_key):
        _fit = np.zeros_like(_A)
        _fit[np.random.default_rng(0).choice(np.flatnonzero(_A), 600_000, replace=False)] = True
        _clf = LogisticRegression(max_iter=300).fit(branch_design(_fit, sp_key), branch_sc[_fit])
        _m1, _m0 = moment_tables(_A, sp_key)
        return wp_rows(y_sc[_B], rational_readout(_clf.predict_proba(branch_design(_B, sp_key)), _B, _m1, _m0, sp_key).astype(np.float32))


    decoded_readout_table = pl.DataFrame([
        {"spread used": "oracle (from targets)", **_pipeline(np.minimum(sp_sc, 12))},
        {"spread used": "decoded (single-row GBM)", **_pipeline(np.minimum(sp_hat_sc, 12))},
        {"spread used": "none (all = 1)", **_pipeline(np.ones_like(sp_sc))},
    ])
    mo.vstack([mo.md(f"decoded spread exact accuracy on scored rows: {(np.minimum(sp_hat_sc, 8) == np.minimum(sp_sc, 8)).mean():.3f}"), decoded_readout_table])
    return


if __name__ == "__main__":
    app.run()
