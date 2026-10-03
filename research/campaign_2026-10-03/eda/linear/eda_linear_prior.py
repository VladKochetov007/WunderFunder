import marimo

__generated_with = "0.25.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pyarrow.parquet as pq
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path
    ROOT = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
    OUT = ROOT / "research/campaign_2026-10-03/eda/linear"
    return OUT, ROOT, np, plt, pq


@app.cell(hide_code=True)
def _(ROOT, pq):

    FEATS = (
        [f"i0_p{i}" for i in range(22)] + [f"i0_v{i}" for i in range(22)]
        + [f"i0_dp{i}" for i in range(4)] + [f"i0_dv{i}" for i in range(4)]
        + [f"i1_p{i}" for i in range(22)] + [f"i1_v{i}" for i in range(22)]
        + [f"i1_dp{i}" for i in range(4)] + [f"i1_dv{i}" for i in range(4)]
        + [f"a{i}" for i in range(8)]
    )
    PV = pq.ParquetFile(ROOT / "datasets/valid.parquet")
    PT = pq.ParquetFile(ROOT / "datasets/train.parquet")

    def load_seq(pf, rg, cols=None):
        """One sequence (row group) -> dict of float32/bool numpy arrays."""
        tb = pf.read_row_group(int(rg), columns=cols)
        return {n: tb.column(n).to_numpy() for n in tb.column_names}


    return FEATS, PV, load_seq


@app.cell(hide_code=True)
def _(FEATS, OUT, PV, load_seq, np, plt):

    def fig_seq(rg, lo=0, hi=20000, fname=None):
        s = load_seq(PV, rg)
        X = np.stack([s[n] for n in FEATS], 1)
        t = np.arange(lo, hi)
        groups = [("t0/t1", None), ("i0 bid p", slice(0, 11)), ("i0 ask p", slice(11, 22)), ("i0 dp", slice(44, 48)),
                  ("i0 dv", slice(48, 52)), ("i1 bid p", slice(52, 63)), ("a0-a3", slice(104, 108)), ("a4-a7", slice(108, 112))]
        fig, axs = plt.subplots(len(groups), 1, figsize=(16, 2.0 * len(groups)), sharex=True)
        for ax, (nm, sl) in zip(axs, groups):
            if sl is None:
                ax.plot(t, s["t0"][lo:hi], lw=0.8, label="t0", color="#1f5fa8")
                ax.plot(t, s["t1"][lo:hi], lw=0.8, label="t1", color="#c0562b")
                sc = s["is_scored"][lo:hi]
                ax.scatter(t[sc], np.full(sc.sum(), ax.get_ylim()[0]), s=2, color="k", label="is_scored")
                ax.legend(loc="upper right", fontsize=7)
            else:
                Z = X[lo:hi, sl]
                for j in range(Z.shape[1]):
                    ax.plot(t, Z[:, j], lw=0.5, label=FEATS[sl][j])
                ax.legend(loc="upper right", fontsize=6, ncol=6)
            ax.set_ylabel(nm, fontsize=8)
        axs[0].set_title(f"valid rg {rg} seq {s['seq_ix'][0]}  rows {lo}-{hi}")
        fig.tight_layout()
        if fname: fig.savefig(OUT / fname, dpi=80)
        return fig

    fig_seq(0, 0, 20000, "fig01_seq0_full.png")
    return (fig_seq,)


@app.cell(hide_code=True)
def _(fig_seq):
    fig_seq(0, 5000, 5400, "fig02_seq0_zoom.png")
    return


@app.cell(hide_code=True)
def _(FEATS, OUT, PV, np):

    import sys as _sys
    _sys.path.insert(0, str(OUT))
    import common as C

    def dt_on_dx(rgs, pf=None):
        """Contemporaneous LS of dt_k(t) on dX(t) (all 112 feature diffs): how much of the target's step-to-step change is
        explained by the *current* feature change? Returns per-target R^2 in-sample (pooled) and out-of-sample (half split)."""
        pf = pf or PV
        G = np.zeros((113, 113)); b = np.zeros((113, 2)); yy = np.zeros(2); rows = []
        for rg in rgs:
            X, Y, _ = C.load(pf, rg)
            dX = np.diff(X, axis=0).astype(np.float64); dY = np.diff(Y, axis=0).astype(np.float64)
            rows.append((dX, dY))
        half = len(rows) // 2
        for dX, dY in rows[:half]:
            A = np.c_[dX, np.ones(len(dX))]
            G += A.T @ A; b += A.T @ dY
        beta = np.linalg.solve(G + 1e-3 * np.eye(113), b)
        r2 = []
        for dX, dY in rows[half:]:
            A = np.c_[dX, np.ones(len(dX))]
            r2.append([np.corrcoef(A @ beta[:, k], dY[:, k])[0, 1] ** 2 for k in range(2)])
        return beta, np.array(r2)

    _rgs = np.random.default_rng(7).choice(PV.metadata.num_row_groups, 24, replace=False)
    dt_beta, dt_r2 = dt_on_dx(_rgs)
    print("OOS R2 of dt0,dt1 on contemporaneous dX (per held-out seq):", dt_r2.round(3).tolist())
    print("mean", dt_r2.mean(0))
    _top = np.argsort(-np.abs(dt_beta[:112, 0]))[:12]
    print("top |beta| for dt0:", [(FEATS[i], round(float(dt_beta[i, 0]), 3)) for i in _top])
    _top = np.argsort(-np.abs(dt_beta[:112, 1]))[:12]
    print("top |beta| for dt1:", [(FEATS[i], round(float(dt_beta[i, 1]), 3)) for i in _top])
    return C, dt_beta


@app.cell(hide_code=True)
def _(C, PV, dt_beta, np):

    def per_seq_dt_fit(rgs, pf=None):
        """Within-sequence LS of dt on dX (first 10k rows fit, last 10k evaluate) + the sequence's target quantum q."""
        pf = pf or PV
        out = []
        for rg in rgs:
            X, Y, _ = C.load(pf, rg)
            dX = np.diff(X, axis=0).astype(np.float64); dY = np.diff(Y, axis=0).astype(np.float64)
            A = np.c_[dX, np.ones(len(dX))]
            h = len(A) // 2
            beta = np.linalg.lstsq(A[:h], dY[:h], rcond=None)[0]
            r2 = [np.corrcoef(A[h:] @ beta[:, k], dY[h:, k])[0, 1] ** 2 for k in range(2)]
            nz = np.abs(Y[Y != 0]); q = nz.min()
            # global-beta R2 for comparison
            r2g = [np.corrcoef(A[h:] @ dt_beta[:, k], dY[h:, k])[0, 1] ** 2 for k in range(2)]
            out.append((rg, q, r2[0], r2[1], r2g[0], r2g[1], (dY[:, 0] != 0).mean(), np.abs(beta[:112, 0]).sum()))
        return np.array(out)

    seq_fit = per_seq_dt_fit(np.random.default_rng(7).choice(PV.metadata.num_row_groups, 24, replace=False))
    print("rg  q  r2_own0 r2_own1 r2_glob0 r2_glob1 frac_dt0!=0 sum|beta0|")
    print(np.array2string(seq_fit, precision=3, suppress_small=True))
    return


@app.cell(hide_code=True)
def _(OUT, np, plt):

    # Fig 3: target ACF / cross-correlation and ACF of target first differences (60 valid sequences)
    _a = np.load(OUT / "target_acf.npz")
    _fig, _ax = plt.subplots(1, 3, figsize=(16, 4))
    _lags = np.arange(len(_a["y0y0"]))
    _ax[0].plot(_lags, _a["y0y0"], label="acf t0", color="#1f5fa8"); _ax[0].plot(_lags, _a["y1y1"], label="acf t1", color="#c0562b")
    _ax[0].plot(_lags, _a["y0y1"], label="corr(t0(t), t1(t+k))", color="#555555"); _ax[0].set_xscale("symlog", linthresh=10)
    _ax[0].axhline(0, color="#999", lw=0.5); _ax[0].legend(fontsize=8); _ax[0].set_title("target auto/cross-correlation"); _ax[0].set_xlabel("lag k")
    _ax[1].plot(_lags[1:121], _a["d0d0"][1:121], label="acf dt0", color="#1f5fa8"); _ax[1].plot(_lags[1:121], _a["d1d1"][1:121], label="acf dt1", color="#c0562b")
    _ax[1].axhline(0, color="#999", lw=0.5); _ax[1].legend(fontsize=8); _ax[1].set_title("ACF of target first differences"); _ax[1].set_xlabel("lag")
    _ax[2].plot(_lags[:121], _a["d0d1"][:121], label="corr(dt0(t), dt1(t+k))", color="#555555")
    _ax[2].axhline(0, color="#999", lw=0.5); _ax[2].legend(fontsize=8); _ax[2].set_title("cross-corr of target differences"); _ax[2].set_xlabel("lag")
    _fig.tight_layout(); _fig.savefig(OUT / "fig03_target_acf.png", dpi=80)
    _fig

    return


@app.cell(hide_code=True)
def _(FEATS, OUT, np, plt):

    # Fig 4: pooled |y|-weighted corr of feature x transform with t0/t1 on scored valid rows (300 seqs),
    # and pooled-vs-within-sequence comparison (between-sequence offsets inflate pooled corr)
    uni = np.load(OUT / "univariate.npz")
    _TR = list(uni["TR"]); _nf = 112
    _fig, _ax = plt.subplots(3, 1, figsize=(18, 10))
    for _k, _key in enumerate(["wc0", "wc1"]):
        _M = uni[_key].reshape(len(_TR), _nf)
        _im = _ax[_k].imshow(_M, aspect="auto", cmap="RdBu_r", vmin=-0.45, vmax=0.45, interpolation="nearest")
        _ax[_k].set_yticks(range(len(_TR))); _ax[_k].set_yticklabels(_TR)
        _ax[_k].set_xticks(range(0, _nf, 2)); _ax[_k].set_xticklabels(FEATS[::2], rotation=90, fontsize=6)
        _ax[_k].set_title(f"pooled weighted corr with t{_k} (scored valid rows)")
        _fig.colorbar(_im, ax=_ax[_k], fraction=0.015)
    _per = uni["per"]
    for _k, _key, _col in [(0, "wc0", "#1f5fa8"), (1, "wc1", "#c0562b")]:
        _med = np.nanmedian(_per[:, _k], 0)
        _ax[2].scatter(_med, uni[_key], s=5, color=_col, label=f"t{_k}", alpha=0.6)
    _ax[2].plot([-0.5, 0.5], [-0.5, 0.5], color="#999", lw=0.5)
    _ax[2].set_xlabel("median within-sequence weighted corr"); _ax[2].set_ylabel("pooled weighted corr"); _ax[2].legend()
    _ax[2].set_title("pooled vs within-sequence correlation, all feature x transform pairs")
    _fig.tight_layout(); _fig.savefig(OUT / "fig04_univariate_heatmap.png", dpi=80)
    _fig

    return (uni,)


@app.cell(hide_code=True)
def _(FEATS, np, uni):

    # Which whole-sequence feature statistics separate the 4 target-quantum classes? (eta^2 = between-class var share)
    _q = np.round(uni["q"], 3); _cl = np.searchsorted(np.unique(_q), _q)
    def _eta2(v):
        m = v.mean(); tot = ((v - m) ** 2).sum()
        return sum(((v[_cl == c].mean() - m) ** 2) * (_cl == c).sum() for c in range(4)) / (tot + 1e-12)
    class_sep = {}
    for _st in ["fmean", "fstd", "fchg"]:
        _M = uni[_st]
        class_sep[_st] = np.array([_eta2(_M[:, j]) for j in range(112)])
        _top = np.argsort(-class_sep[_st])[:8]
        print(_st, [(FEATS[j], round(float(class_sep[_st][j]), 3)) for j in _top])
    for _st, _j in [("fchg", FEATS.index("i0_p0")), ("fchg", FEATS.index("i0_dv0")), ("fstd", FEATS.index("i0_p0"))]:
        print(_st, FEATS[_j], [np.round(np.quantile(uni[_st][_cl == c, _j], [0.1, 0.5, 0.9]), 3).tolist() for c in range(4)])

    return


@app.cell(hide_code=True)
def _(C, PV, np):

    # Decompose targets into k-units: s = (t0+t1)/q ("spread-like", current state?) and d = (t0-t1)/q ("future move").
    # Per class: OOS R^2 of ds, dd (first differences) on contemporaneous dX with class-pooled beta.
    def sd_decomp(nseq=40, seed=5):
        rgs = np.random.default_rng(seed).choice(PV.metadata.num_row_groups, nseq, replace=False)
        by = {c: [] for c in range(4)}
        QS = np.array([0.1254, 0.2413, 0.4685, 0.6569])
        for rg in rgs:
            X, Y, _ = C.load(PV, rg)
            q = np.abs(Y[Y != 0]).min(); c = int(np.argmin(np.abs(QS - q)))
            k = np.round(Y / q)
            s = k[:, 0] + k[:, 1]; d = k[:, 0] - k[:, 1]
            by[c].append((np.diff(X, axis=0).astype(np.float64), np.diff(s), np.diff(d), s, d))
        out = {}
        for c, L in by.items():
            if len(L) < 4: continue
            h = len(L) // 2
            A = np.concatenate([l[0] for l in L[:h]]); A = np.c_[A, np.ones(len(A))]
            res = []
            for j, nm in [(1, "ds"), (2, "dd")]:
                v = np.concatenate([l[j] for l in L[:h]])
                b = np.linalg.lstsq(A, v, rcond=None)[0]
                At = np.concatenate([l[0] for l in L[h:]]); At = np.c_[At, np.ones(len(At))]
                vt = np.concatenate([l[j] for l in L[h:]])
                res.append(round(float(np.corrcoef(At @ b, vt)[0, 1] ** 2), 3))
            vs = np.concatenate([l[3] for l in L]); vd = np.concatenate([l[4] for l in L])
            out[c] = dict(n=len(L), r2_ds=res[0], r2_dd=res[1], var_s=round(float(vs.var()), 2), var_d=round(float(vd.var()), 2),
                          mean_s=round(float(vs.mean()), 2), mean_d=round(float(vd.mean()), 2))
        return out
    sd_res = sd_decomp()
    for _c, _v in sd_res.items(): print("class", _c, _v)

    return


if __name__ == "__main__":
    app.run()
