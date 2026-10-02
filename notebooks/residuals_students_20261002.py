import marimo

__generated_with = "0.16.0"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    import sys
    from pathlib import Path

    import marimo as mo
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
    sys.path.insert(0, str(ROOT / "src"))
    from wnn.splits import load_campaign_split

    PLOTS = ROOT / "research" / "campaign_2026-10-02" / "plots"
    PLOTS.mkdir(parents=True, exist_ok=True)
    CLASS_COLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    return CLASS_COLORS, PLOTS, ROOT, json, load_campaign_split, mo, np, plt


@app.cell
def _(ROOT, load_campaign_split, np):
    def _cell():
        # Out-of-fold panel: evaluation sequences of every fold where both students exist (fold 0, plus fold 1 when ready).
        vt = np.load(ROOT / "cache" / "valid_targets.npz", mmap_mode="r")
        arms = {
            "GRU256": ("c20261001_d256_s3_f{k}", "c20261001_ft_d256_s3_f{k}h"),
            "LSTM192": ("c20261002_dlstm192a05_s3_f{k}", "c20261002_ft_dlstm192a05_s3_f{k}h"),
            "GRU256_a07": ("c20261001_d256a07_s3_f{k}", "c20261001_ft_d256a07_s3_f{k}h"),
        }
        folds = [k for k in range(4) if all((ROOT / "cache" / f"pred_{n.format(k=k)}_valid.npy").exists() for pair in arms.values() for n in pair)]
        rows, fold_of = [], []
        for k in folds:
            split = load_campaign_split(ROOT / f"research/campaign_2026-09-29/artifacts/splits/fold{k}.json", ROOT)
            ix = np.flatnonzero(np.isin(np.asarray(vt["row_group"]), np.array(sorted(split.evaluation_groups))))
            rows.append(ix); fold_of.append(np.full(len(ix), k))
        IDX = np.concatenate(rows); FOLD = np.concatenate(fold_of)
        Y = np.clip(np.asarray(vt["y"][IDX], dtype=np.float32), -2, 2)
        M = np.asarray(vt["mask"][IDX]); CLS = np.asarray(vt["market_class"][IDX])
        P = {}
        for name, (par, ft) in arms.items():
            acc = np.zeros(Y.shape, np.float32)
            for k in folds:
                sel = FOLD == k
                for n in (par, ft):
                    acc[sel] += 0.5 * np.asarray(np.load(ROOT / "cache" / f"pred_{n.format(k=k)}_valid.npy", mmap_mode="r")[IDX[sel]], dtype=np.float32)
            P[name] = np.clip(acc, -2, 2)
        P["PAIR_GL"] = 0.5 * (P["GRU256"] + P["LSTM192"])
        P["PAIR_GG"] = 0.5 * (P["GRU256"] + P["GRU256_a07"])
        print("folds", folds, "sequences", len(IDX))
        return CLS, FOLD, IDX, M, P, Y, folds
    CLS, FOLD, IDX, M, P, Y, folds = _cell()
    return CLS, FOLD, IDX, M, P, Y, folds


@app.cell
def _(CLS, M, P, Y, np):
    def _cell():
        def wp(p, sel=None):
            """Global weighted Pearson (|y| weights, scored rows, mean of t0/t1) on a boolean sequence/row selection."""
            m = M if sel is None else (M & sel)
            r = []
            for t in range(2):
                y, q = Y[..., t][m].astype(np.float64), p[..., t][m].astype(np.float64)
                w = np.abs(y); sw = w.sum(); my, mq = (w * y).sum() / sw, (w * q).sum() / sw
                r.append((w * (y - my) * (q - mq)).sum() / np.sqrt((w * (y - my) ** 2).sum() * (w * (q - mq) ** 2).sum()))
            return float(np.mean(r))

        seq_cls = np.broadcast_to(CLS[:, None], M.shape)
        SUMMARY = {name: [wp(p)] + [wp(p, seq_cls == c) for c in range(4)] for name, p in P.items()}
        for name, v in SUMMARY.items():
            print(f"{name:11s} pooled {v[0]:.5f}  " + "  ".join(f"c{c} {v[c + 1]:.4f}" for c in range(4)))
        return SUMMARY, seq_cls, wp
    SUMMARY, seq_cls, wp = _cell()
    return SUMMARY, seq_cls, wp


@app.cell
def _(CLS, M, P, Y, np):
    def _cell():
        # Segment (event-clock) residual series: one value per target run = |y|-weighted mean of (y - beta_c * p) over its
        # scored rows; beta_c is the |y|-weighted LS slope per class (so residuals are not dominated by scale).
        def segment_residuals(p, t=0):
            out = {c: [] for c in range(4)}
            for c in range(4):
                sel = (CLS == c)
                y, q, m = Y[sel][..., t], p[sel][..., t], M[sel]
                w = np.abs(y) * m
                beta = (w * y * q).sum() / max((w * q * q).sum(), 1e-12)
                for i in range(y.shape[0]):
                    change = np.flatnonzero(np.diff(y[i]) != 0) + 1
                    bounds = np.concatenate([[0], change, [y.shape[1]]])
                    res = []
                    for a, b in zip(bounds[:-1], bounds[1:]):
                        ww = w[i, a:b]
                        if ww.sum() > 0:
                            res.append(float((ww * (y[i, a:b] - beta * q[i, a:b])).sum() / ww.sum()))
                    if len(res) > 40:
                        out[c].append(np.asarray(res))
            return out

        def acf(series, nlags=30):
            num = np.zeros(nlags + 1); den = 0.0
            for s in series:
                s = s - s.mean(); den += (s * s).sum()
                for l in range(nlags + 1):
                    num[l] += (s[l:] * s[:len(s) - l]).sum()
            return num / den

        def pacf_from_acf(r):
            """Durbin-Levinson."""
            n = len(r) - 1; phi = np.zeros((n + 1, n + 1)); out = [1.0]
            phi[1, 1] = r[1]; out.append(r[1])
            for k in range(2, n + 1):
                num = r[k] - sum(phi[k - 1, j] * r[k - j] for j in range(1, k))
                den = 1 - sum(phi[k - 1, j] * r[j] for j in range(1, k))
                phi[k, k] = num / den
                for j in range(1, k):
                    phi[k, j] = phi[k - 1, j] - phi[k, k] * phi[k - 1, k - j]
                out.append(phi[k, k])
            return np.asarray(out)

        SEG = {name: segment_residuals(P[name]) for name in ("GRU256", "LSTM192")}
        return SEG, acf, pacf_from_acf
    SEG, acf, pacf_from_acf = _cell()
    return SEG, acf, pacf_from_acf


@app.cell
def _(CLASS_COLORS, PLOTS, SEG, acf, np, pacf_from_acf, plt):
    def _cell():
        fig, axes = plt.subplots(2, 4, figsize=(18, 7), sharey="row")
        lags = np.arange(31)
        for c in range(4):
            for name, ls in (("GRU256", "-"), ("LSTM192", "--")):
                series = SEG[name][c]
                if not series:
                    continue
                r = acf(series); pr = pacf_from_acf(r)
                n_eff = sum(len(s) for s in series)
                axes[0, c].plot(lags[1:], r[1:], ls, color=CLASS_COLORS[c], label=name)
                axes[1, c].plot(lags[1:], pr[1:], ls, color=CLASS_COLORS[c], label=name)
            band = 2 / np.sqrt(max(n_eff, 1))
            for row in (0, 1):
                axes[row, c].axhspan(-band, band, color="0.85", zorder=0); axes[row, c].axhline(0, color="0.4", lw=0.8)
            axes[0, c].set_title(f"class {c}: residual ACF, event clock (t0)"); axes[1, c].set_title(f"class {c}: residual PACF")
            axes[1, c].set_xlabel("lag (target runs)")
        axes[0, 0].legend(); axes[0, 0].set_ylabel("autocorrelation"); axes[1, 0].set_ylabel("partial autocorrelation")
        fig.suptitle("Segment-level residuals (y - beta*p per target run), folds with both students; grey = ±2/sqrt(n)")
        fig.tight_layout(); fig.savefig(PLOTS / "R1_residual_acf_pacf_event_clock.png", dpi=110); plt.close(fig)
        return None
    _cell()
    return


@app.cell
def _(CLASS_COLORS, CLS, M, P, PLOTS, Y, np, plt, wp):
    def _cell():
        # Diversity: residual correlation between the two students, per class, and per-sequence WP scatter.
        def seq_wp_one(p, i):
            r = []
            for t in range(2):
                y, q = Y[i, :, t][M[i]].astype(np.float64), p[i, :, t][M[i]].astype(np.float64)
                w = np.abs(y); sw = w.sum()
                if sw <= 0:
                    return np.nan
                my, mq = (w * y).sum() / sw, (w * q).sum() / sw
                den = np.sqrt((w * (y - my) ** 2).sum() * (w * (q - mq) ** 2).sum())
                r.append((w * (y - my) * (q - mq)).sum() / den if den > 0 else np.nan)
            return float(np.mean(r))

        seq_wp = {name: np.array([seq_wp_one(P[name], i) for i in range(len(CLS))]) for name in ("GRU256", "LSTM192", "GRU256_a07")}
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        rc = []
        for c in range(4):
            sel = (CLS == c)[:, None] & M
            w = np.abs(Y[..., 0])[sel]
            r_g = (Y[..., 0] - P["GRU256"][..., 0])[sel]; r_l = (Y[..., 0] - P["LSTM192"][..., 0])[sel]; r_a = (Y[..., 0] - P["GRU256_a07"][..., 0])[sel]
            cw = lambda a, b: float(np.cov(a, b, aweights=w)[0, 1] / np.sqrt(np.cov(a, aweights=w) * np.cov(b, aweights=w)))
            dg = (P["GRU256"][..., 0] - P["LSTM192"][..., 0])[sel]; da = (P["GRU256"][..., 0] - P["GRU256_a07"][..., 0])[sel]
            rc.append((cw(r_g, r_l), cw(r_g, r_a), float(np.sqrt(np.average(dg ** 2, weights=w))), float(np.sqrt(np.average(da ** 2, weights=w)))))
        x = np.arange(4)
        axes[0].bar(x - 0.2, [r[2] for r in rc], 0.4, label="GRU vs LSTM student", color="#DD8452")
        axes[0].bar(x + 0.2, [r[3] for r in rc], 0.4, label="GRU vs GRU (alpha 0.7) student", color="#4C72B0")
        for i, r in enumerate(rc):
            axes[0].text(i - 0.2, r[2], f"rc {r[0]:.3f}", ha="center", va="bottom", fontsize=8)
            axes[0].text(i + 0.2, r[3], f"rc {r[1]:.3f}", ha="center", va="bottom", fontsize=8)
        axes[0].set_xticks(x, [f"class {c}" for c in range(4)]); axes[0].set_ylabel("|y|-weighted RMS prediction difference (t0)")
        axes[0].set_title("How different are the partners? (rc = residual correlation)"); axes[0].legend()
        for c in range(4):
            s = CLS == c
            axes[1].scatter(seq_wp["GRU256"][s] - seq_wp["GRU256_a07"][s], seq_wp["GRU256"][s] - seq_wp["LSTM192"][s], s=8, alpha=0.5,
                            color=CLASS_COLORS[c], label=f"class {c}")
        axes[1].axhline(0, color="0.5", lw=0.8); axes[1].axvline(0, color="0.5", lw=0.8)
        axes[1].set_xlabel("per-sequence WP: GRU256 - GRU256(alpha 0.7)"); axes[1].set_ylabel("per-sequence WP: GRU256 - LSTM192")
        axes[1].set_title("Per-sequence disagreement (spread = diversity)"); axes[1].legend()
        fig.tight_layout(); fig.savefig(PLOTS / "R2_partner_diversity.png", dpi=110); plt.close(fig)
        print("residual corr / rms diff per class:", [tuple(round(v, 4) for v in r) for r in rc])
        return None
    _cell()
    return


@app.cell
def _(CLASS_COLORS, CLS, M, P, PLOTS, Y, np, plt, wp):
    def _cell():
        # Where in a target run does each model / pair win? WP by rows until the next target change, per class.
        edges = np.array([0, 2, 8, 32, 128, 512, 20000])
        nxt = np.zeros(Y.shape[:2], np.int32)
        for i in range(Y.shape[0]):
            ch = np.flatnonzero(np.diff(Y[i, :, 0]) != 0) + 1
            pos = np.searchsorted(ch, np.arange(Y.shape[1]), side="right")
            nxt_change = np.append(ch, Y.shape[1])[pos]
            nxt[i] = nxt_change - np.arange(Y.shape[1])
        fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
        names = ["LSTM192", "PAIR_GL", "PAIR_GG", "GRU256_a07"]
        for c in range(4):
            base = []
            for name in names:
                vals = []
                for lo, hi in zip(edges[:-1], edges[1:]):
                    sel = (CLS == c)[:, None] & (nxt >= lo) & (nxt < hi)
                    vals.append(wp(P[name], sel) - wp(P["GRU256"], sel))
                axes[c].plot(range(len(vals)), np.asarray(vals) * 1e4, marker="o", label=name)
            axes[c].axhline(0, color="0.5", lw=0.8)
            axes[c].set_xticks(range(len(edges) - 1), [f"{lo}-{hi}" for lo, hi in zip(edges[:-1], edges[1:])], rotation=30)
            axes[c].set_title(f"class {c}: delta WP vs GRU256 student (x1e-4)"); axes[c].set_xlabel("rows until next target change")
        axes[0].legend(); fig.tight_layout(); fig.savefig(PLOTS / "R3_delta_by_run_position.png", dpi=110); plt.close(fig)
        return None
    _cell()
    return


@app.cell
def _(PLOTS, ROOT, json, np, plt):
    def _cell():
        # Local LB probe: holdout (public-like class mix) vs public for every public submission.
        d = json.loads((ROOT / "research" / "lb" / "holdout_stats.json").read_text())
        counts = np.array(d["classes"]); share = counts / counts.sum(); f = np.array([0.183, 0.16, 0.198, 0.459]) / share

        def wp_stats(st):
            s = (np.asarray(st) * f[:, None, None]).sum(0); r = []
            for t in range(2):
                n, sy, sp, syy, spp, syp = s[t]
                r.append((syp / n - sy / n * sp / n) / np.sqrt((syy / n - (sy / n) ** 2) * (spp / n - (sp / n) ** 2)))
            return float(np.mean(r))

        names = list(d["subs"]); pub = np.array([d["subs"][n]["public"] for n in names]); x = np.array([wp_stats(d["subs"][n]["stats"]) for n in names])
        b = np.mean(pub - x)
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.scatter(x + b, pub, color="#4C72B0")
        for n, xi, pi in zip(names, x + b, pub):
            ax.annotate(n.replace("_distilled", "").replace("_shared", ""), (xi, pi), fontsize=7, xytext=(3, 3), textcoords="offset points")
        lo, hi = min(pub.min(), (x + b).min()) - 3e-4, max(pub.max(), (x + b).max()) + 3e-4
        ax.plot([lo, hi], [lo, hi], color="0.5", lw=0.8); ax.set_xlabel("holdout (public-like mix) + shift"); ax.set_ylabel("public")
        ax.set_title(f"Local LB probe: corr {np.corrcoef(x, pub)[0, 1]:.3f}, rmse {np.sqrt(np.mean((pub - x - b) ** 2)) * 1e4:.1f}e-4")
        fig.tight_layout(); fig.savefig(PLOTS / "R4_lb_probe.png", dpi=110); plt.close(fig)
        return None
    _cell()
    return


if __name__ == "__main__":
    app.run()
