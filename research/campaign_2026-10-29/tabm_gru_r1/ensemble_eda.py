import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _():
    import json, sys
    from pathlib import Path
    import numpy as np
    import marimo as mo
    import matplotlib.pyplot as plt
    ROOT = Path.cwd()
    sys.path.insert(0, str(ROOT / "src"))
    from wnn.gmetric import global_wp
    PLOTS = ROOT / "research/campaign_2026-10-29/tabm_gru_r1/plots"
    PLOTS.mkdir(exist_ok=True)
    return PLOTS, ROOT, global_wp, json, mo, np, plt


@app.cell(hide_code=True)
def _(ROOT, global_wp, json, np):
    RUNS = {
        0: ["optim_f0_s1001_muon_r1", "krepro_f0_s2001_muon_r3", "sat_f0_s1001_s2_muon_rms03_r1", "sat_f0_s1001_s3_muon_h512x2_r1",
            "krepro_f0_s2001_adamw_r3", "mech_f0_s1001_e1_adamw_lr5e4_r1", "optim_f0_s1001_cautious_r1", "optim_f0_s1001_ademamix_r1"],
        2: ["optim_f2_s1002_muon_r1", "krepro_f2_s2002_muon_r3", "sat_f2_s1002_s2_muon_rms03_r1", "sat_f2_s1002_s3_muon_h512x2_r1",
            "krepro_f2_s2002_adamw_r3", "mech_f2_s1002_e1_adamw_lr5e4_r1", "optim_f2_s1002_cautious_r1", "optim_f2_s1002_ademamix_r1"],
    }

    def load_fold(fold, runs):
        split = json.loads((ROOT / f"research/campaign_2026-09-29/artifacts/splits/fold{fold}.json").read_text())
        groups = [int(g) for g in split["evaluation"]["valid"]]
        with np.load(ROOT / "cache/valid_targets.npz") as z:
            pos = {int(g): i for i, g in enumerate(z["row_group"])}
            idx = [pos[g] for g in groups]
            y, mask = z["y"][idx].astype(np.float32), z["mask"][idx]
        preds = {}
        for r in runs:
            side = json.loads((ROOT / f"cache/pred_{r}_valid.json").read_text())
            arr = np.load(ROOT / f"cache/pred_{r}_valid.npy", mmap_mode="r")
            rp = {int(g): i for i, g in enumerate(side["row_groups"])}
            preds[r] = np.stack([np.asarray(arr[rp[g]], np.float32) for g in groups])[mask]
        return y[mask], preds

    def wp(y, p):
        return float(global_wp(y, p, np.ones(len(y), bool))["wp"])

    data = {f: load_fold(f, rs) for f, rs in RUNS.items()}
    return data, load_fold, wp


@app.cell(hide_code=True)
def _(data, mo, np, wp):
    import itertools
    def _pairs():
        out = []
        for fold, (y, preds) in data.items():
            w = np.abs(np.clip(y, -2, 2)).ravel()
            single = {r: wp(y, p) for r, p in preds.items()}
            for a, b in itertools.combinations(preds, 2):
                pa, pb = preds[a].ravel(), preds[b].ravel()
                da, db = pa - np.average(pa, weights=w), pb - np.average(pb, weights=w)
                corr = float(np.average(da * db, weights=w) / np.sqrt(np.average(da**2, weights=w) * np.average(db**2, weights=w)))
                pair = wp(y, 0.5 * (preds[a] + preds[b]))
                out.append(dict(fold=fold, a=a, b=b, corr=corr, gain_vs_mean=pair - 0.5 * (single[a] + single[b]),
                                gain_vs_best=pair - max(single[a], single[b]),
                                kind="muon-muon" if ("muon" in a and "muon" in b) else "mixed"))
        return out, {f: {r: wp(y, p) for r, p in pr.items()} for f, (y, pr) in data.items()}
    pairs, singles = _pairs()
    mo.ui.table(pairs)
    return itertools, pairs


@app.cell(hide_code=True)
def _(PLOTS, np, pairs, plt):
    def _plot():
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        for fold, mk in ((0, "o"), (2, "s")):
            for kind, col in (("muon-muon", "tab:blue"), ("mixed", "tab:orange")):
                sel = [r for r in pairs if r["fold"] == fold and r["kind"] == kind]
                ax.scatter([1 - r["corr"] for r in sel], [1e3 * r["gain_vs_mean"] for r in sel], marker=mk, c=col, label=f"F{fold} {kind}", alpha=0.8)
        x = np.array([1 - r["corr"] for r in pairs]); g = np.array([r["gain_vs_mean"] for r in pairs])
        k = float((x * g).sum() / (x * x).sum())
        xs = np.linspace(0, x.max() * 1.05, 50); ax.plot(xs, 1e3 * k * xs, "k--", lw=1, label=f"fit: gain = {k:.3f}·(1−corr)")
        ax.set_xlabel("1 − |y|-weighted prediction correlation"); ax.set_ylabel("pair WP − mean single WP (×1e-3)")
        ax.set_title("Two-model ensemble gain vs disagreement (cold H512 panel)"); ax.legend(fontsize=7); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(PLOTS / "pair_gain_vs_corr.png", dpi=130)
        return fig, k
    pair_fig, slope = _plot()
    pair_fig
    return


@app.cell(hide_code=True)
def _(ROOT, itertools, load_fold, np, wp):
    def member_stats(run, control, fold=0, k=4):
        names = [f"{run}_m{i}" for i in range(k)]
        y, pr = load_fold(fold, [r for r in [run, control, *names] if (ROOT / f"cache/pred_{r}_valid.npy").exists()])
        if run not in pr:
            pr[run] = np.mean([pr[n] for n in names], axis=0)
        w = np.abs(np.clip(y, -2, 2)).ravel()
        def corr(a, b):
            a, b = a.ravel() - np.average(a.ravel(), weights=w), b.ravel() - np.average(b.ravel(), weights=w)
            return float(np.average(a * b, weights=w) / np.sqrt(np.average(a * a, weights=w) * np.average(b * b, weights=w)))
        member_wp = [wp(y, pr[n]) for n in names]
        rhos = [corr(pr[a], pr[b]) for a, b in itertools.combinations(names, 2)]
        mean_pred = np.mean([pr[n] for n in names], axis=0)
        return dict(ensemble=wp(y, pr[run]), ensemble_recomputed=wp(y, mean_pred), control=wp(y, pr[control]),
                    members=member_wp, member_mean=float(np.mean(member_wp)), member_rho_mean=float(np.mean(rhos)),
                    rho_ens_control=corr(pr[run], pr[control]),
                    law_pred_gain=0.168 * (1 - float(np.mean(rhos))) * (1 - 1 / k) * 2)
    tabm4_local = member_stats("tabm29_f0_s1001_tabm4_local_r1", "tabm29_f0_s1001_muon_local_r1")
    tabm4_local
    return member_stats, tabm4_local


@app.cell(hide_code=True)
def _(member_stats, mo):
    tabm4_ones_local = member_stats("tabm29_f0_s1001_tabm4ones_local_r1", "tabm29_f0_s1001_muon_local_r1")
    tabm8_f0 = member_stats("tabm29_f0_s2001_tabm8_r1", "krepro_f0_s2001_muon_r3", fold=0, k=8)
    tabm8_f2 = member_stats("tabm29_f2_s2002_tabm8_r1", "krepro_f2_s2002_muon_r3", fold=2, k=8)
    mo.md(f"ones {tabm4_ones_local}  \n k8 F0 {tabm8_f0}  \n k8 F2 {tabm8_f2}")
    return tabm4_ones_local, tabm8_f0, tabm8_f2


@app.cell(hide_code=True)
def _(PLOTS, np, plt, tabm4_local, tabm4_ones_local, tabm8_f0, tabm8_f2):
    def _tradeoff():
        pts = [("k4 ones F0", tabm4_ones_local, 4), ("k4 signs F0", tabm4_local, 4), ("k8 signs F0", tabm8_f0, 8), ("k8 signs F2", tabm8_f2, 8)]
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        xs = np.linspace(0, 0.015, 50)
        for k, ls in ((4, ":"), (8, "--")):
            ax.plot(xs, 1e3 * 0.336 * (1 - 1 / k) * xs, "k" + ls, lw=1, label=f"ensemble gain law, k={k}")
        for name, s, k in pts:
            x = 1 - s["member_rho_mean"]; deficit = s["control"] - s["member_mean"]
            ax.scatter([x], [1e3 * deficit], s=50, label=f"{name}: member deficit")
            ax.annotate(f"net {1e3*(s['ensemble']-s['control']):+.1f}", (x, 1e3 * deficit), textcoords="offset points", xytext=(5, -10), fontsize=7)
        ax.set_xlabel("1 − mean member-pair correlation"); ax.set_ylabel("×1e-3 WP")
        ax.set_title("TabM-GRU: member weakness grows faster than ensemble gain"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(PLOTS / "tabm_tradeoff.png", dpi=130)
        return fig
    tradeoff_fig = _tradeoff()
    tradeoff_fig
    return


if __name__ == "__main__":
    app.run()
