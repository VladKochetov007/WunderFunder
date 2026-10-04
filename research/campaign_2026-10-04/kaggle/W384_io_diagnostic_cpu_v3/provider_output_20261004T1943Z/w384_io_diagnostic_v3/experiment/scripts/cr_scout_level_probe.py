"""Scout falsifiers on cached clean-fold OOF predictions (no training, CPU, seconds-minutes).

F1  between- vs within-sequence decomposition of |y|-weighted cov/var, per class and target.
F1b causal sequence-level de-meaning/amplification: p' = p - lam * causal_mean_seq(p), lam swept per class.
F1c label-free per-sequence scale: p' = p * (causal_rms_seq(p) / class_rms_other_folds)^gamma, per class.
F3  per-sequence residual persistence: does the |y|-weighted mean residual of earlier (gapped) blocks
    predict the residual of later blocks? (upper bound on what matured-label online correction could add)

    WNN_CPU_PCT=19 WNN_MEM_MAX=8G scripts/limited .venv/bin/python scripts/cr_scout_level_probe.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wnn.gmetric import wp_from_sums  # noqa: E402

OUT = ROOT / "research/campaign_2026-10-01/creative/cr_scout/probe_level.json"
MEMBERS = ("c20260929_ft_since_s3_f{k}", "c20260929_ft_since_s4_f{k}")
LAMS = (-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0)
GAMMAS = (-0.5, -0.25, 0.0, 0.25, 0.5, 1.0)
WARM = 99
BLOCK, GAP = 1000, 1


def causal_mean(p: np.ndarray) -> np.ndarray:
    """Running mean of p over rows WARM..t (inclusive), per sequence; zeros before WARM."""
    out = np.zeros_like(p, dtype=np.float64)
    c = np.cumsum(p[:, WARM:].astype(np.float64), axis=1)
    n = np.arange(1, c.shape[1] + 1, dtype=np.float64)[None, :, None]
    out[:, WARM:] = c / n
    return out


def causal_rms(p: np.ndarray) -> np.ndarray:
    """Running RMS of p over rows WARM..t, per sequence; ones before WARM."""
    out = np.ones_like(p, dtype=np.float64)
    c = np.cumsum(p[:, WARM:].astype(np.float64) ** 2, axis=1)
    n = np.arange(1, c.shape[1] + 1, dtype=np.float64)[None, :, None]
    out[:, WARM:] = np.sqrt(c / n)
    return out


def decompose(y, p, m) -> dict:
    """Between/within split of |y|-weighted moments for one target (y,p: [S,T])."""
    yc, pc = np.clip(y, -2, 2), np.clip(p, -2, 2)
    w = np.where(m, np.abs(yc), 0.0)
    W = w.sum()
    ws = w.sum(1)
    ok = ws > 0
    ys = (w * yc).sum(1)[ok] / ws[ok]
    ps = (w * pc).sum(1)[ok] / ws[ok]
    my, mp = (w * yc).sum() / W, (w * pc).sum() / W
    wsn = ws[ok] / W
    cb = (wsn * (ys - my) * (ps - mp)).sum()
    vyb, vpb = (wsn * (ys - my) ** 2).sum(), (wsn * (ps - mp) ** 2).sum()
    ysf = np.zeros(len(ws)); ysf[ok] = ys
    psf = np.zeros(len(ws)); psf[ok] = ps
    yd, pd = yc - ysf[:, None], pc - psf[:, None]
    cw = (w * yd * pd).sum() / W
    vyw, vpw = (w * yd * yd).sum() / W, (w * pd * pd).sum() / W
    return {
        "var_y_between_share": vyb / (vyb + vyw), "var_p_between_share": vpb / (vpb + vpw),
        "corr_between": cb / np.sqrt(vyb * vpb), "corr_within": cw / np.sqrt(vyw * vpw),
        "cov_between_share": cb / (cb + cw), "corr_total": (cb + cw) / np.sqrt((vyb + vyw) * (vpb + vpw)),
    }


def residual_persistence(y, p, m) -> dict:
    """Weighted corr between a block's mean residual and the mean residual of all blocks <= b-1-GAP."""
    yc, pc = np.clip(y, -2, 2), np.clip(p, -2, 2)
    w = np.where(m, np.abs(yc), 0.0)
    beta = (w * yc * pc).sum() / (w * pc * pc).sum()
    r = yc - beta * pc
    S, T = y.shape
    nb = T // BLOCK
    wb = w[:, : nb * BLOCK].reshape(S, nb, BLOCK).sum(2)
    rb = (w * r)[:, : nb * BLOCK].reshape(S, nb, BLOCK).sum(2)
    xs, ts, ws = [], [], []
    for b in range(1 + GAP, nb):
        past_w = wb[:, : b - GAP].sum(1)
        ok = (past_w > 0) & (wb[:, b] > 0)
        xs.append(rb[ok, : b - GAP].sum(1) / past_w[ok])
        ts.append(rb[ok, b] / wb[ok, b])
        ws.append(wb[ok, b])
    x, t, ww = map(np.concatenate, (xs, ts, ws))
    mx, mt = np.average(x, weights=ww), np.average(t, weights=ww)
    c = np.average((x - mx) * (t - mt), weights=ww)
    corr = c / np.sqrt(np.average((x - mx) ** 2, weights=ww) * np.average((t - mt) ** 2, weights=ww))
    slope = c / np.average((x - mx) ** 2, weights=ww)
    return {"beta": float(beta), "corr_past_resid_vs_block_resid": float(corr), "slope": float(slope), "n": int(len(x))}


def sums(y, q, m, extra=None) -> np.ndarray:
    """Six raw |y|-weighted sums (W, Wy, Wq, Wyy, Wqq, Wyq) for one target, predictions clipped like the metric."""
    yc, qc = np.clip(y, -2, 2), np.clip(q, -2, 2)
    w = np.where(m, np.abs(yc), 0.0)
    wy, wq = w * yc, w * qc
    return np.array([w.sum(), wy.sum(), wq.sum(), (wy * yc).sum(), (wq * qc).sum(), (wy * qc).sum()])


def main() -> None:
    z = np.load(ROOT / "cache/valid_targets.npz")
    row_group, cls_all = z["row_group"], z["market_class"]
    pos = {int(g): i for i, g in enumerate(row_group)}
    report = {"members": MEMBERS, "lams": LAMS, "gammas": GAMMAS, "folds": {}}
    data = {}
    for k in range(4):
        man = json.loads((ROOT / f"research/campaign_2026-09-29/artifacts/splits/fold{k}.json").read_text())
        idx = np.array(sorted(pos[int(g)] for g in man["evaluation"]["valid"]))
        p = sum(np.load(ROOT / f"cache/pred_{name.format(k=k)}_valid.npy", mmap_mode="r")[idx].astype(np.float64)
                for name in MEMBERS) / len(MEMBERS)
        data[k] = (idx, p)
    # class RMS of predictions per fold (for F1c, used out-of-fold)
    crms = {k: {c: np.sqrt((p[cls_all[idx] == c, WARM:] ** 2).mean(axis=(0, 1))) for c in range(4)}
            for k, (idx, p) in data.items()}
    for k, (idx, p) in data.items():
        y = z["y"][idx].astype(np.float64)
        m = z["mask"][idx]
        cls = cls_all[idx]
        cm = causal_mean(p)
        rms = causal_rms(p)
        # per (variant, class, target) raw sums; totals are sums over classes
        base = np.stack([[sums(y[cls == c, :, j], p[cls == c, :, j], m[cls == c]) for j in range(2)] for c in range(4)])
        lam_s = {lam: np.stack([[sums(y[cls == c, :, j], p[cls == c, :, j] - lam * cm[cls == c, :, j], m[cls == c])
                                 for j in range(2)] for c in range(4)]) for lam in LAMS}
        gam_s = {}
        for g in GAMMAS:
            rows = []
            for c in range(4):
                sel = cls == c
                ref = np.mean([crms[j][c] for j in data if j != k], axis=0)
                ratio = np.clip(rms[sel] / ref, 0.25, 4.0) ** g
                rows.append([sums(y[sel, :, j], p[sel, :, j] * ratio[:, :, j], m[sel]) for j in range(2)])
            gam_s[g] = np.stack(rows)
        b = float(wp_from_sums(base.sum(0)).mean())

        def swap(alt, c):
            t = base.copy()
            t[c] = alt[c]
            return float(wp_from_sums(t.sum(0)).mean()) - b

        fold = {"base_wp": b, "classes": {}}
        fold["lam_delta_per_class"] = {c: {lam: swap(lam_s[lam], c) for lam in LAMS} for c in range(4)}
        fold["lam_delta_all"] = {lam: float(wp_from_sums(lam_s[lam].sum((0))).mean()) - b for lam in LAMS}
        fold["gamma_delta_per_class"] = {c: {g: swap(gam_s[g], c) for g in GAMMAS} for c in range(4)}
        for c in range(4):
            sel = cls == c
            fold["classes"][c] = {
                "n_seq": int(sel.sum()),
                "class_wp": wp_from_sums(base[c]).tolist(),
                "decomp": [decompose(y[sel, :, j], p[sel, :, j], m[sel]) for j in range(2)],
                "resid": [residual_persistence(y[sel, :, j], p[sel, :, j], m[sel]) for j in range(2)],
            }
        report["folds"][k] = fold
        print(f"fold {k} base {b:.5f}")
        print("  lam  ", json.dumps({c: {l: round(v, 5) for l, v in fold["lam_delta_per_class"][c].items()} for c in range(4)}))
        print("  gamma", json.dumps({c: {g: round(v, 5) for g, v in fold["gamma_delta_per_class"][c].items()} for c in range(4)}))
        for c in range(4):
            d = fold["classes"][c]["decomp"]
            r = fold["classes"][c]["resid"]
            print(f"  cls{c} n={fold['classes'][c]['n_seq']} " + " | ".join(
                f"t{j}: vyB={d[j]['var_y_between_share']:.3f} vpB={d[j]['var_p_between_share']:.3f} "
                f"rB={d[j]['corr_between']:.3f} rW={d[j]['corr_within']:.3f} covB={d[j]['cov_between_share']:.3f} "
                f"res_pers={r[j]['corr_past_resid_vs_block_resid']:.3f}/{r[j]['slope']:.3f}" for j in range(2)), flush=True)
        OUT.write_text(json.dumps(report, indent=1, default=float))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
