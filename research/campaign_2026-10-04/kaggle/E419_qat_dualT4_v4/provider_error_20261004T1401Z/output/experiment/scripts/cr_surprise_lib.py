"""Shared helpers for the surprise-miner forensic scripts (row table from cr_surprise_build.py)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

ROWS = Path(os.environ.get("CR_ROWS", "/tmp/claude-1000/-home-vlad-development-wunder-wnn-connectome-starterpack/"
                           "7ea48a59-c833-4cf5-b1fe-467c6d7c1ac5/scratchpad/cr_surprise/rows.npz"))
CLIP = 2.0


def load(fields=None):
    z = np.load(ROWS)
    keys = z.files if fields is None else fields
    return {k: z[k] for k in keys}


def moments(y, p, sel=None):
    """Weighted (|y|) means/vars/cov for one target over selected rows."""
    y = np.clip(y.astype(np.float64), -CLIP, CLIP)
    p = np.clip(p.astype(np.float64), -CLIP, CLIP)
    if sel is not None:
        y, p = y[sel], p[sel]
    w = np.abs(y)
    W = w.sum()
    my, mp = (w * y).sum() / W, (w * p).sum() / W
    vy = (w * (y - my) ** 2).sum() / W
    vp = (w * (p - mp) ** 2).sum() / W
    c = (w * (y - my) * (p - mp)).sum() / W
    return W, my, mp, vy, vp, c


def wp1(y, p, sel=None):
    _, _, _, vy, vp, c = moments(y, p, sel)
    return c / np.sqrt(vy * vp) if vy > 0 and vp > 0 else np.nan


def wp(d, fam="ft", sel=None, p0=None, p1=None):
    p0 = d[fam + "0"] if p0 is None else p0
    p1 = d[fam + "1"] if p1 is None else p1
    return 0.5 * (wp1(d["y0"], p0, sel) + wp1(d["y1"], p1, sel))


def row_loss(d, fam="ft", p0=None, p1=None):
    """Exact additive split of 1 - WP over rows: mean over targets of w (yhat - phat)^2 / (2 W)."""
    out = np.zeros(len(d["y0"]))
    for j, p in ((0, d[fam + "0"] if p0 is None else p0), (1, d[fam + "1"] if p1 is None else p1)):
        y = np.clip(d[f"y{j}"].astype(np.float64), -CLIP, CLIP)
        p = np.clip(p.astype(np.float64), -CLIP, CLIP)
        w = np.abs(y)
        W = w.sum()
        my, mp = (w * y).sum() / W, (w * p).sum() / W
        sy = np.sqrt((w * (y - my) ** 2).sum() / W)
        sp = np.sqrt((w * (p - mp) ** 2).sum() / W)
        out += 0.5 * w * ((y - my) / sy - (p - mp) / sp) ** 2 / (2 * W)
    return out


def weight(d):
    return 0.5 * (np.abs(np.clip(d["y0"], -CLIP, CLIP)) / np.abs(np.clip(d["y0"], -CLIP, CLIP)).sum()
                  + np.abs(np.clip(d["y1"], -CLIP, CLIP)) / np.abs(np.clip(d["y1"], -CLIP, CLIP)).sum())


def recal(d, groups, fam="ft", fold=None, p0=None, p1=None):
    """Per-group |y|-weighted affine remap y ~ a_g + b_g p (per target). With fold: fit on other folds."""
    outs = []
    for j in (0, 1):
        y = np.clip(d[f"y{j}"].astype(np.float64), -CLIP, CLIP)
        p = np.clip((d[f"{fam}{j}"] if (p0 is None) else (p0, p1)[j]).astype(np.float64), -CLIP, CLIP)
        w = np.abs(y)
        q = np.empty_like(p)
        folds = [None] if fold is None else range(4)
        for k in folds:
            fit = np.ones(len(y), bool) if k is None else fold != k
            app = np.ones(len(y), bool) if k is None else fold == k
            G = groups.max() + 1
            sw = np.bincount(groups[fit], w[fit], G)
            sx = np.bincount(groups[fit], w[fit] * p[fit], G)
            sy = np.bincount(groups[fit], w[fit] * y[fit], G)
            sxx = np.bincount(groups[fit], w[fit] * p[fit] ** 2, G)
            sxy = np.bincount(groups[fit], w[fit] * p[fit] * y[fit], G)
            ok = sw > 0
            mx, my = sx / np.where(ok, sw, 1), sy / np.where(ok, sw, 1)
            vx = sxx / np.where(ok, sw, 1) - mx ** 2
            b = np.where(vx > 1e-12, (sxy / np.where(ok, sw, 1) - mx * my) / np.maximum(vx, 1e-12), 1.0)
            a = my - b * mx
            g = groups[app]
            q[app] = a[g] + b[g] * p[app]
        outs.append(q)
    return outs


def profile(d, groups, labels=None, fam="ft", L=None, fold=None):
    """Per-bin weight share, loss share, within-bin WP, calibration slope; plus global gain of a per-bin remap."""
    L = row_loss(d, fam) if L is None else L
    w = weight(d)
    base = wp(d, fam)
    rows = []
    for g in np.unique(groups):
        s = groups == g
        r = {"bin": (labels[g] if labels is not None else int(g)), "rows": int(s.sum()),
             "w_share": float(w[s].sum() / w.sum()), "loss_share": float(L[s].sum() / L.sum())}
        r["loss_per_w"] = r["loss_share"] / max(r["w_share"], 1e-12)
        r["wp_in"] = float(wp(d, fam, s)) if s.sum() > 50 else float("nan")
        sl = []
        for j in (0, 1):
            y = np.clip(d[f"y{j}"][s].astype(np.float64), -2, 2)
            p = np.clip(d[f"{fam}{j}"][s].astype(np.float64), -2, 2)
            ww = np.abs(y)
            mp, my = (ww * p).sum() / ww.sum(), (ww * y).sum() / ww.sum()
            vp = (ww * (p - mp) ** 2).sum() / ww.sum()
            sl.append(((ww * (p - mp) * (y - my)).sum() / ww.sum()) / vp if vp > 0 else np.nan)
        r["slope"] = float(np.mean(sl))
        rows.append(r)
    q0, q1 = recal(d, groups, fam)
    gain_in = wp(d, fam, p0=q0, p1=q1) - base
    gain_cf = None
    if fold is not None:
        q0, q1 = recal(d, groups, fam, fold)
        gain_cf = wp(d, fam, p0=q0, p1=q1) - base
    return rows, gain_in, gain_cf


def show(title, rows, gain_in, gain_cf):
    print(f"== {title}  remap gain in-sample {gain_in:+.5f}  crossfit {gain_cf if gain_cf is None else f'{gain_cf:+.5f}'}")
    print(f"{'bin':>14} {'rows':>9} {'w%':>6} {'loss%':>6} {'L/w':>5} {'wp_in':>6} {'slope':>6}")
    for r in rows:
        print(f"{str(r['bin']):>14} {r['rows']:>9} {100*r['w_share']:6.2f} {100*r['loss_share']:6.2f} "
              f"{r['loss_per_w']:5.2f} {r['wp_in']:6.3f} {r['slope']:6.3f}")
