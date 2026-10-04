"""Shared helpers for the cr_dtw_labels probes: load the event/segment tables and score additive corrections.

Segment sums (from cr_dtw_labels_build.py) per (model, target): |y|-weighted
  0 W, 1 WY, 2 WP, 3 WYY, 4 WPP, 5 WYP, 6 WD, 7 WPD, 8 WDD, 9 WYD   (D = p_t - p at segment start).
A correction p' = p + c_s (constant on a segment) or p + g D updates the six Pearson sums exactly
(the +-2 clip of p' is ignored; corrections are small).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research/campaign_2026-10-02/dtw/labels"
PUBLIC_MIX = np.array([0.183, 0.160, 0.198, 0.459])


def load(path: Path = OUT / "tables.npz") -> dict:
    z = np.load(path)
    t = {k: z[k] for k in z.files}
    t["sg_seq"] = t["ev_seq"][t["sg_sidx"]]
    t["sg_cls"] = t["cls_of_seq"][t["sg_seq"]]
    t["sg_fold"] = t["fold_of_seq"][t["sg_seq"]]
    return t


def seq_weights(t: dict, mix: np.ndarray | None) -> np.ndarray:
    """Per-scored-segment weight moving the panel's sequence class shares to mix (screen_score rule)."""
    if mix is None:
        return np.ones(len(t["sg_seq"]))
    share = np.bincount(t["cls_of_seq"], minlength=4) / len(t["cls_of_seq"])
    return (mix / share)[t["sg_cls"]]


def wp6(s: np.ndarray) -> np.ndarray:
    """Pearson per target from (..., 2, 6) pooled sums."""
    W, WY, WP, WYY, WPP, WYP = np.moveaxis(s, -1, 0)
    my, mp = WY / W, WP / W
    vy, vp = WYY / W - my**2, WPP / W - mp**2
    return (WYP / W - my * mp) / np.sqrt(np.maximum(vy * vp, 1e-30))


def corrected(sums: np.ndarray, c: np.ndarray | None = None, g: float | np.ndarray = 0.0) -> np.ndarray:
    """(n, 2, 6) Pearson sums of p + c (per segment, per target) + g * D for one model's (n, 2, 10) sums."""
    s = sums[..., :6].astype(np.float64).copy()
    W, WY, WP = sums[..., 0], sums[..., 1], sums[..., 2]
    if c is not None:
        s[..., 2] += c * W
        s[..., 4] += 2 * c * WP + c * c * W
        s[..., 5] += c * WY
    g = np.asarray(g, dtype=np.float64)
    if np.any(g != 0):
        WD, WPD, WDD, WYD = sums[..., 6], sums[..., 7], sums[..., 8], sums[..., 9]
        s[..., 2] += g * WD
        s[..., 4] += 2 * g * WPD + g * g * WDD + (2 * c * g * WD if c is not None else 0)
        s[..., 5] += g * WYD
    return s


def score(s6: np.ndarray, w: np.ndarray, sel: np.ndarray | None = None) -> float:
    """Mean-of-targets pooled WP of selected segments with per-segment weights."""
    if sel is not None:
        s6, w = s6[sel], w[sel]
    tot = (s6 * w[:, None, None]).sum(0)
    return float(wp6(tot).mean())


def report(t: dict, s_ref: np.ndarray, s_new: np.ndarray) -> dict:
    """Delta WP pooled, per class, public mix, per fold (pooled within fold)."""
    out = {}
    for name, mix in (("pooled", None), ("public_mix", PUBLIC_MIX)):
        w = seq_weights(t, mix)
        out[name] = score(s_new, w) - score(s_ref, w)
        out[name + "_ref"] = score(s_ref, w)
        out[name + "_fold"] = [score(s_new, w, t["sg_fold"] == k) - score(s_ref, w, t["sg_fold"] == k)
                               for k in range(4)]
    w = np.ones(len(s_ref))
    out["class"] = [score(s_new, w, t["sg_cls"] == c) - score(s_ref, w, t["sg_cls"] == c) for c in range(4)]
    out["class_ref"] = [score(s_ref, w, t["sg_cls"] == c) for c in range(4)]
    return out


def cells(t: dict, s: np.ndarray) -> np.ndarray:
    """Aggregate per-segment (n, 2, k) sums into (class, fold, 2, k) cells."""
    cid = t["sg_cls"] * 4 + t["sg_fold"]
    out = np.zeros((16,) + s.shape[1:])
    np.add.at(out, cid, s)
    return out.reshape((4, 4) + s.shape[1:])


def corrected_cells(cs: np.ndarray, g: np.ndarray) -> np.ndarray:
    """(4, 4, 2, 6) Pearson sums of p + g[class, fold] * D from (4, 4, 2, 10) cell sums."""
    g = np.asarray(g, dtype=np.float64)[..., None, None] * np.ones(cs.shape[:3] + (1,))
    g = g[..., 0]
    s = cs[..., :6].copy()
    s[..., 2] += g * cs[..., 6]
    s[..., 4] += 2 * g * cs[..., 7] + g * g * cs[..., 8]
    s[..., 5] += g * cs[..., 9]
    return s


def report_cells(t: dict, ref: np.ndarray, new: np.ndarray) -> dict:
    """Delta WP from (4 class, 4 fold, 2, 6) cells: pooled, public mix, per fold, per class."""
    share = np.bincount(t["cls_of_seq"], minlength=4) / len(t["cls_of_seq"])
    out = {}
    for name, cw in (("pooled", np.ones(4)), ("public_mix", PUBLIC_MIX / share)):
        f = lambda c, sel=(slice(None), slice(None)): float(wp6((c * cw[:, None, None, None])[sel].sum((0, 1))).mean())
        out[name] = f(new) - f(ref)
        out[name + "_ref"] = f(ref)
        out[name + "_fold"] = [float(wp6((new * cw[:, None, None, None])[:, k].sum(0)).mean()
                                     - wp6((ref * cw[:, None, None, None])[:, k].sum(0)).mean()) for k in range(4)]
    out["class"] = [float(wp6(new[c].sum(0)).mean() - wp6(ref[c].sum(0)).mean()) for c in range(4)]
    out["class_ref"] = [float(wp6(ref[c].sum(0)).mean()) for c in range(4)]
    out["class_fold"] = [[float(wp6(new[c, k]).mean() - wp6(ref[c, k]).mean()) for k in range(4)] for c in range(4)]
    return out
