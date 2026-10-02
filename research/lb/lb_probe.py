"""Local leaderboard probe: every public submission re-scored on the 368-sequence holdout none of its models trained on.

build : assemble each submission's holdout prediction from cached member predictions (registry.json), archive it as
        float16 under research/lb/preds/<name>.npy, and store holdout sufficient statistics per class.
fit   : relate holdout metrics to public scores (pooled, public-like class mix, fitted class mix), with leave-one-out
        error, so a new candidate's public score can be estimated from its holdout prediction alone.
score : estimate public for any member spec, e.g.  lb_probe.py score "RUN|HEAD:1.0"  or a saved npy.
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from wnn.splits import is_holdout  # noqa: E402

LB = ROOT / "research" / "lb"
PUB143 = np.array([0.183, 0.16, 0.198, 0.459])


def holdout_index():
    vt = np.load(ROOT / "cache" / "valid_targets.npz", mmap_mode="r")
    idx = np.flatnonzero([is_holdout(int(s)) for s in vt["seq_ix"]])
    return vt, idx


def member_pred(source: str, idx: np.ndarray) -> np.ndarray:
    parts = source.split("|")
    out = None
    for p in parts:
        run = p.split("@")[0]
        a = np.asarray(np.load(ROOT / "cache" / f"pred_{run}_valid.npy", mmap_mode="r")[idx], dtype=np.float32)
        out = a if out is None else out + a
    return out / len(parts)


def assemble(spec: dict, idx, classes) -> np.ndarray:
    if "routed" in spec:
        r = spec["routed"]
        groups = {g: assemble({"members": r[g]}, idx, classes) for g in ("a", "b")}
        pick = np.array([r["class_group"][str(int(c))] for c in classes])
        return np.where((pick == "a")[:, None, None], groups["a"], groups["b"])
    return sum(coef * member_pred(src, idx) for src, coef in spec["members"])


def class_stats(y, p, m, classes):
    """Per class and target: [sum w, sum wy, sum wp, sum wyy, sum wpp, sum wyp], w = |y| on scored rows (clip 2)."""
    out = np.zeros((4, 2, 6))
    for c in range(4):
        sel = classes == c
        for t in range(2):
            yy = np.clip(y[sel][..., t], -2, 2)[m[sel]].astype(np.float64)
            pp = np.clip(p[sel][..., t], -2, 2)[m[sel]].astype(np.float64)
            w = np.abs(yy)
            out[c, t] = [w.sum(), (w * yy).sum(), (w * pp).sum(), (w * yy * yy).sum(), (w * pp * pp).sum(), (w * yy * pp).sum()]
    return out


def wp(stats, class_factor=np.ones(4)):
    s = (stats * class_factor[:, None, None]).sum(0)
    r = []
    for t in range(2):
        n, sy, sp, syy, spp, syp = s[t]
        cov = syp / n - sy / n * sp / n
        r.append(cov / np.sqrt((syy / n - (sy / n) ** 2) * (spp / n - (sp / n) ** 2)))
    return float(np.mean(r))


def mix_factor(mix, classes):
    share = np.bincount(classes, minlength=4) / len(classes)
    return np.asarray(mix) / np.maximum(share, 1e-12)


def cmd_build(_):
    reg = json.loads((LB / "registry.json").read_text())
    vt, idx = holdout_index()
    y = np.asarray(vt["y"][idx]); m = np.asarray(vt["mask"][idx]); cls = np.asarray(vt["market_class"][idx])
    table = {}
    for s in reg["subs"]:
        p = assemble(s, idx, cls)
        np.save(LB / "preds" / f"{s['name']}.npy", p.astype(np.float16))
        st = class_stats(y, p, m, cls)
        table[s["name"]] = {"public": s["public"], "stats": st.tolist()}
        print(f"{s['name']:32s} public {s['public']:.4f} holdout {wp(st):.5f} pub143 {wp(st, mix_factor(PUB143, cls)):.5f} "
              f"cls {[round(wp(st[c:c + 1]), 4) for c in range(4)]}")
    (LB / "holdout_stats.json").write_text(json.dumps({"classes": np.bincount(cls, minlength=4).tolist(), "subs": table}))


def cmd_fit(_):
    d = json.loads((LB / "holdout_stats.json").read_text())
    counts = np.array(d["classes"]); share = counts / counts.sum()
    names = list(d["subs"]); pub = np.array([d["subs"][n]["public"] for n in names])
    stats = {n: np.array(d["subs"][n]["stats"]) for n in names}

    def feats(mix):
        f = np.asarray(mix) / share
        return np.array([wp(stats[n], f) for n in names])

    def loo(x):
        err = []
        for i in range(len(names)):
            k = np.arange(len(names)) != i
            b = np.mean(pub[k] - x[k])
            err.append(pub[i] - (x[i] + b))
        return np.sqrt(np.mean(np.square(err))), np.array(err)

    rows = []
    for label, mix in (("holdout", share), ("pub143", PUB143)):
        x = feats(mix); r, _ = loo(x); rows.append((r, label, mix, np.corrcoef(x, pub)[0, 1]))
    grid = [m for m in itertools.product(np.arange(0.05, 0.75, 0.05), repeat=3) if 0.05 <= 1 - sum(m) <= 0.8]
    best = min(((loo(feats((*m, 1 - sum(m))))[0], m) for m in grid), key=lambda t: t[0])
    mix = (*best[1], 1 - sum(best[1])); x = feats(mix); rows.append((best[0], "fitted(in-sample grid; optimistic)", mix, np.corrcoef(x, pub)[0, 1]))
    for r, label, mix, c in rows:
        print(f"{label:36s} LOO rmse {r * 1e4:5.1f}e-4  corr {c:+.3f}  mix {np.round(mix, 3).tolist()}")
    x = feats(PUB143); b = np.mean(pub - x)
    print("pub143 shift b =", round(b, 5))
    for n, xi, pi in zip(names, x, pub):
        print(f"  {n:32s} public {pi:.4f}  proxy {xi + b:.4f}  resid {(pi - xi - b) * 1e4:+5.1f}e-4")


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build"); sub.add_parser("fit")
    a = ap.parse_args(); {"build": cmd_build, "fit": cmd_fit}[a.cmd](a)


if __name__ == "__main__":
    main()
