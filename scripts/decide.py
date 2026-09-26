"""Model decisions from cached per-sequence metric sums (research decision rule, E-084).

    scripts/limited .venv/bin/python scripts/decide.py cache MODEL [MODEL ...]
    scripts/limited .venv/bin/python scripts/decide.py compare A B [--lam 0,0.02]
    scripts/limited .venv/bin/python scripts/decide.py blend A B [--weights 0.5]

``cache`` turns cache/pred_<MODEL>_valid.npy into cache/seqsums_<MODEL>.npz: per sequence
and target, the six WP sums over scored rows and over unscored required rows. Global WP
under a softened mask (unscored rows weighted by lam, the critic's test-mask model) is a
merge of these sums. ``compare`` reports A - B per split (devA, devB, holdout, all) with a
paired chain bootstrap and applies the rule: spend a leaderboard slot only if the gap exceeds
2 sd and has the same sign on every fold (default: 4 chain-grouped folds; --grouping dev for the
old devA/devB/holdout split, which leaks chain neighbours). Model names may carry a scale,
e.g. baseline*0.25.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from wnn.gmetric import wp_from_sums, wp_sums
from wnn.splits import dev_halves, valid_row_groups

ROOT = Path(__file__).resolve().parents[1]
WARMUP = 99


def _pred(spec):
    name, _, scale = spec.partition("*")
    p = np.load(ROOT / "cache" / f"pred_{name}_valid.npy", mmap_mode="r")
    return p, float(scale or 1.0)


def _sums6(y, p, sel):
    """(B, 6) WP sums over the last axis for rows where sel; y, p already clipped float64."""
    w = np.abs(y) * sel
    wy, wp = w * y, w * p
    return np.stack([w.sum(-1), wy.sum(-1), wp.sum(-1), (wy * y).sum(-1), (wp * p).sum(-1), (wy * p).sum(-1)], -1)


def build_sums(spec, weights=None, chunk=100):
    """(B, 2 [scored/unscored], 2 targets, 6) sums; spec may be 'A' or blend list [(spec, w)]."""
    d = np.load(ROOT / "cache" / "valid_targets.npz")
    y, m = d["y"], d["mask"]
    parts = [(_pred(s), w) for s, w in (weights or [(spec, 1.0)])]
    req = np.arange(y.shape[1]) >= WARMUP
    out = np.zeros((len(y), 2, 2, 6))
    for a in range(0, len(y), chunk):
        sl = slice(a, a + chunk)
        p = sum(w * sc * np.asarray(arr[sl], dtype=np.float64) for (arr, sc), w in parts)
        yc = np.clip(y[sl].astype(np.float64), -2, 2)
        pc = np.clip(p, -2, 2)
        ms = m[sl]
        for k in range(2):
            out[sl, 0, k] = _sums6(yc[..., k], pc[..., k], ms)
            out[sl, 1, k] = _sums6(yc[..., k], pc[..., k], req & ~ms)
    return out


def load_sums(spec):
    f = ROOT / "cache" / f"seqsums_{spec.replace('*', 'x')}.npz"
    if not f.exists():
        np.savez(f, s=build_sums(spec))
    return np.load(f)["s"]


def wp(s, lam=0.0):
    t = s[:, 0] + lam * s[:, 1]
    return float(np.mean([wp_from_sums(t[:, k].sum(0)) for k in range(2)]))


GROUPING = "chain"  # "chain": 4 chain-grouped folds (cache/chain_folds.npz); "dev": devA/devB/holdout


def _chains():
    rg = np.load(ROOT / "cache" / "valid_targets.npz")["row_group"]
    c = np.load(ROOT / "cache" / "chain_folds.npz")
    v = c["split"] == "valid"
    pos = {int(g): i for i, g in enumerate(c["row_group"][v])}
    order = np.array([pos[int(g)] for g in rg])
    return c["chain"][v][order], c["fold"][v][order]


STRESS = False  # restrict to the 2/3 of validation sequences least like train (cache/adv_valid_score.npy)


def _stress_keep():
    s = np.load(ROOT / "cache" / "adv_valid_score.npy")
    cls = np.load(ROOT / "cache" / "valid_targets.npz")["market_class"]
    keep = np.zeros(len(s), bool)
    for k in range(4):  # within class, so the class mix is unchanged
        idx = np.flatnonzero(cls == k)
        keep[idx[s[idx] >= np.nanquantile(s[idx], 1 / 3)]] = True
    return keep


def splits():
    rg = np.load(ROOT / "cache" / "valid_targets.npz")["row_group"]
    if STRESS:
        keep = _stress_keep()
        base = _splits_plain(rg)
        return {k: v[keep[v]] for k, v in base.items()}
    return _splits_plain(rg)


def _splits_plain(rg):
    if GROUPING == "chain":
        _, fold = _chains()
        out = {f"fold{k}": np.flatnonzero(fold == k) for k in range(4)}
        out["all"] = np.arange(len(rg))
        return out
    path = ROOT / "datasets" / "valid.parquet"
    _, hold = valid_row_groups(path)
    dev_a, dev_b = dev_halves(path)
    return {"devA": np.flatnonzero(np.isin(rg, np.asarray(dev_a))), "devB": np.flatnonzero(np.isin(rg, np.asarray(dev_b))),
            "holdout": np.flatnonzero(np.isin(rg, np.asarray(hold))), "all": np.arange(len(rg))}


def _resample(idx, rng, chain):
    """Bootstrap by chain (sequences of one chain move together) or by sequence."""
    if chain is None:
        return rng.choice(idx, len(idx))
    ids = np.unique(chain[idx])
    pick = rng.choice(ids, len(ids))
    members = {c: idx[chain[idx] == c] for c in ids}
    return np.concatenate([members[c] for c in pick])


def compare(sa, sb, lam, n_boot=300, seed=0):
    rng = np.random.default_rng(seed)
    chain = _chains()[0] if GROUPING == "chain" else None
    rows = {}
    for name, idx in splits().items():
        delta = wp(sa[idx], lam) - wp(sb[idx], lam)
        boots = []
        for _ in range(n_boot):
            b = _resample(idx, rng, chain)
            boots.append(wp(sa[b], lam) - wp(sb[b], lam))
        rows[name] = (delta, float(np.std(boots)), wp(sa[idx], lam), wp(sb[idx], lam))
    parts = [k for k in rows if k != "all"]
    ok = all(np.sign(rows[s][0]) == np.sign(rows["all"][0]) for s in parts) and abs(rows["all"][0]) > 2 * rows["all"][1]
    return rows, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["cache", "compare", "blend"])
    ap.add_argument("models", nargs="+")
    ap.add_argument("--lam", default="0,0.02")
    ap.add_argument("--weights", default="0.5")
    ap.add_argument("--grouping", choices=["chain", "dev"], default="chain")
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--stress", action="store_true", help="only the 2/3 least train-like sequences per class (LB-ordering proxy)")
    a = ap.parse_args()
    global GROUPING, STRESS
    GROUPING = a.grouping
    STRESS = a.stress
    lams = [float(v) for v in a.lam.split(",")]
    if a.mode == "cache":
        for mdl in a.models:
            s = load_sums(mdl)
            print(mdl, " ".join(f"lam={lam}: {wp(s, lam):.4f}" for lam in lams))
        return
    if a.mode == "blend":
        wa = float(a.weights)
        spec = f"blend[{a.models[0]}:{wa}|{a.models[1]}:{1 - wa}]"
        f = ROOT / "cache" / f"seqsums_{spec.replace('*', 'x').replace('|', '_').replace(':', '-')}.npz"
        if not f.exists():
            np.savez(f, s=build_sums(None, [(a.models[0], wa), (a.models[1], 1 - wa)]))
        sa = np.load(f)["s"]
        for ref in a.models:
            for lam in lams:
                rows, ok = compare(sa, load_sums(ref), lam)
                print(f"{spec} - {ref} lam={lam}: " + "  ".join(f"{k} {v[0]:+.4f}±{v[1]:.4f}" for k, v in rows.items()) + f"  pass={ok}")
        return
    sa, sb = load_sums(a.models[0]), load_sums(a.models[1])
    for lam in lams:
        rows, ok = compare(sa, sb, lam)
        print(f"{a.models[0]} - {a.models[1]} lam={lam}: " + "  ".join(f"{k} {v[0]:+.4f}±{v[1]:.4f} ({v[2]:.4f} vs {v[3]:.4f})" for k, v in rows.items()) + f"  pass={ok}")


if __name__ == "__main__":
    main()
