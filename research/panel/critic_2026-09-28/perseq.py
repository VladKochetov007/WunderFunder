"""Independent keep_k4 vs incumbent evaluation from per-sequence sums (sums.py output)."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
sys.path.insert(0, str(ROOT / "src"))
from wnn.splits import valid_row_groups  # noqa: E402

S = dict(np.load(sys.argv[1]))
d = np.load(ROOT / "cache/valid_targets.npz")
rg, mc = d["row_group"], d["market_class"]
N = len(rg)
cf = np.load(ROOT / "cache/chain_folds.npz")
v = cf["split"] == "valid"
m_chain = dict(zip(cf["row_group"][v].tolist(), cf["chain"][v].tolist()))
m_fold = dict(zip(cf["row_group"][v].tolist(), cf["fold"][v].tolist()))
chain = np.array([m_chain[int(g)] for g in rg])
fold = np.array([m_fold[int(g)] for g in rg])
train_chains = set(cf["chain"][~v].tolist())
pure = np.array([c not in train_chains for c in chain])
_, held = valid_row_groups(ROOT / "datasets/valid.parquet")
hold = np.isin(rg, held)
# holdout sequence whose chain also has dev (trained-on) valid sequences or train sequences
dev_chains = set(chain[~hold].tolist())
hold_nb = hold & np.array([(c in dev_chains) or (c in train_chains) for c in chain])
train_mix = np.array([1305, 1257, 3488, 4557], float)
train_mix /= train_mix.sum()


def wp_tot(t):  # t: (2, 6) summed
    out = []
    for k in range(2):
        w, wy, wp_, wyy, wpp, wyp = t[k]
        my, mp = wy / w, wp_ / w
        cov = wyp / w - my * mp
        vy, vp = max(wyy / w - my * my, 0), max(wpp / w - mp * mp, 0)
        out.append(cov / np.sqrt(vy * vp))
    return float(np.mean(out))


def wp_batch(T):  # T: (B, 2, 6)
    w = T[..., 0]
    my, mp = T[..., 1] / w, T[..., 2] / w
    cov = T[..., 5] / w - my * mp
    vy = np.maximum(T[..., 3] / w - my * my, 0)
    vp = np.maximum(T[..., 4] / w - mp * mp, 0)
    return (cov / np.sqrt(vy * vp)).mean(-1)


def seq_weights(idx, mix):
    """Per-sequence weights so that the class shares within idx equal `mix` (None = as is)."""
    if mix is None:
        return np.ones(len(idx))
    cls = mc[idx]
    share = np.array([(cls == c).mean() for c in range(4)])
    return np.array([mix[c] / share[c] for c in cls])


def delta(A, B, idx, mix=None, n_boot=2000, seed=0):
    """A, B: (N, 2, 6) sums. Returns delta, chain-bootstrap sd, P(delta<=0), wpA, wpB."""
    wts = seq_weights(idx, mix)
    a = A[idx] * wts[:, None, None]
    b = B[idx] * wts[:, None, None]
    ch = chain[idx]
    uc, inv = np.unique(ch, return_inverse=True)
    ca = np.zeros((len(uc), 2, 6)); np.add.at(ca, inv, a)
    cb = np.zeros((len(uc), 2, 6)); np.add.at(cb, inv, b)
    base = wp_tot(ca.sum(0)) - wp_tot(cb.sum(0))
    rng = np.random.default_rng(seed)
    cnt = rng.multinomial(len(uc), np.full(len(uc), 1 / len(uc)), size=n_boot).astype(float)
    ba = np.einsum("bc,ckj->bkj", cnt, ca)
    bb = np.einsum("bc,ckj->bkj", cnt, cb)
    ds = wp_batch(ba) - wp_batch(bb)
    return base, float(ds.std()), float((ds <= 0).mean()), wp_tot(ca.sum(0)), wp_tot(cb.sum(0)), len(idx), len(uc)


def fmt(name, r):
    return f"{name:<44} d={r[0]*1e4:+7.1f}e-4 sd={r[1]*1e4:5.1f}e-4 z={r[0]/r[1] if r[1] else 0:+5.2f} P(d<=0)={r[2]:.3f}  A={r[3]:.5f} B={r[4]:.5f} n={r[5]} chains={r[6]}"


def routed(SA, SK):
    out = SK.copy()
    out[mc < 3] = SA[mc < 3]
    return out


# pooled out-of-fold analog sums: fold-k rows scored with fold-k models
oof = np.isin(fold, [0, 1])


def pooled(stem):
    X = np.zeros((N, 2, 6))
    for k in (0, 1):
        X[fold == k] = S[f"P{k}_{stem}"][fold == k]
    return X


P = {s: pooled(s) for s in ["inc", "k4", "A", "se", "se_k4", "ee", "ee_k4", "s3", "b4", "ch", "on"]}
P["routed"] = routed(P["A"], P["k4"])
F = {k[2:]: S[k] for k in S if k.startswith("F_")}
F["routed"] = routed(F["A"], F["k4"])

lines = []
def seq_wp(X):  # (n,2,6) -> per-sequence mean-of-targets WP, nan where undefined
    w = X[..., 0]; ok = w > 1e-8; w = np.where(ok, w, 1)
    my, mp = X[..., 1] / w, X[..., 2] / w
    cov = X[..., 5] / w - my * mp
    vy = np.maximum(X[..., 3] / w - my * my, 0); vp = np.maximum(X[..., 4] / w - mp * mp, 0)
    r = np.where(ok & (vy > 1e-16) & (vp > 1e-16), cov / np.sqrt(np.maximum(vy * vp, 1e-300)), 0.0)
    return r.mean(-1), (ok & (vy > 1e-16)).all(-1)
def seqdelta(A, B, idx, n_boot=2000):
    a, ea = seq_wp(A[idx]); b, _ = seq_wp(B[idx]); d = (a - b)[ea]; ch = chain[idx][ea]
    uc, inv = np.unique(ch, return_inverse=True)
    s = np.zeros(len(uc)); np.add.at(s, inv, d); c = np.bincount(inv).astype(float)
    rng = np.random.default_rng(0); cnt = rng.multinomial(len(uc), np.full(len(uc), 1/len(uc)), size=n_boot)
    bs = (cnt @ s) / (cnt @ c)
    return d.mean(), bs.std(), ea.sum()
for nm, A, B, idx in [("FINAL holdout k4-inc", F["k4"], F["inc"], np.flatnonzero(hold)),
                      ("FOLD oof k4-inc", P["k4"], P["inc"], np.flatnonzero(oof)),
                      ("FINAL holdout routed-inc", F["routed"], F["inc"], np.flatnonzero(hold)),
                      ("FOLD oof routed-inc", P["routed"], P["inc"], np.flatnonzero(oof))]:
    r = seqdelta(A, B, idx); print(f"per-sequence metric {nm}: {r[0]*1e4:+.1f}e-4 sd {r[1]*1e4:.1f}e-4 (n={r[2]})")
