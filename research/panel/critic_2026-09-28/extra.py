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
L = lines.append
h189 = np.flatnonzero(hold & oof)
L(fmt("FINAL k4-inc on 189 holdout seqs in folds0/1", delta(F["k4"], F["inc"], h189)))
L(fmt("FOLD  k4-inc on same 189", delta(P["k4"], P["inc"], h189)))
h179 = np.flatnonzero(hold & ~oof)
L(fmt("FINAL k4-inc on 179 holdout seqs in folds2/3", delta(F["k4"], F["inc"], h179)))
# difference-in-differences on the 189: (P_k4-P_inc) - (F_k4-F_inc), paired chain bootstrap
def did(idx, n_boot=2000):
    ch = chain[idx]; uc, inv = np.unique(ch, return_inverse=True)
    arrs = []
    for X in (P["k4"], P["inc"], F["k4"], F["inc"]):
        c = np.zeros((len(uc), 2, 6)); np.add.at(c, inv, X[idx]); arrs.append(c)
    base = (wp_tot(arrs[0].sum(0)) - wp_tot(arrs[1].sum(0))) - (wp_tot(arrs[2].sum(0)) - wp_tot(arrs[3].sum(0)))
    rng = np.random.default_rng(1)
    cnt = rng.multinomial(len(uc), np.full(len(uc), 1 / len(uc)), size=n_boot).astype(float)
    b = [wp_batch(np.einsum("bc,ckj->bkj", cnt, a)) for a in arrs]
    ds = (b[0] - b[1]) - (b[2] - b[3])
    return base, ds.std()
r = did(h189); L(f"diff-in-diff (fold lever - final lever) on 189: {r[0]*1e4:+.1f}e-4 sd {r[1]*1e4:.1f}e-4")
print("isolated holdout class mix", np.bincount(mc[hold & ~hold_nb], minlength=4))
# per target
def wpk(t, k):
    w, wy, wp_, wyy, wpp, wyp = t[k]
    my, mp = wy / w, wp_ / w
    return (wyp / w - my * mp) / np.sqrt((wyy / w - my * my) * (wpp / w - mp * mp))
for nm, A, B, idx in [("FINAL holdout", F["k4"], F["inc"], np.flatnonzero(hold)), ("FOLD oof", P["k4"], P["inc"], np.flatnonzero(oof))]:
    a, b = A[idx].sum(0), B[idx].sum(0)
    L(f"{nm} per-target delta: t0 {1e4*(wpk(a,0)-wpk(b,0)):+.1f}e-4 t1 {1e4*(wpk(a,1)-wpk(b,1)):+.1f}e-4")
# in-sample gap for fold analog: fold-0 models on folds 1-3 (in-sample) vs final in-sample dev
ins = np.flatnonzero(fold != 0)
L(fmt("[in-sample] P0 k4-inc on folds1-3", delta(S["P0_k4"], S["P0_inc"], ins)))
# lambda sensitivity is not available (scored rows only); leave-one-class-out on fold analog
for c in range(4):
    L(fmt(f"P k4-inc without class {c}", delta(P["k4"], P["inc"], np.flatnonzero(oof & (mc != c)))))
# members-of-final on OOF-for-them rows? Final models saw folds 0/1 dev rows; only holdout is OOF.
# single-member gaps, holdout vs fold, on the same 189 seqs
L(fmt("FINAL ch - s3g on 189", delta(F["ch"], F["s3g"], h189)))
L(fmt("FOLD  ch - s3  on 189", delta(P["ch"], P["s3"], h189)))
L(fmt("FINAL on - s3g on 189", delta(F["on"], F["s3g"], h189)))
L(fmt("FOLD  on - s3  on 189", delta(P["on"], P["s3"], h189)))
print("\n".join(lines))
