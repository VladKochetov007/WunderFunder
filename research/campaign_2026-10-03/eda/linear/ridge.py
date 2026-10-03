"""Causal linear (ridge/WLS) models: Gram accumulation on train sequences, pooled WP on valid scored rows.
Blocks: L=levels, D1=first diff, E32/E256=causal EMAs, CM=expanding (sequence-to-date) mean. Grams kept per target class
(q quantum, known in train from the targets) so that pooled and class-specific fits come from one pass."""
import sys, time, numpy as np
sys.path.insert(0, "/home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-03/eda/linear")
from common import *

BLOCKS = ["L", "D1", "E32", "E256", "CM"]
NB = len(FEATS)
QS = np.array([0.1254, 0.2413, 0.4685, 0.6569])

def blocks(X):
    cm = np.cumsum(X, 0, dtype=np.float64) / np.arange(1, len(X) + 1)[:, None]
    return np.concatenate([X, X - lag(X, 1), ema(X, 1 / 32), ema(X, 1 / 256), cm.astype(np.float32)], 1)

def qclass(Y):
    nz = np.abs(Y[Y != 0]); q = nz.min()
    return int(np.argmin(np.abs(QS - q)))

# P(scored | a3) table, 40 quantile bins, from the mask probe (valid seqs seed 3)
def a3_table():
    pf = pfile("valid"); rgs = np.random.default_rng(3).choice(pf.metadata.num_row_groups, 120, replace=False)
    a3 = []; s = []
    for rg in rgs:
        tb = pf.read_row_group(int(rg), columns=["a3", "is_scored", "need_prediction"])
        m = tb.column("need_prediction").to_numpy()
        a3.append(tb.column("a3").to_numpy()[m]); s.append(tb.column("is_scored").to_numpy()[m])
    a3 = np.concatenate(a3); s = np.concatenate(s)
    edges = np.quantile(a3, np.linspace(0, 1, 41)); edges[0] = -np.inf; edges[-1] = np.inf
    b = np.clip(np.searchsorted(edges, a3, side="right") - 1, 0, 39)
    return edges, np.array([s[b == i].mean() for i in range(40)])

def pi_a3(a3, edges, tab):
    return tab[np.clip(np.searchsorted(edges, a3, side="right") - 1, 0, 39)]

if __name__ == "__main__":
    NTR = int(sys.argv[1]); STRIDE = int(sys.argv[2]); tag = sys.argv[3]
    edges, tab = a3_table()
    P = NB * len(BLOCKS) + 1
    G = np.zeros((2, 4, 2, P, P))      # [weighting, class, target]
    b = np.zeros((2, 4, 2, P)); nseq = np.zeros(4)
    pf = pfile("train")
    rgs = np.random.default_rng(11).choice(pf.metadata.num_row_groups, NTR, replace=False)
    t0 = time.time()
    for i, rg in enumerate(rgs):
        X, Y, ex = load(pf, rg)
        c = qclass(Y); nseq[c] += 1
        T = blocks(X)
        sel = np.arange(99 + (i % STRIDE), len(X), STRIDE)
        A = np.c_[T[sel], np.ones(len(sel), np.float32)].astype(np.float64)
        pa = pi_a3(X[sel, 107], edges, tab)
        for k in range(2):
            yc, w = clipw(Y[sel, k].astype(np.float64))
            for wi, ww in enumerate([w, w * pa]):
                Aw = A * ww[:, None]
                G[wi, c, k] += A.T @ Aw; b[wi, c, k] += Aw.T @ yc
        if i % 25 == 0: print(i, round(time.time() - t0, 1), nseq, flush=True)
    np.savez(f"{OUT}/gram_{tag}.npz", G=G, b=b, nseq=nseq, rgs=rgs, edges=edges, tab=tab)
    print("saved", time.time() - t0)
