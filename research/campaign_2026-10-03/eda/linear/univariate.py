"""Pooled |y|-weighted correlation of every feature x causal transform with t0/t1 on scored valid rows,
per-sequence correlations (stability), scored-vs-unscored contrasts and per-sequence summary stats."""
import sys, time, numpy as np
sys.path.insert(0, "/home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-03/eda/linear")
from common import *

TR = ["L", "D1", "D8", "DE4", "DE32", "DE256", "E32"]
def transforms(X):
    e4, e32, e256 = ema(X, 1 / 4), ema(X, 1 / 32), ema(X, 1 / 256)
    return [X, X - lag(X, 1), X - lag(X, 8), X - e4, X - e32, X - e256, e32]

pf = pfile("valid")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
rgs = np.random.default_rng(42).choice(pf.metadata.num_row_groups, N, replace=False)
nf = len(FEATS) * len(TR)
acc = [WAcc(nf), WAcc(nf)]
accu = [WAcc(nf), WAcc(nf)]  # unweighted, all need_prediction rows
per = np.zeros((N, 2, nf), np.float32)
sc_mean = np.zeros(nf); un_mean = np.zeros(nf); all_sq = np.zeros(nf); n_sc = 0; n_un = 0
seqstat = {k: [] for k in ["q", "std0", "std1", "frac_sc", "corr01", "mabs0_sc", "mabs1_sc", "fmean", "fstd", "fchg"]}
t0 = time.time()
for i, rg in enumerate(rgs):
    X, Y, ex = load(pf, rg, extra=["is_scored", "need_prediction"])
    sc = ex["is_scored"]; npred = ex["need_prediction"]
    T = np.concatenate(transforms(X), 1)
    for k in range(2):
        yc, w = clipw(Y[:, k])
        acc[k].add(T[sc], yc[sc], w[sc])
        accu[k].add(T[npred], yc[npred], np.ones(npred.sum()))
        # per-seq weighted corr
        a = WAcc(nf); a.add(T[sc], yc[sc], w[sc]); per[i, k] = a.corr()
    Tm = T[npred].astype(np.float64); scn = sc[npred]
    sc_mean += Tm[scn].sum(0); un_mean += Tm[~scn].sum(0); all_sq += (Tm ** 2).sum(0); n_sc += scn.sum(); n_un += (~scn).sum()
    nz = np.abs(Y[Y != 0]); q = nz.min() if len(nz) else np.nan
    seqstat["q"].append(q); seqstat["std0"].append(Y[:, 0].std()); seqstat["std1"].append(Y[:, 1].std())
    seqstat["frac_sc"].append(sc.mean()); seqstat["corr01"].append(np.corrcoef(Y[:, 0], Y[:, 1])[0, 1])
    seqstat["mabs0_sc"].append(np.abs(np.clip(Y[sc, 0], -2, 2)).mean()); seqstat["mabs1_sc"].append(np.abs(np.clip(Y[sc, 1], -2, 2)).mean())
    seqstat["fmean"].append(X.mean(0)); seqstat["fstd"].append(X.std(0)); seqstat["fchg"].append((np.diff(X, axis=0) != 0).mean(0))
    if i % 50 == 0: print(i, time.time() - t0, flush=True)
n_all = n_sc + n_un
mu = (sc_mean + un_mean) / n_all; sd = np.sqrt(all_sq / n_all - mu ** 2)
smd = (sc_mean / n_sc - un_mean / n_un) / (sd + 1e-12)
np.savez(OUT + "/univariate.npz", TR=np.array(TR), FEATS=np.array(FEATS), rgs=rgs,
         wc0=acc[0].corr(), wc1=acc[1].corr(), uc0=accu[0].corr(), uc1=accu[1].corr(), per=per, smd=smd,
         **{k: np.array(v) for k, v in seqstat.items()})
print("done", time.time() - t0)
