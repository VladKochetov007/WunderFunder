import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from util import *
d = np.load("scan1_train.npz"); qa = np.round(np.minimum(d["q0"], d["q1"]), 5)
L = 3000
clocks = ["row", "i0event", "pricechg"]
fig, ax = plt.subplots(4, 3, figsize=(20, 14))
for ci, cls in enumerate([0.12543, 0.24133, 0.46854, 0.65687]):
    rgs = np.flatnonzero(qa == cls)[:15]
    acc = {c: np.zeros(L+1) for c in clocks}; var = {c: 0. for c in clocks}
    for rg in rgs:
        df = load("train", rg)
        X = df[[c for c in df.columns if c.startswith("i0_")]].values
        P = df[[f"i0_p{i}" for i in range(22)]].values
        sel = {"row": np.arange(len(df)), "i0event": np.flatnonzero(np.r_[True, np.any(X[1:] != X[:-1], axis=1)]),
               "pricechg": np.flatnonzero(np.r_[True, np.any(P[1:] != P[:-1], axis=1)])}
        for c in clocks:
            k = np.round(df["t0"].values[sel[c]]/cls) - np.round(df["t1"].values[sel[c]]/cls)
            dk = np.diff(k); dk = dk - dk.mean(); n = len(dk)
            f = np.fft.rfft(dk, 2*n); ac = np.fft.irfft(f*np.conj(f), 2*n)[:L+1]
            if len(ac) < L+1: ac = np.r_[ac, np.zeros(L+1-len(ac))]
            acc[c] += ac; var[c] += (dk*dk).sum()
    for j, c in enumerate(clocks):
        a = acc[c]/var[c]
        ax[ci, j].plot(np.arange(1, L+1), a[1:], lw=.6); ax[ci, j].set_ylim(-0.3, 0.1)
        ax[ci, j].set_title(f"q={cls} ACF Δ(k0-k1), clock={c}")
        top = np.argsort(a[11:])[:5] + 11
        print(cls, c, "lag1", a[1].round(3), "most negative lags >10", top, a[top].round(3))
plt.tight_layout(); plt.savefig("fig04_acf_dtarget_clocks.png", dpi=55)
