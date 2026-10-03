import numpy as np, pandas as pd
from util import *
np.set_printoptions(linewidth=220, precision=3, suppress=True)
d = np.load("scan1_train.npz"); qa = np.round(np.minimum(d["q0"], d["q1"]), 5)
def i0_events(df):
    X = df[[c for c in df.columns if c.startswith("i0_")]].values
    return np.r_[True, np.any(X[1:] != X[:-1], axis=1)]
lags = np.array([1,2,5,10,20,50,100,200,300,500,700,1000,1500,2000,3000])
res = {}
for cls in [0.12543, 0.24133, 0.46854, 0.65687]:
    rgs = np.flatnonzero(qa == cls)[:12]
    acs_d, acs_s, acs_row = [], [], []
    for rg in rgs:
        df = load("train", rg)
        ev = i0_events(df)
        k0 = df.t0.values[ev]/cls; k1 = df.t1.values[ev]/cls
        dd = k0 - k1; ss = k0 + k1
        def acf(x, L):
            x = x - x.mean(); v = (x*x).mean()
            return np.array([(x[:-l]*x[l:]).mean()/v if l < len(x) else np.nan for l in L])
        acs_d.append(acf(dd, lags)); acs_s.append(acf(ss, lags)); acs_row.append(acf((df.t0.values - df.t1.values), lags))
    print(cls, "nev mean", "\n  ACF (t0-t1) in i0-event lags", lags, "\n ", np.nanmean(acs_d,0), "\n  ACF (t0+t1) event lags\n ", np.nanmean(acs_s,0), "\n  ACF (t0-t1) row lags\n ", np.nanmean(acs_row,0))
