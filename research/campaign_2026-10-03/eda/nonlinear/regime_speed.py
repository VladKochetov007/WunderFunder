# Regime persistence: ACF of slow state variables and of |target|, burstiness of book-change events.
import numpy as np, sys
sys.path.insert(0, "research/campaign_2026-10-03/eda/nonlinear")
from nlcommon import *
from build_state import state
lags = np.unique(np.round(np.logspace(0, np.log10(8000), 30)).astype(int))
ids = seq_ids("valid", 40, seed=1)[::3]
names = ["a2", "a3", "a6", "act0_h100", "abs_k0", "chg0", "trd0", "absd1_p"]
acf = {n: [] for n in names}; cv = []; units = []
for rg in ids:
    d = load("valid", rg)
    x = np.stack([d[c] for c in FEATS], 1); st, dx = state(x)
    u = unit_of(d["t0"]); units.append(u)
    ser = {"a2": d["a2"], "a3": d["a3"], "a6": d["a6"], "act0_h100": st[:, 7], "abs_k0": np.abs(d["t0"] / u),
           "chg0": st[:, 0], "trd0": st[:, 3], "absd1_p": np.abs(dx[:, :22]).sum(1)}
    for n, s in ser.items():
        s = s[99:].astype(np.float64); s = s - s.mean(); v = (s * s).mean()
        acf[n].append([np.mean(s[:-L] * s[L:]) / v if v > 0 else np.nan for L in lags])
    ev = np.where(st[:, 0] > 0)[0]; gaps = np.diff(ev)
    cv.append(gaps.std() / gaps.mean())
np.savez(OUT + "/regime_speed.npz", lags=lags, units=np.array(units), cv=np.array(cv), **{n: np.array(v) for n, v in acf.items()})
print("units", np.round(units, 3)); print("CV of inter-change gaps", np.round(cv, 2))
for n in names:
    a = np.nanmean(acf[n], 0)
    half = lags[np.argmax(a < 0.5)] if (a < 0.5).any() else ">8000"
    print(f"{n:10s} acf@1 {a[0]:.3f} @10 {a[np.searchsorted(lags,10)]:.3f} @100 {a[np.searchsorted(lags,100)]:.3f} @1000 {a[np.searchsorted(lags,1000)]:.3f} @5000 {a[np.searchsorted(lags,5000)]:.3f}  lag(acf<0.5)={half}")
