# Build a per-row table of raw features + causal state variables + targets for a sample of sequences.
# train: every 4th row after warm-up; valid: all scored rows + every 8th unscored row (flag kept).
import numpy as np, sys
sys.path.insert(0, "research/campaign_2026-10-03/eda/nonlinear")
from nlcommon import *

def ewma(x, hl):
    a = 1 - 0.5 ** (1.0 / hl)
    out = np.empty_like(x, dtype=np.float32); m = 0.0
    for i in range(len(x)):
        m += a * (x[i] - m); out[i] = m
    return out

try:
    import numba
    ewma = numba.njit(cache=False)(ewma)
except Exception:
    pass

P0 = [FEATS.index(f"i0_p{i}") for i in range(22)]
V0 = [FEATS.index(f"i0_v{i}") for i in range(22)]
DV0 = [FEATS.index(f"i0_dv{i}") for i in range(4)]
P1 = [FEATS.index(f"i1_p{i}") for i in range(22)]
DV1 = [FEATS.index(f"i1_dv{i}") for i in range(4)]
SN = ["chg0", "chgb0", "chga0", "trd0", "chg1", "trd1",
      "act0_h10", "act0_h100", "act0_h1000", "trd0_h10", "trd0_h100", "trd0_h1000", "act1_h100", "trd1_h100",
      "tsince_chg0", "tsince_trd0", "absdx0_h10", "absdx0_h100", "step"]

def state(x):
    dx = np.vstack([np.zeros((1, x.shape[1]), np.float32), np.diff(x, axis=0)])
    chgb = (dx[:, P0[:11]] != 0).any(1).astype(np.float32)
    chga = (dx[:, P0[11:]] != 0).any(1).astype(np.float32)
    chg0 = np.maximum(chgb, chga)
    trd0 = (dx[:, DV0] != 0).any(1).astype(np.float32)
    chg1 = (dx[:, P1] != 0).any(1).astype(np.float32)
    trd1 = (dx[:, DV1] != 0).any(1).astype(np.float32)
    def tsince(ind):
        out = np.empty(len(ind), np.float32); c = 1e4
        for i in range(len(ind)):
            c = 0 if ind[i] else c + 1; out[i] = c
        return np.log1p(out)
    absdx = np.abs(dx[:, P0 + V0]).sum(1).astype(np.float32)
    cols = [chg0, chgb, chga, trd0, chg1, trd1,
            ewma(chg0, 10.0), ewma(chg0, 100.0), ewma(chg0, 1000.0), ewma(trd0, 10.0), ewma(trd0, 100.0), ewma(trd0, 1000.0),
            ewma(chg1, 100.0), ewma(trd1, 100.0), tsince(chg0), tsince(trd0), ewma(absdx, 10.0), ewma(absdx, 100.0),
            np.arange(len(x), dtype=np.float32)]
    return np.stack(cols, 1), dx

def build(split, ids, every, every_unscored=8):
    out = {k: [] for k in ["X", "DX", "S", "Y", "K", "U", "G", "SC"]}
    for rg in ids:
        d = load(split, rg)
        x = np.stack([d[c] for c in FEATS], 1)
        st, dx = state(x)
        y = np.stack([d["t0"], d["t1"]], 1)
        u = unit_of(np.concatenate([d["t0"], d["t1"]]))
        n = len(x); r = np.arange(n)
        if split == "valid":
            sc = d["is_scored"]
            keep = (r >= 99) & (sc | (r % every_unscored == 0))
        else:
            sc = np.zeros(n, bool); keep = (r >= 99) & (r % every == 0)
        out["X"].append(x[keep]); out["DX"].append(dx[keep]); out["S"].append(st[keep]); out["Y"].append(y[keep])
        out["K"].append(np.round(y[keep] / u).astype(np.int16)); out["U"].append(np.full(keep.sum(), u, np.float32))
        out["G"].append(np.full(keep.sum(), rg, np.int32)); out["SC"].append(sc[keep])
    return {k: np.concatenate(v) for k, v in out.items()}

if __name__ == "__main__":
    tr = build("train", seq_ids("train", 48, seed=2), every=4)
    np.savez(OUT + "/cache_train48.npz", **tr, SN=np.array(SN))
    print("train", {k: v.shape for k, v in tr.items()})
    va = build("valid", seq_ids("valid", 40, seed=1), every=1)
    np.savez(OUT + "/cache_valid40.npz", **va, SN=np.array(SN))
    print("valid", {k: v.shape for k, v in va.items()}, "scored", va["SC"].sum())
