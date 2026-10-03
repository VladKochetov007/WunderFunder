"""Shared helpers for the linear EDA (causal transforms + weighted-correlation accumulators)."""
import numpy as np
import pyarrow.parquet as pq
from scipy.signal import lfilter

ROOT = "/home/vlad/development/wunder/wnn_connectome_starterpack"
OUT = ROOT + "/research/campaign_2026-10-03/eda/linear"
FEATS = (
    [f"i0_p{i}" for i in range(22)] + [f"i0_v{i}" for i in range(22)]
    + [f"i0_dp{i}" for i in range(4)] + [f"i0_dv{i}" for i in range(4)]
    + [f"i1_p{i}" for i in range(22)] + [f"i1_v{i}" for i in range(22)]
    + [f"i1_dp{i}" for i in range(4)] + [f"i1_dv{i}" for i in range(4)]
    + [f"a{i}" for i in range(8)]
)


def pfile(split):
    return pq.ParquetFile(f"{ROOT}/datasets/{split}.parquet")


def load(pf, rg, extra=()):
    cols = list(extra) + FEATS + ["t0", "t1"]
    tb = pf.read_row_group(int(rg), columns=cols)
    X = np.stack([tb.column(n).to_numpy() for n in FEATS], 1).astype(np.float32)
    Y = np.stack([tb.column("t0").to_numpy(), tb.column("t1").to_numpy()], 1).astype(np.float32)
    ex = {n: tb.column(n).to_numpy() for n in extra}
    return X, Y, ex


def ema(x, alpha):
    """Causal EMA along axis 0, initialised at the first row."""
    zi = (1 - alpha) * x[:1]
    y, _ = lfilter([alpha], [1, -(1 - alpha)], x, axis=0, zi=zi)
    return y.astype(np.float32)


def lag(x, k):
    """x(t-k) with the first row repeated for t<k (causal)."""
    if k == 0:
        return x
    out = np.empty_like(x)
    out[k:] = x[:-k]
    out[:k] = x[:1]
    return out


def clipw(y):
    yc = np.clip(y, -2, 2)
    return yc, np.abs(yc)


def wcorr(p, y):
    """Official metric for one target: weights |clip(y)|, both clipped."""
    yc, w = clipw(y)
    pc = np.clip(p, -2, 2)
    sw = w.sum()
    my = (w * yc).sum() / sw; mp = (w * pc).sum() / sw
    cov = (w * (yc - my) * (pc - mp)).sum()
    return cov / np.sqrt((w * (yc - my) ** 2).sum() * (w * (pc - mp) ** 2).sum())


class WAcc:
    """Accumulates weighted moments of many features vs one target for pooled weighted corr."""
    def __init__(self, nf):
        self.sw = 0.0; self.sx = np.zeros(nf); self.sxx = np.zeros(nf); self.sxy = np.zeros(nf)
        self.sy = 0.0; self.syy = 0.0

    def add(self, X, y, w):
        X = X.astype(np.float64); y = y.astype(np.float64)
        self.sw += w.sum(); self.sx += w @ X; self.sxx += w @ (X * X); self.sxy += (w * y) @ X
        self.sy += (w * y).sum(); self.syy += (w * y * y).sum()

    def corr(self):
        mx = self.sx / self.sw; my = self.sy / self.sw
        vx = self.sxx / self.sw - mx ** 2; vy = self.syy / self.sw - my ** 2
        return (self.sxy / self.sw - mx * my) / np.sqrt(vx * vy + 1e-30)
