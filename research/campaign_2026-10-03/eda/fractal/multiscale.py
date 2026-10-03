"""Cheap test: does causal multi-scale structure add to a current-state linear model?
Streaming |y|-weighted normal equations over train seqs; nested feature blocks solved from one X'WX; WP on valid seqs (is_scored)."""
import sys; sys.path.insert(0, '/home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-03/eda/fractal')
import numpy as np, fx
from numba import njit
F = fx.FEAT; P0 = [fx.IDX[f'i0_p{k}'] for k in range(22)]
B = np.load(fx.OUT + 'lin_base_B.npz'); b0, b1 = B['all0'], B['all1']
SP = [4, 16, 64, 256, 1024]; SA = [16, 64, 256, 1024, 4096]


@njit(cache=True)
def ema_cols(X, spans):
    n, m = X.shape; out = np.empty((n, m * len(spans)))
    for si in range(len(spans)):
        a = 2.0 / (spans[si] + 1.0)
        for j in range(m):
            v = X[0, j]
            for i in range(n):
                v += a * (X[i, j] - v); out[i, si * m + j] = X[i, j] - v
    return out


def build(d):
    X = np.stack([d[c] for c in F], 1).astype(np.float64); n = len(X)
    ev = np.r_[0.0, np.any(np.diff(X[:, P0], axis=0) != 0, axis=1).astype(np.float64)]
    s0 = X @ b0[:-1] + b0[-1]; s1 = X @ b1[:-1] + b1[-1]
    ds = np.r_[0.0, np.abs(np.diff(s0))] + np.r_[0.0, np.abs(np.diff(s1))]
    blocks = {'raw': X, 'ema': ema_cols(X, np.array(SP, np.float64))}
    rate = np.stack([fx.ema(ev, s) for s in SA], 1); vol = np.stack([fx.ema(ds, s) for s in SA], 1)
    blocks['act'] = np.c_[rate, np.log(vol + 1e-3)]
    vn = vol / 0.05  # fixed global scale (a per-sequence mean would look ahead)
    blocks['inter'] = np.c_[s0[:, None] * vn, s1[:, None] * vn, s0[:, None] * rate, s1[:, None] * rate]
    # local variance ratio of the dominant price-like direction (pc1 proxy), k=8, EMA span 512
    pc = X[:, P0] @ U
    d1 = np.r_[0.0, np.diff(pc)]; d8 = np.r_[np.zeros(8), pc[8:] - pc[:-8]]
    vr = fx.ema(d8 ** 2, 512) / (8 * fx.ema(d1 ** 2, 512) + 1e-9)
    dev0 = s0 - fx.ema(s0, 64); dev1 = s1 - fx.ema(s1, 64)
    blocks['vr'] = np.c_[vr, vr * dev0, vr * dev1, vr * s0, vr * s1]
    return blocks


# pc1 direction of price-like increments from one train seq
d = fx.read_seq(fx.TRAIN, 5, [F[j] for j in P0]); Pm = np.stack([d[F[j]] for j in P0], 1).astype(np.float64)
dP = np.diff(Pm, axis=0); U = np.linalg.svd(dP[np.any(dP != 0, 1)], full_matrices=False)[2][0]
ORDER = ['raw', 'ema', 'act', 'inter', 'vr']
XtX = None; Xty = None; sl = {}
for gi, g in enumerate(range(3, 10607, 208)):
    d = fx.read_seq(fx.TRAIN, g, F + ['t0', 't1']); bl = build(d)
    if XtX is None:
        o = 0
        for k in ORDER: sl[k] = list(range(o, o + bl[k].shape[1])); o += bl[k].shape[1]
        m = o + 1; XtX = np.zeros((2, m, m)); Xty = np.zeros((2, m))
    A = np.c_[np.concatenate([bl[k] for k in ORDER], 1), np.ones(20000)][99:]
    for j, tc in enumerate(['t0', 't1']):
        y = d[tc][99:].astype(np.float64); w = np.abs(np.clip(y, -2, 2)); Aw = A * w[:, None]
        XtX[j] += Aw.T @ A; Xty[j] += Aw.T @ y
    print('train', gi, flush=True)
m = XtX.shape[1]; one = [m - 1]
models = {'M0 raw': ['raw'], 'M1 raw+ema(4..1024)': ['raw', 'ema'], 'M2 +act/vol': ['raw', 'ema', 'act'],
          'M3 +scale interactions': ['raw', 'ema', 'act', 'inter'], 'M4 +local VR': ['raw', 'ema', 'act', 'inter', 'vr'],
          'Mx raw+act+inter (no ema)': ['raw', 'act', 'inter']}
for si, s in enumerate(SP):  # single-scale ablations
    models[f'S raw+ema{s}'] = ('raw', si)
coef = {}
for name, spec in models.items():
    if isinstance(spec, tuple):
        si = spec[1]; cols = sl['raw'] + [sl['ema'][0] + si * 112 + j for j in range(112)]
    else:
        cols = sum([sl[k] for k in spec], [])
    c = cols + one
    coef[name] = (c, [np.linalg.solve(XtX[j][np.ix_(c, c)] + 1e-1 * np.eye(len(c)), Xty[j][c]) for j in range(2)])
Y = [[], []]; Pr = {k: [[], []] for k in models}
for gi, g in enumerate(range(5, 1873, 38)):
    d = fx.read_seq(fx.VALID, g, F + ['t0', 't1', 'is_scored']); bl = build(d); msk = d['is_scored']
    A = np.c_[np.concatenate([bl[k] for k in ORDER], 1), np.ones(20000)][msk]
    for j, tc in enumerate(['t0', 't1']):
        Y[j].append(d[tc][msk])
        for k, (c, bb) in coef.items(): Pr[k][j].append(A[:, c] @ bb[j])
res = {}
for k in models:
    r = [fx.wp(np.concatenate(Y[j]), np.concatenate(Pr[k][j])) for j in range(2)]; res[k] = r
    print('%-30s WP t0 %.4f t1 %.4f mean %.4f' % (k, r[0], r[1], np.mean(r)), flush=True)
np.savez(fx.OUT + 'multiscale.npz', names=list(res), wp=np.array([res[k] for k in res]), SP=SP)
