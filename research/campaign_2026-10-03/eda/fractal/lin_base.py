"""Pooled |y|-weighted linear regression of targets on raw feature groups (streaming normal equations)."""
import sys; sys.path.insert(0, '/home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-03/eda/fractal')
import numpy as np, fx
F = fx.FEAT; n = len(F)
XtX = np.zeros((2, n + 1, n + 1)); Xty = np.zeros((2, n + 1))
for g in range(0, 10607, 212):  # 51 train seqs
    d = fx.read_seq(fx.TRAIN, g, F + ['t0', 't1'])
    X = np.c_[np.stack([d[c] for c in F], 1)[99:].astype(np.float64), np.ones(19901)]
    for j, tc in enumerate(['t0', 't1']):
        y = d[tc][99:].astype(np.float64); w = np.abs(np.clip(y, -2, 2))
        XtX[j] += (X * w[:, None]).T @ X; Xty[j] += (X * w[:, None]).T @ y
groups = {'i0_p': [fx.IDX[f'i0_p{k}'] for k in range(22)], 'i0_p+v': [fx.IDX[f'i0_{g}{k}'] for g in 'pv' for k in range(22)],
          'i0 all': list(range(52)), 'i1 all': list(range(52, 104)), 'a': list(range(104, 112)), 'all': list(range(112))}
B = {}
for k, cols in groups.items():
    c = cols + [n]; B[k] = [np.linalg.solve(XtX[j][np.ix_(c, c)] + 1e-3 * np.eye(len(c)), Xty[j][c]) for j in range(2)]
np.savez(fx.OUT + 'lin_base_B.npz', all0=B['all'][0], all1=B['all'][1], i0p0=B['i0_p'][0], i0p1=B['i0_p'][1])
P = {k: [[], []] for k in groups}; Y = [[], []]
for g in range(0, 1873, 37):
    d = fx.read_seq(fx.VALID, g, F + ['t0', 't1', 'is_scored']); m = d['is_scored']
    X = np.c_[np.stack([d[c] for c in F], 1).astype(np.float64), np.ones(20000)][m]
    for j, tc in enumerate(['t0', 't1']):
        Y[j].append(d[tc][m])
        for k, cols in groups.items(): P[k][j].append(X[:, cols + [n]] @ B[k][j])
for k in groups:
    r = [fx.wp(np.concatenate(Y[j]), np.concatenate(P[k][j])) for j in range(2)]
    print('%-8s WP t0 %.4f t1 %.4f mean %.4f' % (k, r[0], r[1], np.mean(r)))
