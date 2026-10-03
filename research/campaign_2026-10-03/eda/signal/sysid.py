"""System identification (analysis only, uses FUTURE inputs on purpose): regress t(n) on x(n+h)-x(n)
for input columns x; R^2 vs h reveals the horizon and whether the target's underlying is spanned by inputs."""
import numpy as np, sys
import sio

clock = sys.argv[1] if len(sys.argv) > 1 else 'ev'
cols_all = sio.columns()
grp = [c for c in cols_all if c.startswith('i0_') or c.startswith('i1_')]
pc = [f'i0_p{k}' for k in range(22)]
rng = np.random.default_rng(5)
idx = rng.choice(sio.n_seq(), 40, replace=False)
H = [1, 2, 3, 5, 8, 12, 16, 24, 32, 48, 64, 96, 128] if clock == 'ev' else [1, 5, 10, 25, 50, 100, 200, 400, 800, 1600]
seqs = []
for rg in idx:
    d = sio.load(rg, grp + ['t0', 't1'])
    X = np.stack([d[c] for c in grp], 1)
    if clock == 'ev':
        P = np.stack([d[c] for c in pc], 1)
        ev = np.r_[True, np.any(np.diff(P, axis=0) != 0, 1)]
        ev[:99] = False
        sel = np.where(ev)[0]
    else:
        sel = np.arange(99, 20000)
    seqs.append((X[sel], d['t0'][sel], d['t1'][sel]))
tr, te = seqs[:30], seqs[30:]
for h in H:
    def build(S):
        A, Y = [], []
        for X, y0, y1 in S:
            if len(X) <= h + 10: continue
            F = X[h:] - X[:-h]
            A.append(np.c_[F, X[:-h]]); Y.append(np.c_[y0[:-h], y1[:-h]])
        return np.concatenate(A).astype(np.float64), np.concatenate(Y).astype(np.float64)
    A, Y = build(tr); At, Yt = build(te)
    mu, sd = A.mean(0), A.std(0) + 1e-9
    A = (A - mu) / sd; At = (At - mu) / sd
    nf = A.shape[1] // 2
    res = []
    for use in (slice(0, nf), slice(0, 2 * nf)):
        G = A[:, use].T @ A[:, use] + 1e-1 * np.eye(A[:, use].shape[1])
        W = np.linalg.solve(G, A[:, use].T @ Y)
        P_ = At[:, use] @ W
        r = [np.corrcoef(P_[:, j], Yt[:, j])[0, 1] for j in range(2)]
        res += r
    print(f'h={h:5d}  corr(t0) diff-only={res[0]:.3f} diff+level={res[2]:.3f} | corr(t1) diff-only={res[1]:.3f} diff+level={res[3]:.3f}', flush=True)
