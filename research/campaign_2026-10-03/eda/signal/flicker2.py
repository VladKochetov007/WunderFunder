"""Is the cycle predictable? P(next state == state k events ago | last m events obeyed period k). Per side."""
import numpy as np
import sio, feat
rng = np.random.default_rng(10)
idx = rng.choice(sio.n_seq(), 60, replace=False)
sides = {'all': list(range(22)), 'A(p0-10)': list(range(11)), 'B(p11-21)': list(range(11, 22))}
res = {}
for rg in idx:
    d = sio.load(int(rg))
    P = np.stack([d[f'i0_p{k}'] for k in range(22)], 1)
    e0p, _, _ = feat.clocks(d)
    for sname, cols in sides.items():
        Ps = P[:, cols]
        evs = np.r_[True, np.any(np.diff(Ps, axis=0) != 0, 1)]
        Pe = Ps[evs]
        n = len(Pe)
        if n < 200: continue
        # integer state ids
        _, sid = np.unique(Pe, axis=0, return_inverse=True)
        sid = sid.ravel()
        for k in (2, 3, 4, 6):
            eq = np.r_[np.zeros(k, bool), sid[k:] == sid[:-k]]
            for m in (0, 1, 2, 4, 8):
                # condition: eq at n-1..n-m all true
                if m == 0:
                    cond = np.ones(n, bool)
                else:
                    c = np.ones(n, bool)
                    for j in range(1, m + 1):
                        c &= np.r_[np.zeros(j, bool), eq[:-j]]
                    cond = c
                key = (sname, k, m)
                a = res.setdefault(key, [0, 0])
                a[0] += (eq & cond).sum(); a[1] += cond.sum()
        # number of distinct states and entropy rate proxy
        res.setdefault((sname, 'nstates'), []).append(len(np.unique(sid)) / n)
for sname in sides:
    print(sname, 'distinct-state fraction per event (median):', np.median(res[(sname, 'nstates')]).round(3))
    for k in (2, 3, 4, 6):
        print('   period', k, ' '.join(f'm={m}: P={res[(sname, k, m)][0] / max(res[(sname, k, m)][1], 1):.3f} (n={res[(sname, k, m)][1]})' for m in (0, 1, 2, 4, 8)))
