"""Exact state recurrences (flicker): P(state_n == state_{n-k}) in event time; row-clock periodicity of event indicator."""
import numpy as np
import sio, feat
rng = np.random.default_rng(9)
idx = rng.choice(sio.n_seq(), 60, replace=False)
K = 13
rec = np.zeros(K); rec_t = np.zeros(K); n_ev = 0
acf_ind = np.zeros(33); nseq = 0
per_seq = []
for rg in idx:
    d = sio.load(int(rg))
    P = np.stack([d[f'i0_p{k}'] for k in range(22)], 1)
    e0p, e0, e1 = feat.clocks(d)
    ev = np.where(e0p)[0]
    Pe = P[ev]; te = d['t0'][ev]
    n = len(ev)
    if n < 100: continue
    r = np.array([np.mean(np.all(Pe[k:] == Pe[:-k], 1)) for k in range(1, K + 1)])
    rec += r * n; n_ev += n
    rec_t += np.array([np.mean(te[k:] == te[:-k]) for k in range(1, K + 1)]) * n
    per_seq.append(r)
    x = e0p.astype(float) - e0p.mean()
    acf_ind += np.array([np.mean(x[k:] * x[:len(x) - k]) for k in range(33)]) / x.var(); nseq += 1
print('P(P_n == P_{n-k}), k=1..13:', ' '.join(f'{k + 1}:{v:.3f}' for k, v in enumerate(rec / n_ev)))
print('P(t0_n == t0_{n-k}):        ', ' '.join(f'{k + 1}:{v:.3f}' for k, v in enumerate(rec_t / n_ev)))
print('row ACF of i0-price-event indicator:', ' '.join(f'{k}:{v:.3f}' for k, v in enumerate(acf_ind / nseq) if k > 0))
R = np.array(per_seq)
print('per-seq recurrence at k=2: quantiles', np.quantile(R[:, 1], [.1, .5, .9, .99]).round(3), ' k=3', np.quantile(R[:, 2], [.1, .5, .9]).round(3), ' k=6', np.quantile(R[:, 5], [.1, .5, .9]).round(3))
