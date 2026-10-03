"""Can the tick class (target quantum) be identified causally from inputs? Per-sequence summaries over the
first T rows -> gradient boosting classifier; accuracy vs T."""
import numpy as np, time
import sio, feat
from sklearn.ensemble import HistGradientBoostingClassifier
z = np.load('catalogue.npz')
cols = [f'i0_p{k}' for k in range(22)] + [f'i0_dp{k}' for k in range(4)] + ['i0_v0', 'i0_dv0'] + [f'i1_p{k}' for k in range(11)] + ['i1_dp0']
Ts = (100, 500, 2000, 20000)
def summ(d, T):
    P = np.stack([d[f'i0_p{k}'][:T] for k in range(22)], 1)
    ev = np.r_[True, np.any(np.diff(P, axis=0) != 0, 1)]
    Pe = P[ev]; n = len(Pe)
    _, sid = np.unique(Pe, axis=0, return_inverse=True); sid = sid.ravel()
    rec = [np.mean(sid[k:] == sid[:-k]) if n > k else 0 for k in (2, 3, 4, 6)]
    f = [ev.mean(), len(np.unique(sid)) / max(n, 1), *rec]
    for c in ('i0_dp0', 'i0_dp1', 'i0_dp2', 'i0_dp3'):
        x = d[c][:T][ev]
        u, cn = np.unique(x, return_counts=True)
        f += [len(u) / max(n, 1), cn.max() / n, x.mean(), x.std()]
    for k in range(22):
        x = Pe[:, k]; f += [x.mean(), x.std()]
    P1 = np.stack([d[f'i1_p{k}'][:T] for k in range(11)], 1)
    e1 = np.r_[True, np.any(np.diff(P1, axis=0) != 0, 1)]
    f += [e1.mean()] + list(P1[e1].mean(0))
    dv = d['i0_dv0'][:T]; f += [np.mean(dv != np.median(dv))]
    return f
out = {}
for split in ('train', 'valid'):
    A = z[split]; q = np.round(A[:, 2], 4)
    rng = np.random.default_rng(14)
    keep = np.where(np.isin(q, [0.1254, 0.2413, 0.4685, 0.6569]))[0]
    sel = rng.choice(keep, 1500 if split == 'train' else 600, replace=False)
    X = {T: [] for T in Ts}; y = []
    t0 = time.time()
    for i, j in enumerate(sel):
        d = sio.load(int(A[j, 0]), cols, split)
        for T in Ts: X[T].append(summ(d, T))
        y.append(q[j])
        if i % 300 == 0: print(split, i, time.time() - t0, flush=True)
    out[split] = ({T: np.array(v) for T, v in X.items()}, np.array(y))
classes = np.array([0.1254, 0.2413, 0.4685, 0.6569])
for T in Ts:
    Xtr, ytr = out['train'][0][T], np.searchsorted(classes, out['train'][1])
    Xva, yva = out['valid'][0][T], np.searchsorted(classes, out['valid'][1])
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05).fit(Xtr, ytr)
    pr = clf.predict(Xva)
    cm = np.zeros((4, 4), int)
    for a, b in zip(yva, pr): cm[a, b] += 1
    print(f'T={T:6d} rows: valid accuracy {np.mean(pr == yva):.3f}; confusion (rows=true q, cols=pred)\n', cm)
np.savez('class_id.npz', **{f'{s}_{T}': out[s][0][T] for s in out for T in Ts}, ytr=out['train'][1], yva=out['valid'][1])
