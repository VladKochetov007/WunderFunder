# Class 0.6569: does the period-4 regime matter? Ridge on i0 features (+ phase features), train on all vs periodic-only train seqs,
# score on valid class-0.6569 scored rows.
import sys
sys.path.insert(0, '/home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-03/eda/puzzle_b/scripts')
from common import *
CL = float(sys.argv[1]) if len(sys.argv) > 1 else 0.6569
I0 = [f'i0_p{k}' for k in range(22)] + [f'i0_dp{k}' for k in range(4)] + [f'i0_dv{k}' for k in range(4)] + [f'i0_v{k}' for k in range(22)]
I1 = [f'i1_p{k}' for k in range(22)] + [f'i1_dp{k}' for k in range(4)] + [f'i1_dv{k}' for k in range(4)] + [f'i1_v{k}' for k in range(22)]
def feats(d):
    X = d[I0 + I1].values.astype(np.float32)
    A = d[I0[:22]].values; ev = np.r_[True, (A[1:] != A[:-1]).any(1)]
    idx = np.arange(len(ev)); last = np.maximum.accumulate(np.where(ev, idx, 0)); since = idx - last
    ph = np.zeros((len(ev), 6), np.float32)
    for k in range(4): ph[:, k] = (since % 4) == k
    ph[:, 4] = np.log1p(since); ph[:, 5] = np.minimum(since, 8) / 8
    # causal periodicity strength: EWM of ev(t)*ev(t-4) minus ev(t)*ev(t-3)
    e = ev.astype(np.float32); c4 = np.r_[np.zeros(4), e[4:] * e[:-4]]; c3 = np.r_[np.zeros(3), e[3:] * e[:-3]]
    per = pd.Series(c4 - c3).ewm(alpha=0.002).mean().values.astype(np.float32)
    rate = pd.Series(e).ewm(alpha=0.01).mean().values.astype(np.float32)
    return X, np.c_[ph, per, rate, per[:, None] * ph[:, :4]], per
def prom4(d):
    A = d[I0[:22]].values; ev = np.r_[False, (A[1:] != A[:-1]).any(1)].astype(float)
    x = ev - ev.mean(); v = (x * x).mean() + 1e-12
    a = [(x[:-l] * x[l:]).mean() / v for l in (3, 4, 5)]
    return a[1] - 0.5 * (a[0] + a[2])
def load(path, rgs, step, scored):
    Xs, Ps, Ys, Ss, PR = [], [], [], [], []
    for rg in rgs:
        d = seq(int(rg), I0 + I1 + ['t0', 't1'] + (['is_scored'] if scored else []), path=path)
        X, Ph, _ = feats(d); sl = slice(99, None, step)
        Xs.append(X[sl]); Ps.append(Ph[sl]); Ys.append(d[['t0', 't1']].values[sl])
        Ss.append(d.is_scored.values[sl] if scored else np.ones(len(X[sl]), bool)); PR.append(np.full(len(X[sl]), prom4(d)))
    return np.concatenate(Xs), np.concatenate(Ps), np.concatenate(Ys), np.concatenate(Ss), np.concatenate(PR)
def wcorr(p, y):
    p = np.clip(p, -2, 2); y = np.clip(y, -2, 2); w = np.abs(y)
    mp = (w * p).sum() / w.sum(); my = (w * y).sum() / w.sum()
    return (w * (p - mp) * (y - my)).sum() / np.sqrt((w * (p - mp) ** 2).sum() * (w * (y - my) ** 2).sum())
def ridge(X, y, lam=1.0):
    mu = X.mean(0); sd = X.std(0) + 1e-6; Xn = (X - mu) / sd
    b = np.linalg.solve(Xn.T @ Xn + lam * len(Xn) * 1e-3 * np.eye(X.shape[1]), Xn.T @ (y - y.mean()))
    return lambda Z: ((Z - mu) / sd) @ b + y.mean()
s = pd.read_csv(f'{OUT}/seqscan_train.csv'); s['cls'] = s.t0_scale.round(4)
v = pd.read_csv(f'{OUT}/seqscan_valid.csv'); v['cls'] = v.t0_scale.round(4)
rng = np.random.default_rng(3)
tr = rng.choice(s[s.cls == CL].rg.values, 40, replace=False)
va = rng.choice(v[v.cls == CL].rg.values, 16, replace=False)
Xt, Pt, Yt, _, PRt = load(TRAIN, tr, 6, False)
Xv, Pv, Yv, Sv, PRv = load(VALID, va, 3, True)
print('train periodic seqs frac', (np.unique(PRt) > 0.05).mean(), 'valid', (np.unique(PRv) > 0.05).mean(), flush=True)
nI0 = len(I0)
for name, sel in (('all_train', np.ones(len(Xt), bool)), ('periodic_train_only', PRt > 0.05), ('nonperiodic_train_only', PRt <= 0.05)):
    for fs, f in (('i0', lambda X, P: X[:, :nI0]), ('i0+i1', lambda X, P: X), ('i0+i1+phase', lambda X, P: np.c_[X, P])):
        out = [name, fs, int(sel.sum())]
        for j in range(2):
            m = ridge(f(Xt[sel], Pt[sel]).astype(np.float64), np.clip(Yt[sel, j], -2, 2))
            p = m(f(Xv, Pv).astype(np.float64))
            out.append(round(wcorr(p[Sv], Yv[Sv, j]), 4))
        print(out, flush=True)
