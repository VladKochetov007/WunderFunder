"""Ridge on bank2 (filter bank + event taps + recurrence); targets t0, t1, s=t0+t1, d=t0-t1; group ablations."""
import numpy as np, sys, time, json
import sio, feat, feat2
NTR, NVA = int(sys.argv[1]), int(sys.argv[2])
rng = np.random.default_rng(7)
tr = rng.choice(sio.n_seq('train'), NTR, replace=False)
va = rng.choice(sio.n_seq('valid'), NVA, replace=False)
t0 = time.time(); S = None
for i, rg in enumerate(tr):
    d = sio.load(int(rg), None, 'train')
    F, names = feat2.bank2(d)
    A = np.c_[F[99:], np.ones(len(F) - 99)].astype(np.float64)
    y0, y1 = d['t0'][99:].astype(np.float64), d['t1'][99:].astype(np.float64)
    Y = np.c_[y0, y1, y0 + y1, y0 - y1]
    if S is None: S = {k: [0, 0] for k in ('u', 'w0', 'w1')}
    for k, w in (('u', None), ('w0', np.abs(np.clip(y0, -2, 2))), ('w1', np.abs(np.clip(y1, -2, 2)))):
        Aw = A if w is None else A * w[:, None]
        S[k][0] = S[k][0] + A.T @ Aw; S[k][1] = S[k][1] + Aw.T @ Y
    if i % 25 == 0: print('train', i, time.time() - t0, flush=True)
names = np.array(names); nf = len(names)
kind = np.array(['bank' if '|' in n and n.split('|')[1] in ('lvl', 'band<4', 'band<32', 'band<256') else
                 ('taps' if '|ev-' in n else ('rec' if n.startswith('rec') or n == 'distinct12' else 'clk')) for n in names])
configs = {'bank': np.isin(kind, ['bank', 'clk']), 'bank+taps': np.isin(kind, ['bank', 'clk', 'taps']),
           'bank+rec': np.isin(kind, ['bank', 'clk', 'rec']), 'all': np.ones(nf, bool)}
def solve(M, b, idx, lam=1.0):
    ii = np.r_[np.where(idx)[0], nf]; G = M[np.ix_(ii, ii)]; D = np.diag(G).copy(); D[-1] = 0
    return ii, np.linalg.solve(G + lam * 1e-3 * np.diag(D + 1e-9) + 1e-6 * np.eye(len(ii)), b[ii])
W = {}
for c, idx in configs.items():
    W[(c, 'u')] = solve(S['u'][0], S['u'][1], idx)
    ii, B0 = solve(S['w0'][0], S['w0'][1], idx); _, B1 = solve(S['w1'][0], S['w1'][1], idx)
    W[(c, 'w')] = (ii, np.c_[B0[:, 0], B1[:, 1], B0[:, 2], B0[:, 3]])
P = {k: [] for k in W}; Yv = []; Cls = []
cat = np.load('catalogue.npz')['valid']; qv = dict(zip(cat[:, 0].astype(int), np.round(cat[:, 2], 4)))
for i, rg in enumerate(va):
    d = sio.load(int(rg), None, 'valid')
    F, _ = feat2.bank2(d); m = d['is_scored']
    A = np.c_[F[m], np.ones(m.sum())].astype(np.float64)
    Yv.append(np.c_[d['t0'][m], d['t1'][m]]); Cls.append(np.full(m.sum(), qv[int(rg)]))
    for k, (ii, B) in W.items(): P[k].append((A[:, ii] @ B).astype(np.float32))
Yv = np.concatenate(Yv); Cls = np.concatenate(Cls); res = {}
for k in W:
    p = np.concatenate(P[k])
    r = [feat.wcorr(p[:, 0], Yv[:, 0]), feat.wcorr(p[:, 1], Yv[:, 1])]
    # recombination from s and d heads: t0=(s+d)/2, t1=(s-d)/2
    r_sd = [feat.wcorr((p[:, 2] + p[:, 3]) / 2, Yv[:, 0]), feat.wcorr((p[:, 2] - p[:, 3]) / 2, Yv[:, 1])]
    r_s = np.corrcoef(p[:, 2], Yv.sum(1))[0, 1]; r_d = np.corrcoef(p[:, 3], Yv[:, 0] - Yv[:, 1])[0, 1]
    per = {str(qq): [round(feat.wcorr(p[Cls == qq, 0], Yv[Cls == qq, 0]), 4), round(feat.wcorr(p[Cls == qq, 1], Yv[Cls == qq, 1]), 4)] for qq in np.unique(Cls)}
    res[f"{k[0]}|{k[1]}"] = dict(t0=float(r[0]), t1=float(r[1]), t0_sd=float(r_sd[0]), t1_sd=float(r_sd[1]), corr_s=float(r_s), corr_d=float(r_d), per_class={a: [float(x) for x in b] for a, b in per.items()})
    print(f'{k[0]:10s} {k[1]} t0={r[0]:.4f} t1={r[1]:.4f} mean={np.mean(r):.4f} | via s,d: {np.mean(r_sd):.4f} | corr(s)={r_s:.3f} corr(d)={r_d:.3f} | per-class {per}', flush=True)
json.dump(res, open('ridge2.json', 'w'), indent=1)
np.savez('ridge2_valid_preds.npz', p=np.concatenate(P[('all', 'w')]), y=Yv, cls=Cls)
