"""Linear SNR map: ridge on the causal filter bank; ablate bands, groups, clocks. Pooled |y|-weighted corr on valid."""
import numpy as np, sys, time, json
import sio, feat

clock = sys.argv[1] if len(sys.argv) > 1 else 'ev'
NTR, NVA = int(sys.argv[2]) if len(sys.argv) > 2 else 150, int(sys.argv[3]) if len(sys.argv) > 3 else 100
rng = np.random.default_rng(7)
tr = rng.choice(sio.n_seq('train'), NTR, replace=False)
va = rng.choice(sio.n_seq('valid'), NVA, replace=False)
cols_tr = sio.columns('train')
t0 = time.time()
S = None
for i, rg in enumerate(tr):
    d = sio.load(int(rg), None, 'train')
    F, names = feat.bank(d, clock=clock)
    F = F[99:].astype(np.float64); Y = np.c_[d['t0'][99:], d['t1'][99:]].astype(np.float64)
    A = np.c_[F, np.ones(len(F))]
    for wname, w in (('u', np.ones(len(F))), ('w0', np.abs(np.clip(Y[:, 0], -2, 2))), ('w1', np.abs(np.clip(Y[:, 1], -2, 2)))):
        Aw = A * w[:, None]
        if S is None: S = {}
        S.setdefault(wname, [0, 0])
        S[wname][0] = S[wname][0] + A.T @ Aw
        S[wname][1] = S[wname][1] + Aw.T @ Y
    if i % 25 == 0: print('train', i, time.time() - t0, flush=True)
names = np.array(names)
nf = len(names)
band = np.array([n.split('|')[1] if '|' in n else 'clk' for n in names])
grp = np.array(['clk'] * nf, dtype=object)
for g, cs in feat.GROUPS.items():
    for j, n in enumerate(names):
        if n.split('|')[0] in cs: grp[j] = g
configs = {
    'clock_only': band == 'clk',
    'lvl': (band == 'lvl') | (band == 'clk'),
    'lvl+b4': np.isin(band, ['lvl', 'clk', 'band<4']),
    'lvl+b4+b32': np.isin(band, ['lvl', 'clk', 'band<4', 'band<32']),
    'all': np.ones(nf, bool),
    'bands_only(no lvl)': band != 'lvl',
}
for g in feat.GROUPS:
    configs[f'only_{g}'] = (grp == g) | (band == 'clk')
    configs[f'drop_{g}'] = (grp != g)


def solve(M, b, idx, lam=1.0):
    ii = np.r_[np.where(idx)[0], nf]  # always intercept
    G = M[np.ix_(ii, ii)]
    # scale-aware ridge on the diagonal (not the intercept)
    D = np.diag(G).copy(); D[-1] = 0
    return ii, np.linalg.solve(G + lam * 1e-3 * np.diag(D + 1e-9) + 1e-6 * np.eye(len(ii)), b[ii])


W = {}
for cname, idx in configs.items():
    for wname in ('u', 'w'):
        if wname == 'u':
            ii, B = solve(S['u'][0], S['u'][1], idx)
            W[(cname, 'u')] = (ii, B)
        else:
            ii, B0 = solve(S['w0'][0], S['w0'][1][:, [0]], idx)
            _, B1 = solve(S['w1'][0], S['w1'][1][:, [1]], idx)
            W[(cname, 'w')] = (ii, np.c_[B0, B1])
P = {k: [] for k in W}; Yv = []
for i, rg in enumerate(va):
    d = sio.load(int(rg), None, 'valid')
    F, _ = feat.bank(d, clock=clock)
    m = d['is_scored']
    A = np.c_[F[m], np.ones(m.sum())].astype(np.float64)
    Yv.append(np.c_[d['t0'][m], d['t1'][m]])
    for k, (ii, B) in W.items():
        P[k].append((A[:, ii] @ B).astype(np.float32))
Yv = np.concatenate(Yv)
res = {}
for k in W:
    p = np.concatenate(P[k])
    r0, r1 = feat.wcorr(p[:, 0], Yv[:, 0]), feat.wcorr(p[:, 1], Yv[:, 1])
    res[f"{k[0]}|{k[1]}"] = (float(r0), float(r1))
    print(f'{clock} {k[0]:20s} {k[1]}  t0={r0:.4f} t1={r1:.4f} mean={(r0 + r1) / 2:.4f}  nfeat={len(W[k][0])}', flush=True)
json.dump(res, open(f'ridge_bands_{clock}.json', 'w'), indent=1)
# keep the full-model weights for later inspection
ii, B = W[('all', 'w')]
np.savez(f'ridge_all_{clock}.npz', names=names, ii=ii, B=B)
