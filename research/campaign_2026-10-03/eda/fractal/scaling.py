"""Per-sequence scaling statistics over valid sequences -> scaling.npz (DFA of targets/features, activity, tails, MFDFA)."""
import sys; sys.path.insert(0, '/home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-03/eda/fractal')
import numpy as np, fx
F = fx.FEAT; P0 = [f'i0_p{k}' for k in range(22)]
S = fx.scales_dyadic(4, 4096, 2); SE = fx.scales_dyadic(4, 1024, 2)
QS = np.array([-4., -3, -2, -1, 0, 1, 2, 3, 4])
B = np.load(fx.OUT + 'lin_base_B.npz')
GS = list(range(0, 1873, 39))  # 49 valid sequences
WA = [16, 64, 256, 1024]; ABINS = np.r_[0, np.logspace(-3, 0, 25)]
R = {k: [] for k in ['q', 'rate', 'sd0', 'sd1', 'mabs0', 'mabs1', 'chg0', 'F_t0', 'F_t1', 'F_dt0', 'F_ev', 'F_t0e', 'F_dt0e', 'F_feat', 'F_absdfeat',
                     'F_pred0', 'F_res0', 'F_pred1', 'F_res1', 'band_r0', 'band_r1', 'band_var0', 'band_varres0', 'MF_dt0e', 'MF_ev', 'MF_absdpc', 'MF_res0',
                     'act_sum', 'act_cnt']}
U = None
for g in GS:
    d = fx.read_seq(fx.VALID, g, F + ['t0', 't1'])
    X = np.stack([d[c] for c in F], 1).astype(np.float64); y0 = d['t0'].astype(np.float64); y1 = d['t1'].astype(np.float64)
    P = X[:, [fx.IDX[c] for c in P0]]
    ev = np.r_[True, np.any(np.diff(P, axis=0) != 0, axis=1)]
    nz = np.abs(np.r_[y0, y1]); q = nz[nz > 0].min()
    R['q'].append(q); R['rate'].append(ev.mean()); R['sd0'].append(y0.std()); R['sd1'].append(y1.std())
    R['mabs0'].append(np.abs(y0).mean()); R['mabs1'].append(np.abs(y1).mean()); R['chg0'].append(np.mean(np.diff(y0) != 0))
    R['F_t0'].append(fx.dfa1(y0, S)); R['F_t1'].append(fx.dfa1(y1, S)); R['F_dt0'].append(fx.dfa1(np.diff(y0), S))
    R['F_ev'].append(fx.dfa1(ev.astype(np.float64), S))
    y0e = y0[ev]; R['F_t0e'].append(fx.dfa1(y0e, SE)); R['F_dt0e'].append(fx.dfa1(np.diff(y0e), SE))
    R['F_feat'].append(np.stack([fx.dfa1(X[:, j], S) if X[:, j].std() > 0 else np.full(len(S), np.nan) for j in range(112)]))
    dX = np.abs(np.diff(X, axis=0))
    R['F_absdfeat'].append(np.stack([fx.dfa1(dX[:, j], S) if dX[:, j].std() > 0 else np.full(len(S), np.nan) for j in range(112)]))
    Xc = np.c_[X, np.ones(len(X))]
    for j, (y, b) in enumerate([(y0, B['all0']), (y1, B['all1'])]):
        pr = Xc @ b; res = y - pr
        R[f'F_pred{j}'].append(fx.dfa1(pr, S)); R[f'F_res{j}'].append(fx.dfa1(res, S))
        # band-pass (Haar-like): MA_s - MA_2s, centred; correlation of prediction vs target per band (diagnostic only)
        br = []; bv = []; bvr = []
        for s in S:
            k1 = np.ones(s) / s; k2 = np.ones(2 * s) / (2 * s)
            if s == S[0]:
                fy = y - np.convolve(y, k1, 'same'); fp = pr - np.convolve(pr, k1, 'same')
            else:
                fy = np.convolve(y, k1, 'same') - np.convolve(y, k2, 'same'); fp = np.convolve(pr, k1, 'same') - np.convolve(pr, k2, 'same')
            sl = slice(2 * s, len(y) - 2 * s)
            br.append(np.corrcoef(fy[sl], fp[sl])[0, 1] if fy[sl].std() > 0 else np.nan); bv.append(fy[sl].var()); bvr.append((fy[sl] - fp[sl]).var())
        R[f'band_r{j}'].append(br)
        if j == 0: R['band_var0'].append(bv); R['band_varres0'].append(bvr)
        if j == 0: R['MF_res0'].append(fx.mfdfa1(res, S, QS))
    R['MF_dt0e'].append(fx.mfdfa1(np.diff(y0e), SE, QS)); R['MF_ev'].append(fx.mfdfa1(ev.astype(np.float64), S, QS))
    if U is None:
        dP = np.diff(P, axis=0); U = np.linalg.svd(dP[np.any(dP != 0, 1)], full_matrices=False)[2][0]
    R['MF_absdpc'].append(fx.mfdfa1(np.abs(np.diff(P @ U)), S, QS))
    # |target| vs causal activity (fraction of event rows in the last w rows)
    cs = np.cumsum(ev.astype(np.float64)); asum = np.zeros((len(WA), len(ABINS), 4)); acnt = np.zeros((len(WA), len(ABINS)))
    for wi, w in enumerate(WA):
        a = (cs - np.r_[np.zeros(w), cs[:-w]]) / np.minimum(np.arange(1, 20001), w)
        bi = np.searchsorted(ABINS, a)[99:]
        for kk, v in enumerate([np.abs(y0), np.abs(y1), np.abs(y0) / q, y0 ** 2]):
            asum[wi, :, kk] = np.bincount(bi, v[99:], len(ABINS))[:len(ABINS)]
        acnt[wi] = np.bincount(bi, minlength=len(ABINS))[:len(ABINS)]
    R['act_sum'].append(asum); R['act_cnt'].append(acnt)
    print(g, 'q %.3f rate %.3f' % (q, ev.mean()), flush=True)
np.savez(fx.OUT + 'scaling.npz', S=S, SE=SE, QS=QS, WA=WA, ABINS=ABINS, GS=GS, **{k: np.array(v) for k, v in R.items()})
print('done')
