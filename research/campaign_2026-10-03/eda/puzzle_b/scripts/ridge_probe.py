# Cheap linear probe: which column groups carry signal? Per-class ridge fit on train subsample, scored on valid is_scored rows.
import sys
sys.path.insert(0, '/home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-03/eda/puzzle_b/scripts')
from common import *
NTR = int(sys.argv[1]); NVA = int(sys.argv[2])
s = pd.read_csv(f'{OUT}/seqscan_train.csv'); s['cls'] = s.t0_scale.round(4)
v = pd.read_csv(f'{OUT}/seqscan_valid.csv'); v['cls'] = v.t0_scale.round(4)
I0P = [f'i0_p{k}' for k in range(22)]; I0DP = [f'i0_dp{k}' for k in range(4)]; I0DV = [f'i0_dv{k}' for k in range(4)]
I1P = [f'i1_p{k}' for k in range(22)]; I1DP = [f'i1_dp{k}' for k in range(4)]; I1DV = [f'i1_dv{k}' for k in range(4)]
I0V = [f'i0_v{k}' for k in range(22)]; I1V = [f'i1_v{k}' for k in range(22)]
A = [f'a{k}' for k in range(8)]
RAW = I0P + I0DP + I0DV + I0V + I1P + I1DP + I1DV + I1V + A
def feats(d):
    X = d[RAW].astype(np.float32)
    out = {c: X[c].values for c in RAW}
    # deviation from slow EMA (causal)
    for c in I0P + I1P + ['a0', 'a2', 'a3']:
        out[c + '_dev'] = (X[c] - X[c].ewm(alpha=0.01).mean()).values.astype(np.float32)
    # staleness: rows since last i0 price event, and i1 price change since last i0 event
    P = d[I0P].values; ev = np.r_[True, (P[1:] != P[:-1]).any(1)]
    last = np.maximum.accumulate(np.where(ev, np.arange(len(ev)), 0))
    out['since_i0'] = np.log1p(np.arange(len(ev)) - last).astype(np.float32)
    for c in I1P[:3] + I1P[11:14]:
        x = X[c].values; out[c + '_sinceev'] = (x - x[last]).astype(np.float32)
    return pd.DataFrame(out)
GROUPS = {
    'i0p': I0P, 'i0p+dp': I0P + I0DP,
    'i0all': I0P + I0DP + I0DV + I0V,
    'i0all+i1all': I0P + I0DP + I0DV + I0V + I1P + I1DP + I1DV + I1V,
    '+a': I0P + I0DP + I0DV + I0V + I1P + I1DP + I1DV + I1V + A,
    '+a+dev': None, '+a+dev+stale': 'all',
}
def wcorr(p, y):
    p = np.clip(p, -2, 2); y = np.clip(y, -2, 2); w = np.abs(y)
    mp = (w * p).sum() / w.sum(); my = (w * y).sum() / w.sum()
    return (w * (p - mp) * (y - my)).sum() / np.sqrt((w * (p - mp) ** 2).sum() * (w * (y - my) ** 2).sum())
def load(df, path, n, step, scored):
    Xs, Ys, Ss = [], [], []
    for _, r in df.groupby('cls').head(n).iterrows():
        d = seq(int(r.rg), path=path)
        F = feats(d)
        sl = slice(99, None, step)
        global FCOLS
        FCOLS = list(F.columns)
        Xs.append(F.values[sl].astype(np.float32)); Ys.append(d[['t0', 't1']].values[sl]); Ss.append(np.full(len(Xs[-1]), r.cls))
        if scored: Ss[-1] = np.c_[Ss[-1], d.is_scored.values[sl]]
        del d, F
    return pd.DataFrame(np.concatenate(Xs), columns=FCOLS), np.concatenate(Ys), np.concatenate(Ss)
Xtr, Ytr, Ctr = load(s, TRAIN, NTR, 4, False)
Xva, Yva, Cva = load(v, VALID, NVA, 2, True)
msk = Cva[:, 1].astype(bool); cva = Cva[:, 0]
allc = list(Xtr.columns)
devc = [c for c in allc if c.endswith('_dev')]
GROUPS['+a+dev'] = GROUPS['+a'] + devc
GROUPS['+a+dev+stale'] = allc
res = []
for g, cs in GROUPS.items():
    P = np.zeros_like(Yva)
    for cl in sorted(np.unique(Ctr)):
        tr = Ctr == cl; va = cva == cl
        X = Xtr.loc[tr, cs].values.astype(np.float64); mu = X.mean(0); sd = X.std(0) + 1e-6
        Xn = (X - mu) / sd
        for j in range(2):
            y = np.clip(Ytr[tr, j], -2, 2)
            beta = np.linalg.solve(Xn.T @ Xn + 10.0 * len(Xn) * 1e-3 * np.eye(Xn.shape[1]), Xn.T @ (y - y.mean()))
            P[va, j] = ((Xva.loc[va, cs].values - mu.astype(np.float32)) / sd.astype(np.float32)) @ beta.astype(np.float32) + y.mean()
    row = {'group': g, 'nfeat': len(cs)}
    for j in range(2):
        row[f't{j}_all'] = wcorr(P[msk, j], Yva[msk, j])
        for cl in sorted(np.unique(cva)):
            m = msk & (cva == cl); row[f't{j}_{cl}'] = wcorr(P[m, j], Yva[m, j])
    res.append(row); print(row, flush=True)
pd.DataFrame(res).to_csv(f'{OUT}/ridge_probe.csv', index=False)
