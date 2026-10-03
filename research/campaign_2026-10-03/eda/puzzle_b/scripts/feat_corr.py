# Univariate weighted correlations (score-like) of each input column (level and deviation from EMA) with t0/t1, per class.
import sys
sys.path.insert(0, '/home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-03/eda/puzzle_b/scripts')
from common import *
s = pd.read_csv(f'{OUT}/seqscan_train.csv'); s['cls'] = s.t0_scale.round(4)
pick = s.groupby('cls').head(30)
cs = [c for c in cols() if c not in ('seq_ix', 'step_in_seq', 'need_prediction', 't0', 't1')]
def ema(x, a):
    out = np.empty_like(x); m = x[0]
    for i in range(len(x)):
        m += a * (x[i] - m); out[i] = m
    return out
# accumulate weighted moments per class/feature/target: sum w, w*x, w*y, w*x*x, w*y*y, w*x*y
acc = {}
for _, r in pick.iterrows():
    d = seq(int(r.rg))
    d = d.iloc[99:]
    for tn in ['t0', 't1']:
        y = np.clip(d[tn].values.astype(np.float64), -2, 2); w = np.abs(y)
        for c in cs:
            x = d[c].values.astype(np.float64)
            for kind, f in (('lvl', x), ('dev', x - pd.Series(x).ewm(alpha=0.01).mean().values)):
                k = (r.cls, tn, c, kind)
                a = acc.setdefault(k, np.zeros(6))
                a += [w.sum(), (w * f).sum(), (w * y).sum(), (w * f * f).sum(), (w * y * y).sum(), (w * f * y).sum()]
rows = []
for (cl, tn, c, kind), a in acc.items():
    W, sx, sy, sxx, syy, sxy = a
    mx, my = sx / W, sy / W
    cov = sxy / W - mx * my; vx = sxx / W - mx * mx; vy = syy / W - my * my
    rows.append((cl, tn, c, kind, cov / np.sqrt(vx * vy + 1e-30)))
R = pd.DataFrame(rows, columns=['cls', 'target', 'col', 'kind', 'wcorr'])
R.to_csv(f'{OUT}/feat_corr.csv', index=False)
print('done')
