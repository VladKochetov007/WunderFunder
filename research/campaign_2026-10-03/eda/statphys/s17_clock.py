# Event time vs clock time: is a2/a3 a clock (inter-event duration) proxy? Waiting-time distributions.
import sys; sys.path.insert(0, 'research/campaign_2026-10-03/eda/statphys')
from common import *
import pandas as pd
pf = pq.ParquetFile(VALID)
rng = np.random.default_rng(10)
R = []; WT = {}; WTt = {}
for rg in rng.choice(pf.metadata.num_row_groups, 40, replace=False):
    X, Y, u, ex = load(pf, rg, extra=('is_scored',))
    T = np.round(Y / u)
    ch = np.r_[False, np.any(np.diff(T, axis=0) != 0, axis=1)]
    pch = np.r_[False, np.any(np.diff(X[:, :22], axis=0) != 0, axis=1)]
    sc = ex['is_scored'].astype(bool)
    c = ucls(u)
    R.append(pd.DataFrame({'c': c, 'a2': X[:, 106], 'a3': X[:, 107], 'ch': ch, 'pch': pch, 'sc': sc, 'absdT': np.abs(np.r_[0, np.diff(T[:, 0])])}).iloc[99:])
    idx = np.where(pch)[0]; WT.setdefault(c, []).append(np.diff(idx))
    idx = np.where(ch)[0]; WTt.setdefault(c, []).append(np.diff(idx))
D = pd.concat(R)
D['a3q'] = pd.qcut(D.a3, 10, labels=False, duplicates='drop'); D['a2q'] = pd.qcut(D.a2, 10, labels=False, duplicates='drop')
pd.set_option('display.width', 250)
print('by a3 decile: P(target change), P(price-state change), P(scored)')
print(D.groupby(['a3q']).agg(a3=('a3', 'mean'), tch=('ch', 'mean'), pch=('pch', 'mean'), sc=('sc', 'mean'), dT=('absdT', 'mean')).round(4).T.to_string())
print('by a2 decile')
print(D.groupby(['a2q']).agg(a2=('a2', 'mean'), tch=('ch', 'mean'), pch=('pch', 'mean'), sc=('sc', 'mean'), dT=('absdT', 'mean')).round(4).T.to_string())
print('P(target change | price-state change) vs not:', D.groupby('pch').ch.mean().round(4).to_dict())
for c in WT:
    w = np.concatenate(WT[c]); wt = np.concatenate(WTt[c])
    cv = w.std() / w.mean(); cvt = wt.std() / wt.mean()
    print(f'class {c}: price-state waiting time mean {w.mean():.2f} CV {cv:.2f} q50/90/99 {np.percentile(w,[50,90,99])};  target-change wait mean {wt.mean():.2f} CV {cvt:.2f} q50/90/99 {np.percentile(wt,[50,90,99])}')
np.save(OUT + 'data/waits.npy', np.array([(c, np.concatenate(WT[c]), np.concatenate(WTt[c])) for c in WT], dtype=object), allow_pickle=True)
D.sample(200000, random_state=0).to_parquet(OUT + 'data/clock_sample.parquet')
