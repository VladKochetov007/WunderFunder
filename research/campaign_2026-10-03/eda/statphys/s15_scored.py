# Structure of the is_scored mask in valid: per-seq fraction, run lengths, position, relation to target activity
import sys; sys.path.insert(0, 'research/campaign_2026-10-03/eda/statphys')
from common import *
import pandas as pd
pf = pq.ParquetFile(VALID)
rng = np.random.default_rng(8)
rows = []
runs = []; pos = np.zeros(20000); n = 0
for rg in rng.choice(pf.metadata.num_row_groups, 60, replace=False):
    t = pf.read_row_group(int(rg), columns=['t0', 't1', 'is_scored', 'i0_dv0', 'i0_p0', 'i0_p11'])
    s = t.column('is_scored').to_numpy().astype(bool); y0 = t.column('t0').to_numpy()
    a = np.abs(y0); a = a[a > 1e-6]; u = a.min()
    T0 = np.round(y0 / u)
    pos += s; n += 1
    # run lengths of scored blocks
    ds = np.diff(np.r_[0, s.astype(int), 0]); st = np.where(ds == 1)[0]; en = np.where(ds == -1)[0]
    runs += list(en - st)
    chg = np.r_[False, np.diff(T0) != 0]
    trade = t.column('i0_dv0').to_numpy() != np.bincount(np.unique(t.column('i0_dv0').to_numpy(), return_inverse=True)[1]).argmax()
    pchg = np.r_[False, np.diff(t.column('i0_p0').to_numpy()) != 0] | np.r_[False, np.diff(t.column('i0_p11').to_numpy()) != 0]
    rows.append((round(u, 4), s.mean(), len(st), np.mean(np.abs(T0[s])) if s.any() else np.nan, np.mean(np.abs(T0[~s])),
                 chg[s].mean() if s.any() else np.nan, chg[~s].mean(), pchg[s].mean() if s.any() else np.nan, pchg[~s].mean(), np.mean(T0[s] == 0) if s.any() else np.nan, np.mean(T0[~s] == 0)))
d = pd.DataFrame(rows, columns=['u', 'frac', 'nblocks', 'absT_sc', 'absT_un', 'tchg_sc', 'tchg_un', 'pchg_sc', 'pchg_un', 'zero_sc', 'zero_un'])
pd.set_option('display.width', 250)
print(d.groupby('u').mean().round(4).to_string())
print(d.sort_values('frac').round(3).head(8).to_string()); print(d.sort_values('frac').round(3).tail(5).to_string())
r = np.array(runs); print('scored block length quantiles', np.percentile(r, [5, 25, 50, 75, 95, 99]), 'n blocks', len(r))
pos /= n
print('mean scored by step bucket', [round(pos[i:i + 2000].mean(), 4) for i in range(0, 20000, 2000)])
