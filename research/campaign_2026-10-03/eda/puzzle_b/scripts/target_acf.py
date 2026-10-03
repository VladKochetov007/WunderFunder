# Target autocorrelation in row time and in i0-event time, per class -> fig08_target_acf.png
import sys
sys.path.insert(0, '/home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-03/eda/puzzle_b/scripts')
from common import *
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
s = pd.read_csv(f'{OUT}/seqscan_train.csv'); s['cls'] = s.t0_scale.round(4)
P = [f'i0_p{k}' for k in range(22)]
row_lags = np.r_[0:20, 20:200:5, 200:2000:50, 2000:8000:250]
ev_lags = np.r_[0:20, 20:100:2, 100:600:10]
def acf(x, lags):
    x = x - x.mean(); v = (x * x).mean()
    return np.array([(x[:len(x) - l] * x[l:]).mean() / v if l < len(x) - 10 else np.nan for l in lags])
res = {}
for cl, g in s.groupby('cls'):
    R = {'t0': [], 't1': [], 'sum': []}; E = {'t0': [], 't1': [], 'sum': []}; gaps = []
    for rg in g.rg.values[:25]:
        d = seq(int(rg), P + ['t0', 't1']).iloc[99:]
        A = d[P].values; ev = np.r_[True, (A[1:] != A[:-1]).any(1)]
        gaps.append(np.diff(np.where(ev)[0]))
        for nm, y in (('t0', d.t0.values), ('t1', d.t1.values), ('sum', d.t0.values + d.t1.values)):
            y = y.astype(np.float64)
            R[nm].append(acf(y, row_lags)); E[nm].append(acf(y[ev], ev_lags))
    res[cl] = ({k: np.nanmean(v, 0) for k, v in R.items()}, {k: np.nanmean(v, 0) for k, v in E.items()}, np.concatenate(gaps))
fig, ax = plt.subplots(1, 3, figsize=(18, 5))
for cl, (R, E, gp) in res.items():
    ax[0].plot(row_lags, R['t0'], label=f'{cl} t0'); ax[0].plot(row_lags, R['sum'], ls=':', label=f'{cl} t0+t1')
    ax[1].plot(ev_lags, E['t0'], label=f'{cl} t0'); ax[1].plot(ev_lags, E['sum'], ls=':', label=f'{cl} t0+t1')
    ax[2].hist(np.log10(gp), bins=80, histtype='step', density=True, label=str(cl))
    print(cl, 'row acf t0 at', dict(zip(row_lags[[1, 5, 10, 20, 40, 60, 80]].tolist(), np.round(R['t0'][[1, 5, 10, 20, 40, 60, 80]], 3).tolist())))
    print(cl, 'evt acf t0 at', dict(zip(ev_lags[[1, 5, 10, 19, 30, 50, 70, 90]].tolist(), np.round(E['t0'][[1, 5, 10, 19, 30, 50, 70, 90]], 3).tolist())))
    print(cl, 'evt acf sum at', dict(zip(ev_lags[[1, 2, 5, 10, 19]].tolist(), np.round(E['sum'][[1, 2, 5, 10, 19]], 3).tolist())))
    print(cl, 'median gap between i0 events', np.median(gp), 'mean', gp.mean().round(2))
ax[0].set_xscale('symlog', linthresh=10); ax[0].set_title('ACF of target in row time'); ax[0].legend(fontsize=7); ax[0].axhline(0, c='k', lw=.5)
ax[1].set_xscale('symlog', linthresh=10); ax[1].set_title('ACF of target sampled at i0 events (event time)'); ax[1].legend(fontsize=7); ax[1].axhline(0, c='k', lw=.5)
ax[2].set_title('log10 gap (rows) between i0 price events'); ax[2].legend()
plt.tight_layout(); plt.savefig(f'{OUT}/fig08_target_acf.png', dpi=70)
