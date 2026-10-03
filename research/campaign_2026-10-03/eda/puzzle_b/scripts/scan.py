# Per-sequence summary scan: target tick grid, clocks, coincidences. Output: seqscan_<split>.csv
import sys, time
sys.path.insert(0, '/home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-03/eda/puzzle_b/scripts')
from common import *
split = sys.argv[1]; n = int(sys.argv[2])
path = TRAIN if split == 'train' else VALID
N = nrg(path)
rng = np.random.default_rng(123)
idx = np.sort(rng.choice(N, min(n, N), replace=False))
names = cols(path)
P0A = [f'i0_p{k}' for k in range(11)]; P0B = [f'i0_p{k}' for k in range(11, 22)]
P1A = [f'i1_p{k}' for k in range(11)]; P1B = [f'i1_p{k}' for k in range(11, 22)]
def chg(x):
    return np.r_[False, (x[1:] != x[:-1]).any(1)] if x.ndim == 2 else np.r_[False, x[1:] != x[:-1]]
def scale_of(t):
    v = np.unique(t.astype(np.float64)); nz = np.abs(v[v != 0])
    if len(nz) == 0: return np.nan, np.nan
    s = nz.min(); r = v / s
    return s, np.abs(r - np.round(r)).max()
rows = []
t0_ = time.time()
for j, i in enumerate(idx):
    d = pf(path).read_row_group(i).to_pandas()
    r = {'rg': i, 'seq_ix': int(d.seq_ix.iloc[0])}
    for t in ['t0', 't1']:
        x = d[t].values
        s, err = scale_of(x)
        r[f'{t}_scale'] = s; r[f'{t}_interr'] = err
        k = np.round(x / s) if s == s else x
        r[f'{t}_kmean'] = k[99:].mean(); r[f'{t}_kstd'] = k[99:].std(); r[f'{t}_kmin'] = k.min(); r[f'{t}_kmax'] = k.max()
        r[f'{t}_zero'] = (x == 0).mean(); r[f'{t}_chg'] = chg(x).mean(); r[f'{t}_absmean'] = np.abs(x[99:]).mean()
        r[f'{t}_clipfrac'] = (np.abs(x[99:]) > 2).mean()
    e = {}
    for nm, cs in [('i0pA', P0A), ('i0pB', P0B), ('i1pA', P1A), ('i1pB', P1B),
                   ('i0dp', [f'i0_dp{k}' for k in range(4)]), ('i1dp', [f'i1_dp{k}' for k in range(4)]),
                   ('i0dv', [f'i0_dv{k}' for k in range(4)]), ('i1dv', [f'i1_dv{k}' for k in range(4)]),
                   ('i0v', [f'i0_v{k}' for k in range(22)]), ('i1v', [f'i1_v{k}' for k in range(22)]), ('a7', ['a7'])]:
        e[nm] = chg(d[cs].values); r[f'chg_{nm}'] = e[nm].mean()
    tc = chg(d['t0'].values) | chg(d['t1'].values)
    i0any = e['i0pA'] | e['i0pB']
    r['tchg_not_i0p'] = (tc & ~i0any).sum(); r['i0p_events'] = i0any.sum()
    r['frac_i0p_events_with_tchg'] = (tc & i0any).sum() / max(1, i0any.sum())
    r['eq_i0i1_p'] = (d[P0A + P0B].values == d[P1A + P1B].values).all(1).mean()
    r['eq_i0i1_dp'] = (d[[f'i0_dp{k}' for k in range(4)]].values == d[[f'i1_dp{k}' for k in range(4)]].values).all(1).mean()
    for a in ['a5', 'a6', 'a7']:
        r[f'{a}_lo'] = (d[a].values < -5.19).mean()
    for a in ['a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7']:
        r[f'{a}_mean'] = d[a].values.mean(); r[f'{a}_std'] = d[a].values.std()
    r['i0dv_mode_frac'] = (d['i0_dv0'].values == 0.076032).mean()
    if 'is_scored' in d:
        r['scored_frac'] = d.is_scored.mean(); r['first_scored'] = int(np.argmax(d.is_scored.values)) if d.is_scored.any() else -1
    r['need_pred_first'] = int(np.argmax(d.need_prediction.values))
    rows.append(r)
    if j % 50 == 0: print(j, time.time() - t0_, flush=True)
pd.DataFrame(rows).to_csv(f'{OUT}/seqscan_{split}.csv', index=False)
print('done', time.time() - t0_)
