"""Per-sequence catalogue: target quantum (tick/scale), target stats, event-clock rates."""
import numpy as np, sys, time
import sio

out = {}
for split in ('train', 'valid'):
    cols = ['seq_ix', 'i0_p0', 'i0_p11', 'i0_v0', 'i1_p0', 'i1_p11', 'a7', 't0', 't1']
    rows = []
    t = time.time()
    for rg in range(sio.n_seq(split)):
        d = sio.load(rg, cols, split)
        u = np.unique(np.concatenate([d['t0'], d['t1']]))
        q = np.diff(u).min() if len(u) > 1 else np.nan
        sl = slice(99, None)
        y0, y1 = d['t0'][sl], d['t1'][sl]
        rows.append([rg, d['seq_ix'][0], q, y0.std(), y1.std(), y0.mean(), y1.mean(),
                     np.corrcoef(y0, y1)[0, 1] if y0.std() > 0 and y1.std() > 0 else np.nan,
                     (np.diff(d['t0']) != 0).sum(), (np.diff(d['t1']) != 0).sum(),
                     (np.diff(d['i0_p0']) != 0).sum(), (np.diff(d['i0_p11']) != 0).sum(),
                     (np.diff(d['i0_v0']) != 0).sum(), (np.diff(d['i1_p0']) != 0).sum(),
                     (d['a7'] > -5).mean(), np.abs(y0).max(), np.abs(y1).max()])
        if rg % 1000 == 0:
            print(split, rg, time.time() - t, flush=True)
    out[split] = np.array(rows, dtype=np.float64)
names = 'rg seq q t0std t1std t0mean t1mean rho n_t0chg n_t1chg n_i0p0 n_i0p11 n_i0v0 n_i1p0 a7_frac t0absmax t1absmax'.split()
np.savez('catalogue.npz', names=np.array(names), **out)
