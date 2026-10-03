"""What are a0..a7? Compare with event-clock quantities (rows since last i0/i1 update) and with each other."""
import numpy as np
import sio
from scipy.stats import spearmanr
rng = np.random.default_rng(6)
idx = rng.choice(sio.n_seq(), 6, replace=False)
def since(ev):
    out = np.zeros(len(ev)); c = 0
    for i, e in enumerate(ev):
        c = 0 if e else c + 1; out[i] = c
    return out
def until(ev):
    return since(ev[::-1])[::-1]
for rg in idx:
    d = sio.load(int(rg))
    P0 = np.stack([d[f'i0_p{k}'] for k in range(22)], 1); V0 = np.stack([d[f'i0_v{k}'] for k in range(22)], 1)
    P1 = np.stack([d[f'i1_p{k}'] for k in range(22)], 1); V1 = np.stack([d[f'i1_v{k}'] for k in range(22)], 1)
    e0p = np.r_[True, np.any(np.diff(P0, axis=0) != 0, 1)]; e0v = np.r_[True, np.any(np.diff(V0, axis=0) != 0, 1)]
    e1p = np.r_[True, np.any(np.diff(P1, axis=0) != 0, 1)]; e1v = np.r_[True, np.any(np.diff(V1, axis=0) != 0, 1)]
    e0 = e0p | e0v; e1 = e1p | e1v
    feats = {'since_i0': since(e0), 'since_i0p': since(e0p), 'since_i1': since(e1), 'since_i1p': since(e1p), 'until_i0': until(e0)}
    # trailing event counts in a window of 50 rows
    for w in (10, 100):
        feats[f'n_i0_last{w}'] = np.convolve(e0.astype(float), np.ones(w), 'full')[:20000]
        feats[f'n_i1_last{w}'] = np.convolve(e1.astype(float), np.ones(w), 'full')[:20000]
    print('rg', rg, 'event frac i0', e0.mean().round(3), 'i1', e1.mean().round(3), 'either', (e0 | e1).mean().round(3))
    for k in range(8):
        a = d[f'a{k}']
        s = {n: spearmanr(a, f)[0] for n, f in feats.items()}
        best = sorted(s.items(), key=lambda kv: -abs(kv[1]))[:3]
        sent = (a < -5).mean()
        r1 = np.corrcoef(a[1:], a[:-1])[0, 1]
        print(f'  a{k}: sentinel {sent:.3f} lag1-acf {r1:+.3f} chg-on-i0ev {np.mean(np.diff(a)[e0[1:]] != 0):.2f} chg-off {np.mean(np.diff(a)[~e0[1:]] != 0):.2f}  best:', ' '.join(f'{n}={v:+.2f}' for n, v in best))
