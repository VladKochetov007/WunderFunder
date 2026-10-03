"""Target horizon clock test (row vs event clock), run-length/jump scaling, tails (Hill), zero-weight share by quantum class."""
import sys; sys.path.insert(0, '/home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-03/eda/fractal')
import numpy as np, fx, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
P0 = [f'i0_p{k}' for k in range(22)]


def cross(a, lev):
    i = np.where(a < lev)[0]; return i[0] if len(i) else np.nan


def hill(x, frac=0.02):
    x = np.sort(x[x > 0])[::-1]; k = max(int(len(x) * frac), 10)
    return 1.0 / np.mean(np.log(x[:k] / x[k])) if len(x) > k + 1 else np.nan


rows = []; RL = []; JS = []; TS = {}; TA = {}
for g in range(2, 1873, 19):  # 99 valid seqs, light columns
    d = fx.read_seq(fx.VALID, g, P0 + ['t0', 't1', 'i0_dv0'])
    P = np.stack([d[c] for c in P0], 1); ev = np.r_[True, np.any(np.diff(P, axis=0) != 0, axis=1)]
    y0 = d['t0'].astype(np.float64); y1 = d['t1'].astype(np.float64)
    nz = np.abs(np.r_[y0, y1]); q = nz[nz > 0].min(); rate = ev.mean()
    a_r = fx.acf(y0, 4000); a_e = fx.acf(y0[ev], min(2000, ev.sum() - 2))
    k0 = np.round(y0 / q).astype(int); dk = np.diff(k0); chg = np.where(dk != 0)[0]
    runs = np.diff(chg); jumps = np.abs(dk[chg[1:]])  # run length before each change, and the jump size at its end
    RL.append(runs); JS.append(jumps)
    # causal: time since last price event vs |t|
    last = np.maximum.accumulate(np.where(ev, np.arange(20000), 0)); tsince = np.arange(20000) - last
    TS.setdefault(round(q, 3), []).append(np.c_[tsince[99:], np.abs(y0[99:]) / q])
    rows.append([q, rate, cross(a_r, .5), cross(a_r, .2), cross(a_e, .5), cross(a_e, .2), np.mean(y0 == 0), np.mean(y1 == 0),
                 hill(np.abs(dk[dk != 0]).astype(float)), hill(np.abs(k0).astype(float) + 0.0), np.mean(y0), np.mean(y1),
                 np.mean(np.abs(np.clip(y0, -2, 2))), np.mean(np.abs(y0) > 2)])
R = np.array(rows)
names = ['q', 'rate', 'Lrow50', 'Lrow20', 'Lev50', 'Lev20', 'zero0', 'zero1', 'hill_dk', 'hill_k', 'mean0', 'mean1', 'mabsclip0', 'clipfrac0']
np.savez(fx.OUT + 'horizon_tails.npz', R=R, names=names)
lr = np.log(R[:, 1])
for j, nm in [(2, 'Lrow50'), (3, 'Lrow20'), (4, 'Lev50'), (5, 'Lev20')]:
    m = np.isfinite(R[:, j]) & (R[:, j] > 0)
    print('%s: median %.1f; slope of log L on log rate = %.2f (corr %.2f)' % (nm, np.median(R[m, j]), np.polyfit(lr[m], np.log(R[m, j]), 1)[0], np.corrcoef(lr[m], np.log(R[m, j]))[0, 1]))
for qq in np.unique(np.round(R[:, 0], 3)):
    m = np.round(R[:, 0], 3) == qq
    print('q=%.3f n=%2d rate %.3f Lrow50 %.0f Lev50 %.0f zero0 %.3f zero1 %.3f hill|dk| %.2f mean0 %+.3f mean1 %+.3f E|y|clip %.3f clip>2 %.4f' % (
        qq, m.sum(), *np.nanmedian(R[m][:, [1, 2, 4, 6, 7, 8, 10, 11, 12, 13]], 0)))
RL = np.concatenate(RL); JS = np.concatenate(JS)
bins = np.unique(np.logspace(0, 3.5, 15).astype(int)); bi = np.digitize(RL, bins)
fig, ax = plt.subplots(1, 3, figsize=(18, 4.5))
mx = [RL[bi == i].mean() for i in range(1, len(bins)) if (bi == i).sum() > 50]; my = [JS[bi == i].mean() for i in range(1, len(bins)) if (bi == i).sum() > 50]
ax[0].loglog(mx, my, 'o-'); ax[0].set_xlabel('run length of constant target before a change (rows)'); ax[0].set_ylabel('E|jump| (quanta)')
sl = np.polyfit(np.log(mx), np.log(my), 1)[0]; ax[0].set_title('target jump size vs waiting time: slope %.2f' % sl); print('jump-vs-wait slope', sl)
u, c = np.unique(JS, return_counts=True); ax[1].loglog(u, c / c.sum(), 'o'); ax[1].set_title('target jump size distribution (quanta), pooled'); ax[1].set_xlabel('|jump|')
for qq, L in TS.items():
    L = np.vstack(L); b = np.unique(np.logspace(0, 3, 13).astype(int)); ii = np.digitize(L[:, 0], b)
    xs = [L[ii == i, 0].mean() for i in range(0, len(b)) if (ii == i).sum() > 200]; ys = [L[ii == i, 1].mean() for i in range(0, len(b)) if (ii == i).sum() > 200]
    ax[2].semilogx(np.maximum(xs, .5), ys, 'o-', label=f'q={qq}')
ax[2].set_xlabel('rows since last i0 price update (causal)'); ax[2].set_ylabel('E|t0|/q'); ax[2].legend(); ax[2].set_title('|target| vs quiet time')
plt.tight_layout(); plt.savefig(fx.OUT + 'fig6_jumps_quiet.png', dpi=80); plt.close()
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for j, nm, c in [(2, 'Lrow50 (rows, ACF<0.5)', 'C0'), (4, 'Lev50 (events)', 'C1')]:
    ax[0].scatter(R[:, 1], R[:, j], c=c, s=12, label=nm)
ax[0].set_xscale('log'); ax[0].set_yscale('log'); ax[0].set_xlabel('seq price-update rate'); ax[0].legend(); ax[0].set_title('target decorrelation lag vs activity')
ax[1].scatter(R[:, 1], R[:, 2] * R[:, 1], s=12); ax[1].set_xscale('log'); ax[1].set_yscale('log'); ax[1].set_title('Lrow50 x rate (= expected events within the lag)')
plt.tight_layout(); plt.savefig(fx.OUT + 'fig7_horizon_clock.png', dpi=80); plt.close()
