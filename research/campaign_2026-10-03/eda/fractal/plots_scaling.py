"""Figures from scaling.npz."""
import sys; sys.path.insert(0, '/home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-03/eda/fractal')
import numpy as np, fx, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
Z = np.load(fx.OUT + 'scaling.npz'); S = Z['S']; SE = Z['SE']; QS = Z['QS']; O = fx.OUT


def local_h(s, F):
    """Local Hurst = d log F / d log s, between consecutive scales (per sequence), returns mid-scales and slopes."""
    lf = np.log(F); ls = np.log(s)
    return np.exp(0.5 * (ls[1:] + ls[:-1])), np.diff(lf, axis=-1) / np.diff(ls)


def band(ax, x, Y, label, c):
    med = np.nanmedian(Y, 0); lo, hi = np.nanpercentile(Y, [25, 75], 0)
    ax.plot(x, med, '-o', ms=3, c=c, label=label); ax.fill_between(x, lo, hi, color=c, alpha=.15)


# 1 target scaling
fig, ax = plt.subplots(1, 3, figsize=(18, 5))
for k, c in [('F_t0', 'C0'), ('F_t1', 'C1'), ('F_dt0', 'C2'), ('F_ev', 'C3'), ('F_res0', 'C4'), ('F_pred0', 'C5')]:
    m, h = local_h(S, Z[k]); band(ax[0], m, h, k[2:], c)
ax[0].axhline(.5, c='k', ls=':'); ax[0].axhline(1, c='k', ls=':'); ax[0].set_xscale('log'); ax[0].set_title('row clock: local DFA-1 exponent vs scale (median, IQR; 49 valid seqs)')
ax[0].set_xlabel('scale (rows)'); ax[0].legend(fontsize=8)
for k, c in [('F_t0e', 'C0'), ('F_dt0e', 'C2')]:
    m, h = local_h(SE, Z[k]); band(ax[1], m, h, k[2:], c)
ax[1].axhline(.5, c='k', ls=':'); ax[1].set_xscale('log'); ax[1].set_title('event clock (i0 price-update rows only)'); ax[1].set_xlabel('scale (events)'); ax[1].legend()
for k, c in [('F_t0', 'C0'), ('F_pred0', 'C5'), ('F_res0', 'C4')]:
    band(ax[2], S, Z[k] / np.sqrt(S), k[2:], c)
ax[2].set_xscale('log'); ax[2].set_yscale('log'); ax[2].set_title('F(s)/sqrt(s): target vs linear current-state prediction vs residual'); ax[2].legend()
plt.tight_layout(); plt.savefig(O + 'fig1_target_scaling.png', dpi=80); plt.close()

# 2 band correlation / explained variance per scale
fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
band(ax[0], S, Z['band_r0'], 't0', 'C0'); band(ax[0], S, Z['band_r1'], 't1', 'C1')
ax[0].set_xscale('log'); ax[0].set_title('corr(target band, prediction band) per Haar-like band [MA_s - MA_2s]'); ax[0].set_xlabel('scale s (rows)'); ax[0].legend()
bv = Z['band_var0']; br = Z['band_varres0']
ax[1].plot(S, np.nanmedian(1 - br / bv, 0), '-o', label='1 - var(resid band)/var(band), t0'); ax[1].plot(S, np.nanmedian(bv / bv.sum(1, keepdims=True), 0), '-s', label='share of t0 variance in band')
ax[1].set_xscale('log'); ax[1].axhline(0, c='k', lw=.5); ax[1].legend(); ax[1].set_title('where the current-state model works')
plt.tight_layout(); plt.savefig(O + 'fig2_band_predictability.png', dpi=80); plt.close()

# 3 features: DFA exponents short vs long
names = fx.FEAT
def hs(Fa, lo, hi): return np.array([[fx.slope(S, Fa[i, j], lo, hi) for j in range(112)] for i in range(len(Fa))])
Hs = np.nanmedian(hs(Z['F_feat'], 8, 64), 0); Hl = np.nanmedian(hs(Z['F_feat'], 256, 4096), 0)
Ha_s = np.nanmedian(hs(Z['F_absdfeat'], 8, 64), 0); Ha_l = np.nanmedian(hs(Z['F_absdfeat'], 256, 4096), 0)
np.savez(O + 'feature_hurst.npz', names=names, Hs=Hs, Hl=Hl, Ha_s=Ha_s, Ha_l=Ha_l)
fig, ax = plt.subplots(2, 1, figsize=(22, 9), sharex=True)
x = np.arange(112)
ax[0].bar(x - .2, Hs, .4, label='level, scales 8-64'); ax[0].bar(x + .2, Hl, .4, label='level, scales 256-4096'); ax[0].axhline(.5, c='k', ls=':'); ax[0].axhline(1, c='k', ls=':')
ax[0].legend(); ax[0].set_title('median DFA-1 exponent of each feature (as a signal); >1 = nonstationary-like, 0.5 = white')
ax[1].bar(x - .2, Ha_s, .4, label='|diff|, 8-64'); ax[1].bar(x + .2, Ha_l, .4, label='|diff|, 256-4096'); ax[1].axhline(.5, c='k', ls=':')
ax[1].legend(); ax[1].set_title('activity clustering: DFA exponent of |first difference| of each feature')
ax[1].set_xticks(x); ax[1].set_xticklabels(names, rotation=90, fontsize=7)
plt.tight_layout(); plt.savefig(O + 'fig3_feature_hurst.png', dpi=70); plt.close()

# 4 MFDFA
fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
for k, s, lo, hi, c in [('MF_dt0e', SE, 8, 512, 'C0'), ('MF_ev', S, 16, 2048, 'C3'), ('MF_absdpc', S, 16, 2048, 'C2'), ('MF_res0', S, 16, 2048, 'C4')]:
    M = Z[k]; hq = np.array([[fx.slope(s, M[i, qi], lo, hi) for qi in range(len(QS))] for i in range(len(M))])
    band(ax[0], QS, hq, f'{k[3:]} ({lo}-{hi})', c)
    tau = QS[None] * hq - 1; a = np.gradient(tau, QS, axis=1); f = QS[None] * a - tau
    ax[1].plot(np.nanmedian(a, 0), np.nanmedian(f, 0), '-o', c=c, label=k[3:])
ax[0].set_title('MFDFA generalized Hurst h(q) (median, IQR)'); ax[0].set_xlabel('q'); ax[0].legend(fontsize=8)
ax[1].set_title('singularity spectrum f(alpha)'); ax[1].set_xlabel('alpha'); ax[1].legend()
plt.tight_layout(); plt.savefig(O + 'fig4_mfdfa.png', dpi=80); plt.close()

# 5 activity vs magnitude
AS = Z['act_sum'].sum(0); AC = Z['act_cnt'].sum(0); AB = Z['ABINS']; WA = Z['WA']
fig, ax = plt.subplots(1, 3, figsize=(18, 4.5))
for wi, w in enumerate(WA):
    m = AC[wi] > 2000
    ax[0].plot(AB[m], AS[wi, m, 0] / AC[wi, m], '-o', ms=3, label=f'E|t0| vs rate, w={w}')
    ax[1].plot(AB[m], AS[wi, m, 2] / AC[wi, m], '-o', ms=3, label=f'E|t0|/q, w={w}')
ax[0].set_xscale('log'); ax[1].set_xscale('log'); ax[0].legend(fontsize=8); ax[1].legend(fontsize=8)
ax[0].set_xlabel('causal i0 price-update rate over last w rows'); ax[1].set_xlabel('rate')
ax[0].set_title('|target| vs local activity (pooled 49 seqs)'); ax[1].set_title('in quanta (ticks)')
q = Z['q']; r = Z['rate']
sc = ax[2].scatter(r, Z['sd0'], c=np.log(q), cmap='viridis'); ax[2].set_xscale('log'); ax[2].set_yscale('log')
ax[2].set_xlabel('seq price-update rate'); ax[2].set_ylabel('std(t0)'); ax[2].set_title('per sequence: std(t0) vs activity, colour = log quantum q')
plt.colorbar(sc, ax=ax[2]); plt.tight_layout(); plt.savefig(O + 'fig5_activity_magnitude.png', dpi=80); plt.close()

print('quantum values:', np.unique(np.round(q, 4), return_counts=True))
for qq in np.unique(np.round(q, 4)):
    m = np.round(q, 4) == qq
    print('q=%.4f n=%d rate med %.3f sd0 med %.3f sd0/q %.2f chg %.3f' % (qq, m.sum(), np.median(r[m]), np.median(Z['sd0'][m]), np.median(Z['sd0'][m] / qq), np.median(Z['chg0'][m])))
print('corr(log rate, log sd0) %.3f  corr(log q, log sd0) %.3f corr(log rate, log q) %.3f' % (np.corrcoef(np.log(r), np.log(Z['sd0']))[0, 1], np.corrcoef(np.log(q), np.log(Z['sd0']))[0, 1], np.corrcoef(np.log(r), np.log(q))[0, 1]))
for k, lo, hi in [('F_t0', 4, 32), ('F_t0', 64, 512), ('F_t0', 512, 4096), ('F_dt0', 4, 64), ('F_dt0', 256, 4096), ('F_ev', 4, 64), ('F_ev', 256, 4096), ('F_res0', 4, 64), ('F_res0', 256, 4096), ('F_pred0', 4, 64), ('F_pred0', 256, 4096)]:
    h = [fx.slope(S, f, lo, hi) for f in Z[k]]; print('%-8s %4d-%4d  H med %.3f IQR %.3f-%.3f' % (k, lo, hi, np.nanmedian(h), *np.nanpercentile(h, [25, 75])))
for k, lo, hi in [('F_t0e', 4, 32), ('F_t0e', 64, 1024), ('F_dt0e', 4, 32), ('F_dt0e', 64, 1024)]:
    h = [fx.slope(SE, f, lo, hi) for f in Z[k]]; print('%-8s %4d-%4d  H med %.3f IQR %.3f-%.3f' % (k, lo, hi, np.nanmedian(h), *np.nanpercentile(h, [25, 75])))
print('band r0', np.round(np.nanmedian(Z['band_r0'], 0), 3)); print('band r1', np.round(np.nanmedian(Z['band_r1'], 0), 3)); print('S', S)
print('band var share t0', np.round(np.nanmedian(bv / bv.sum(1, keepdims=True), 0), 3))
print('band expl t0', np.round(np.nanmedian(1 - br / bv, 0), 3))
o = np.argsort(-Hl)
print('features highest long-scale H:', [(names[i], round(Hs[i], 2), round(Hl[i], 2)) for i in o[:12]])
print('features lowest long-scale H:', [(names[i], round(Hs[i], 2), round(Hl[i], 2)) for i in o[-12:]])
o = np.argsort(-Ha_l); print('|diff| clustering highest:', [(names[i], round(Ha_s[i], 2), round(Ha_l[i], 2)) for i in o[:12]])
