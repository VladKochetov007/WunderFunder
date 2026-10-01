# F6 — Ensemble diversity: correlation between the since member's and the distilled GRU256's residuals (lower = more
# complementary) and the gain of their average over the better single member, by class × rows until next move.
_fig, _ax = plt.subplots(1, 2, figsize=(13, 3.8))
for _c in range(4):
    _sel = CF0 == _c; _rc, _gain = [], []
    for _k in range(len(POS_LABELS)):
        _m = MF0[_sel] & (np.digitize(UF0[_sel], POS_BINS) - 1 == _k)
        if _m.sum() < 2000: _rc.append(np.nan); _gain.append(np.nan); continue
        _y = np.clip(YF0[_sel][_m], -2, 2); _a = np.clip(S4F0[_sel][_m], -2, 2); _b = np.clip(D7F0[_sel][_m], -2, 2)
        _w = np.abs(_y).mean(1)
        _ra = ((_y - _a) * [1, -1]).sum(1) / 2; _rb = ((_y - _b) * [1, -1]).sum(1) / 2
        _rc.append(float(np.cov(_ra, _rb, aweights=_w)[0, 1] / np.sqrt(np.cov(_ra, aweights=_w) * np.cov(_rb, aweights=_w))))
        _one = np.ones(_m.sum(), bool)
        _gain.append(wp_from(wsums(_y, (_a + _b) / 2, _one)) - max(wp_from(wsums(_y, _a, _one)), wp_from(wsums(_y, _b, _one))))
    _ax[0].plot(range(len(POS_LABELS)), _rc, "o-", ms=3, color=CLASS_COLORS[_c], label=f"class {_c}")
    _ax[1].plot(range(len(POS_LABELS)), 1e3 * np.array(_gain), "o-", ms=3, color=CLASS_COLORS[_c])
for _a in _ax: _a.set_xticks(range(len(POS_LABELS))); _a.set_xticklabels(POS_LABELS, fontsize=8); _a.set_xlabel("rows until next quote move")
_ax[0].set_title("residual correlation: since s4 vs distilled GRU256 α0.7 (fold 0)", fontsize=9); _ax[0].legend(fontsize=8)
_ax[1].set_title("WP gain ×1000 of their average over the better single member", fontsize=9); _ax[1].axhline(0, color="k", lw=0.6)
save(_fig, "F6_ensemble_diversity")
