# F3 — Error budget: where is the lost metric? Loss = Σ|y|(y − βp)² with β the per-class OLS scale (metric geometry),
# summed over scored rows of the SOTA OOF. Panels: share of total loss by class × run length, and by class ×
# rows until the next quote move; annotated with the segment's WP so "big and bad" stands out.
_rl_bins = np.array([0, 4, 16, 64, 256, 1024, 20001]); _rl_lab = ["<4", "4–16", "16–64", "64–256", "256–1024", "≥1024"]
_loss = np.zeros((4, len(_rl_lab))); _loss2 = np.zeros((4, len(POS_LABELS)))
_wp1 = np.full((4, len(_rl_lab)), np.nan); _wp2 = np.full((4, len(POS_LABELS)), np.nan)
for _c in range(4):
    _sel = np.flatnonzero(CLS == _c)
    _m = MASK[_sel]; _y = np.clip(Y[_sel][_m], -2, 2).astype(np.float64); _p = np.clip(SOTA[_sel][_m], -2, 2).astype(np.float64)
    _l = (np.abs(_y) * (_y - BETA[_c] * _p) ** 2).sum(1)
    _b1 = np.digitize(RUNLEN[_sel][_m], _rl_bins) - 1; _b2 = np.digitize(UNTIL[_sel][_m], POS_BINS) - 1
    for _k in range(len(_rl_lab)):
        _loss[_c, _k] = _l[_b1 == _k].sum()
        if (_b1 == _k).sum() > 2000: _wp1[_c, _k] = wp_from(wsums(_y[_b1 == _k], _p[_b1 == _k], np.ones((_b1 == _k).sum(), bool)))
    for _k in range(len(POS_LABELS)):
        _loss2[_c, _k] = _l[_b2 == _k].sum()
        if (_b2 == _k).sum() > 2000: _wp2[_c, _k] = wp_from(wsums(_y[_b2 == _k], _p[_b2 == _k], np.ones((_b2 == _k).sum(), bool)))
_tot = _loss.sum()
_fig, _ax = plt.subplots(1, 2, figsize=(15, 4.2))
for _j, (_L, _W, _lab, _title) in enumerate(((_loss, _wp1, _rl_lab, "run length (rows)"), (_loss2, _wp2, POS_LABELS, "rows until the next quote move"))):
    _im = _ax[_j].imshow(100 * _L / _tot, cmap="Reds", aspect="auto")
    for _c in range(4):
        for _k in range(len(_lab)):
            _ax[_j].text(_k, _c, f"{100*_L[_c,_k]/_tot:.1f}%\nWP {_W[_c,_k]:.2f}" if np.isfinite(_W[_c, _k]) else f"{100*_L[_c,_k]/_tot:.1f}%",
                         ha="center", va="center", fontsize=8)
    _ax[_j].set_xticks(range(len(_lab))); _ax[_j].set_xticklabels(_lab); _ax[_j].set_yticks(range(4)); _ax[_j].set_yticklabels([f"class {c}" for c in range(4)])
    _ax[_j].set_xlabel(_title); _ax[_j].grid(False)
    _ax[_j].set_title(f"share of total loss by class × {_title} (cell: loss share, segment WP)", fontsize=9)
save(_fig, "F3_error_budget")
