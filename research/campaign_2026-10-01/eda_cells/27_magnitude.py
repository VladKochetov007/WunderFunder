# F7 — Magnitude allocation. Pooled WP rewards predictions whose scale tracks how predictable each segment is.
# For each segment, β_seg = cov_w(y,p)/var_w(p) is the scale that segment "wants"; β_seg/β_class = 1 everywhere means
# the model spends its prediction magnitude where the signal is. Compare since s4, distilled α 0.5 and α 0.7 (fold 0),
# by rows until the next move (explanatory, non-causal) and by rows since the last move (what a model can see).
def _beta_seg(P, m, cls_sel):
    _y = np.clip(YF0[cls_sel][m], -2, 2).astype(np.float64); _p = np.clip(P[cls_sel][m], -2, 2).astype(np.float64)
    _b = []
    for _t in range(2):
        _w = np.abs(_y[:, _t])
        _mp = (_w * _p[:, _t]).sum() / _w.sum(); _my = (_w * _y[:, _t]).sum() / _w.sum()
        _b.append(((_w * (_p[:, _t] - _mp) * (_y[:, _t] - _my)).sum()) / ((_w * (_p[:, _t] - _mp) ** 2).sum()))
    return float(np.mean(_b))
SINCEF0 = SINCE[np.flatnonzero(FOLD == 0)]
_fig, _ax = plt.subplots(2, 4, figsize=(17, 7))
MAGNITUDE = {}
for _c in range(4):
    _sel = CF0 == _c
    for _row, (_arr, _lab) in enumerate(((UF0, "rows until next move"), (SINCEF0, "rows since last move"))):
        for _name, _P, _col in (("since s4", S4F0, "#1baf7a"), ("distilled α 0.5", D5F0, "#eb6834"), ("distilled α 0.7", D7F0, "#2a78d6")):
            _bc = _beta_seg(_P, MF0[_sel], _sel)
            _vals = []
            for _k in range(len(POS_LABELS)):
                _m = MF0[_sel] & (np.digitize(_arr[_sel], POS_BINS) - 1 == _k)
                _vals.append(_beta_seg(_P, _m, _sel) / _bc if _m.sum() > 2000 else np.nan)
            MAGNITUDE[(_c, _lab, _name)] = _vals
            _ax[_row, _c].plot(range(len(POS_LABELS)), _vals, "o-", ms=3, color=_col, label=_name)
        _ax[_row, _c].axhline(1, color="k", lw=0.6)
        _ax[_row, _c].set_xticks(range(len(POS_LABELS))); _ax[_row, _c].set_xticklabels(POS_LABELS, fontsize=7, rotation=30)
        _ax[_row, _c].set_title(f"class {_c}: wanted scale / class scale, by {_lab}", fontsize=9)
_ax[0, 0].legend(fontsize=7); _ax[0, 0].set_ylabel("β_segment / β_class  (1 = well allocated)")
_ax[1, 0].set_ylabel("β_segment / β_class")
save(_fig, "F7_magnitude_allocation")
