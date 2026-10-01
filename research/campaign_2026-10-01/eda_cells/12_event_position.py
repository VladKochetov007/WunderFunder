# Where inside a run is the metric lost? WP by rows since the last quote move (causal) and rows until the next one (not causal).
_bins = np.array([0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 20001])
_since = np.zeros(Y.shape[:2], np.int32); _until = np.zeros(Y.shape[:2], np.int32)
for _i in range(len(Y)):
    _e = np.flatnonzero(EVENT[_i]); _idx = np.arange(20000)
    _k = np.searchsorted(_e, _idx, side="right") - 1
    _since[_i] = np.where(_k >= 0, _idx - _e[np.maximum(_k, 0)], 9999)
    _k2 = np.searchsorted(_e, _idx, side="right")
    _until[_i] = np.where(_k2 < len(_e), _e[np.minimum(_k2, len(_e) - 1)] - _idx, 9999)
_fig, _ax = plt.subplots(1, 3, figsize=(15, 3.6))
for _j, (_arr, _lab) in enumerate(((_since, "rows since last quote move (causal)"), (_until, "rows until next quote move (non-causal)"))):
    for _c in range(4):
        _sel = CLS == _c
        _bb = np.digitize(_arr[_sel], _bins) - 1
        _vals, _wts = [], []
        for _k in range(len(_bins) - 1):
            _m = MASK[_sel] & (_bb == _k)
            if _m.sum() < 500: _vals.append(np.nan); _wts.append(0); continue
            _vals.append(wp_from(wsums(Y[_sel], SOTA[_sel], _m))); _wts.append(np.abs(np.clip(Y[_sel][_m], -2, 2)).sum())
        _ax[_j].plot(range(len(_vals)), _vals, "o-", ms=3, color=CLASS_COLORS[_c], label=f"class {_c}")
        if _j == 1:
            _ax[2].plot(range(len(_wts)), np.array(_wts) / max(sum(_wts), 1), "o-", ms=3, color=CLASS_COLORS[_c])
    _ax[_j].set_xticks(range(len(_bins) - 1)); _ax[_j].set_xticklabels([f"{a}" for a in _bins[:-1]], rotation=45, fontsize=7)
    _ax[_j].set_title(f"WP by {_lab}"); _ax[_j].set_xlabel("bin lower edge (rows)")
_ax[2].set_xticks(range(len(_bins) - 1)); _ax[2].set_xticklabels([f"{a}" for a in _bins[:-1]], rotation=45, fontsize=7)
_ax[2].set_title("metric weight share by rows until next move"); _ax[0].legend(fontsize=8)
save(_fig, "07_wp_by_event_position")
