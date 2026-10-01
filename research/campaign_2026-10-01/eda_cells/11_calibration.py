# Reliability: |y|-weighted mean target vs prediction bins, per class and target (scored rows, OOF).
_fig, _ax = plt.subplots(1, 2, figsize=(12, 3.8))
for _t in range(2):
    for _c in range(4):
        _sel = CLS == _c
        _p = np.clip(SOTA[_sel][MASK[_sel]][:, _t], -2, 2); _y = np.clip(Y[_sel][MASK[_sel]][:, _t], -2, 2); _w = np.abs(_y)
        _q = np.quantile(_p, np.linspace(0, 1, 26)); _b = np.clip(np.searchsorted(_q, _p) - 1, 0, 24)
        _pm = np.bincount(_b, _w * _p, 25) / np.maximum(np.bincount(_b, _w, 25), 1e-9)
        _ym = np.bincount(_b, _w * _y, 25) / np.maximum(np.bincount(_b, _w, 25), 1e-9)
        _ax[_t].plot(_pm, _ym, "o-", ms=3, color=CLASS_COLORS[_c], label=f"class {_c}")
    _ax[_t].plot([-2, 2], [-2, 2], "k:", lw=0.8); _ax[_t].set_title(f"t{_t}: |y|-weighted mean target per prediction bin")
    _ax[_t].set_xlabel("prediction (bin mean)"); _ax[_t].set_ylabel("target (bin mean)")
_ax[0].legend(fontsize=8)
save(_fig, "06_calibration")
