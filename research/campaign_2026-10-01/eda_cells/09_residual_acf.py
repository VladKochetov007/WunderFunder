_fig, _ax = plt.subplots(2, 3, figsize=(15, 6.4))
RESID_ACF = {}
for _c in range(4):
    for _j, _key in enumerate(["rD_start", "rD_end"]):
        _a = mean_acf(_key, _c); _p = mean_acf(_key, _c, fn="pacf")
        RESID_ACF[(_c, _key)] = (_a, _p)
        _ax[0, _j].plot(LAGS[1:], _a[1:], "o-", ms=3, color=CLASS_COLORS[_c], label=f"class {_c}")
        _ax[1, _j].plot(LAGS[1:], _p[1:], "o-", ms=3, color=CLASS_COLORS[_c])
_ax[0, 0].set_title("residual ACF, p at run START (event time)"); _ax[1, 0].set_title("residual PACF, p at run START")
_ax[0, 1].set_title("residual ACF, p at run END (event time)"); _ax[1, 1].set_title("residual PACF, p at run END")
# row-time ACF of the scored-row residual (log-spaced lags), class means over sequences
_lags = np.unique(np.logspace(0, 3.3, 28).astype(int))
for _c in range(4):
    _acc = np.zeros(len(_lags)); _n = 0
    for _i in np.flatnonzero(CLS == _c)[:200]:
        _p = SOTA[_i, 99:].astype(np.float64); _y = np.clip(Y[_i, 99:].astype(np.float64), -2, 2)
        _r = ((_y[:, 0] - BETA[_c, 0]*_p[:, 0]) - (_y[:, 1] - BETA[_c, 1]*_p[:, 1])) / 2
        _r = _r - _r.mean(); _v = (_r * _r).mean()
        if _v < 1e-12: continue
        _acc += np.array([(_r[:-L] * _r[L:]).mean() / _v for L in _lags]); _n += 1
    _ax[0, 2].semilogx(_lags, _acc / max(_n, 1), "-", color=CLASS_COLORS[_c], label=f"class {_c}")
    _ys = np.clip(Y[np.flatnonzero(CLS == _c)[:200], 99:], -2, 2).astype(np.float64)
    _d = (_ys[..., 0] - _ys[..., 1]) / 2; _d = _d - _d.mean(1, keepdims=True); _vv = (_d*_d).mean(1, keepdims=True) + 1e-12
    _ax[1, 2].semilogx(_lags, [((_d[:, :-L]*_d[:, L:]).mean(1, keepdims=True) / _vv).mean() for L in _lags], "-", color=CLASS_COLORS[_c])
_ax[0, 2].set_title("row-time ACF of residual D (all rows ≥ 99)"); _ax[1, 2].set_title("row-time ACF of target D")
for _a in _ax.ravel(): _a.axhline(0, color="k", lw=0.6)
_ax[0, 0].legend(fontsize=8); _ax[1, 2].set_xlabel("lag (rows)"); _ax[1, 0].set_xlabel("lag (runs)"); _ax[1, 1].set_xlabel("lag (runs)")
save(_fig, "04_residual_acf_pacf")
