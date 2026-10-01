from statsmodels.tsa.stattools import acf as sm_acf, pacf as sm_pacf
def mean_acf(key, cls, nlags=20, fn="acf", max_seqs=250):
    """Length-weighted average of per-sequence ACF/PACF of an event-time series (sequence boundaries respected)."""
    _acc, _w = np.zeros(nlags + 1), 0.0
    for _r in [r for r in RUNS if r["cls"] == cls][:max_seqs]:
        _x = _r[key]
        if len(_x) < 4 * nlags or np.std(_x) < 1e-9: continue
        _v = sm_acf(_x, nlags=nlags, fft=True) if fn == "acf" else sm_pacf(_x, nlags=nlags, method="ywm")
        _acc += len(_x) * _v; _w += len(_x)
    return _acc / max(_w, 1)
LAGS = np.arange(21)
_fig, _ax = plt.subplots(2, 2, figsize=(13, 6.4), sharex=True)
TARGET_ACF = {}
for _c in range(4):
    for _j, _key in enumerate(["D", "S"]):
        _a = mean_acf(_key, _c); _p = mean_acf(_key, _c, fn="pacf")
        TARGET_ACF[(_c, _key)] = (_a, _p)
        _ax[0, _j].plot(LAGS[1:], _a[1:], "o-", ms=3, color=CLASS_COLORS[_c], label=f"class {_c}")
        _ax[1, _j].plot(LAGS[1:], _p[1:], "o-", ms=3, color=CLASS_COLORS[_c])
for _j, _key in enumerate(["D (direction)", "S (t0+t1, spread-like)"]):
    _ax[0, _j].set_title(f"target ACF in event time — {_key}"); _ax[1, _j].set_title(f"target PACF in event time — {_key}")
    _ax[1, _j].set_xlabel("lag (quote-move runs)")
    for _r in range(2): _ax[_r, _j].axhline(0, color="k", lw=0.6)
_ax[0, 0].legend(fontsize=8)
save(_fig, "03_target_acf_pacf_event_time")
