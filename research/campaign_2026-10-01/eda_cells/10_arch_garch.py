# Volatility clustering in event time: ACF of squared (demeaned) series = ARCH effect; Engle ARCH-LM per sequence;
# GARCH(1,1) fits on run-level D and on the residual for a sample of sequences per class.
from statsmodels.stats.diagnostic import het_arch as _het_arch
from arch import arch_model as _arch_model
import warnings as _warnings
_fig, _ax = plt.subplots(1, 3, figsize=(15, 3.6))
GARCH = {}
for _c in range(4):
    _sq = []
    for _key, _ls in (("D", "-"), ("rD_start", "--")):
        _acc, _w = np.zeros(21), 0
        for _r in [r for r in RUNS if r["cls"] == _c][:250]:
            _x = _r[_key] - _r[_key].mean(); _x2 = _x * _x - (_x * _x).mean()
            if _x2.std() < 1e-9 or len(_x2) < 100: continue
            _acc += len(_x2) * sm_acf(_x2, nlags=20, fft=True); _w += len(_x2)
        _ax[0].plot(LAGS[1:], (_acc / max(_w, 1))[1:], _ls, color=CLASS_COLORS[_c], label=f"class {_c} {'target D' if _key=='D' else 'residual'}")
    _pv, _ab = [], []
    for _r in [r for r in RUNS if r["cls"] == _c][:60]:
        _x = _r["rD_start"]
        if len(_x) < 300 or _x.std() < 1e-6: continue
        _pv.append(_het_arch(_x - _x.mean(), nlags=10)[1])
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            try:
                _res = _arch_model(100 * (_x - _x.mean()), mean="Zero", vol="GARCH", p=1, q=1).fit(disp="off")
                _ab.append((_res.params["alpha[1]"], _res.params["beta[1]"]))
            except Exception:
                pass
    _ab = np.array(_ab)
    GARCH[_c] = {"arch_lm_share_p<0.01": float(np.mean(np.array(_pv) < 0.01)), "median_alpha": float(np.median(_ab[:, 0])),
                 "median_beta": float(np.median(_ab[:, 1])), "median_persistence": float(np.median(_ab.sum(1))), "n": len(_ab)}
    _ax[1].hist(_ab.sum(1), bins=20, range=(0, 1.05), histtype="step", color=CLASS_COLORS[_c], label=f"class {_c}")
    _ax[2].bar(_c, GARCH[_c]["arch_lm_share_p<0.01"], color=CLASS_COLORS[_c])
_ax[0].set_title("ACF of squared series (ARCH effect), event time"); _ax[0].legend(fontsize=7); _ax[0].set_xlabel("lag (runs)")
_ax[1].set_title("GARCH(1,1) persistence α+β of residual (per sequence)"); _ax[1].legend(fontsize=8)
_ax[2].set_title("share of sequences with ARCH-LM p<0.01 (residual)"); _ax[2].set_xticks(range(4))
save(_fig, "05_arch_garch")
mo.ui.table([{"class": c, **v} for c, v in GARCH.items()], selection=None)
