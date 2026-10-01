# Run-level (event-time) series per sequence. Run k = rows [e_k, e_{k+1}); the target is constant on a run.
# D = (t0 - t1)/2 is the future direction (~80% of metric variance), S = t0 + t1 the spread-like level.
# Residual uses a per-class |y|-weighted OLS scale so it is orthogonal to the prediction in the metric's geometry.
def _beta(cls):
    _s = SEQ_SOTA[CLS == cls].sum(0)
    n, sy, sp, syy, spp, syp = np.moveaxis(_s, -1, 0)
    return (syp/n - sy*sp/n**2) / (spp/n - (sp/n)**2)
BETA = np.stack([_beta(c) for c in range(4)])      # (class, target)
RUNS = []
for _i in range(len(Y)):
    _e = np.flatnonzero(EVENT[_i, 99:]) + 99
    if len(_e) < 30: continue
    _y = np.clip(Y[_i, _e].astype(np.float64), -2, 2)
    _pend = np.clip(SOTA[_i, np.r_[_e[1:] - 1, 19999]].astype(np.float64), -2, 2)
    _pstart = np.clip(SOTA[_i, _e].astype(np.float64), -2, 2)
    _b = BETA[CLS[_i]]
    RUNS.append({"i": _i, "cls": int(CLS[_i]), "len": np.diff(np.r_[_e, 20000]),
                 "D": (_y[:, 0] - _y[:, 1]) / 2, "S": _y.sum(1),
                 "rD_start": (_y[:, 0] - _b[0]*_pstart[:, 0] - (_y[:, 1] - _b[1]*_pstart[:, 1])) / 2,
                 "rD_end": (_y[:, 0] - _b[0]*_pend[:, 0] - (_y[:, 1] - _b[1]*_pend[:, 1])) / 2})
mo.md(f"Event-time series built for **{len(RUNS)}** sequences; per-class residual scale β (t0, t1): {np.round(BETA, 3).tolist()}")
