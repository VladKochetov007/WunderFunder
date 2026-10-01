# Test of the F7 lever: a causal magnitude correction p' = s[class, since-bin] · p, with s fitted on three folds
# (maximising pooled WP of those folds) and applied to the held-out fold. 4 classes × 6 since-bins = 24 parameters.
from scipy.optimize import minimize as _minimize
_sb = np.digitize(SINCE, POS_BINS) - 1
def _cell_sums(rows, scales):
    _acc = np.zeros((2, 6))
    for _i in rows:
        _m = MASK[_i]
        if not _m.any(): continue
        _sc = scales[CLS[_i], _sb[_i]][_m][:, None]
        _acc += wsums(Y[_i][_m], (SOTA[_i][_m] * _sc), np.ones(_m.sum(), bool))
    return _acc
# precompute per (class, bin) second moments so the objective is cheap: z = [1, y, p] with |y| weights, per target
_mom = np.zeros((4, 4, len(POS_LABELS), 2, 3, 3))   # fold, class, bin, target, 3x3
for _i in range(len(Y)):
    _m = MASK[_i]
    if not _m.any(): continue
    _y = np.clip(Y[_i][_m], -2, 2).astype(np.float64); _p = SOTA[_i][_m].astype(np.float64); _b = _sb[_i][_m]
    for _k in np.unique(_b):
        _s = _b == _k
        for _t in range(2):
            _z = np.stack([np.ones(_s.sum()), _y[_s, _t], _p[_s, _t]], 1); _w = np.abs(_y[_s, _t])
            _mom[FOLD[_i], CLS[_i], _k, _t] += (_z * _w[:, None]).T @ _z
def _wp_scaled(mom, scales):
    # pooled corr with p scaled per (class, bin); clipping ignored in fitting (scales near 1)
    _r = []
    for _t in range(2):
        _S = np.zeros(6)
        for _c in range(4):
            for _k in range(len(POS_LABELS)):
                _M = mom[_c, _k, _t]; _a = scales[_c, _k]
                _S += [_M[0, 0], _M[0, 1], _a * _M[0, 2], _M[1, 1], _a * _a * _M[2, 2], _a * _M[1, 2]]
        n, sy, sp, syy, spp, syp = _S
        _r.append((syp/n - sy*sp/n**2) / np.sqrt((syy/n - (sy/n)**2) * (spp/n - (sp/n)**2)))
    return float(np.mean(_r))
SINCE_SCALE = {}
for _f in range(4):
    _train = _mom[[g for g in range(4) if g != _f]].sum(0)
    _obj = lambda v: -_wp_scaled(_train, np.exp(v.reshape(4, len(POS_LABELS))))
    _res = _minimize(_obj, np.zeros(4 * len(POS_LABELS)), method="L-BFGS-B")
    _sc = np.exp(_res.x.reshape(4, len(POS_LABELS)))
    SINCE_SCALE[_f] = {"base": _wp_scaled(_mom[_f], np.ones((4, len(POS_LABELS)))), "scaled": _wp_scaled(_mom[_f], _sc), "scales": np.round(_sc, 3)}
_d = [SINCE_SCALE[f]["scaled"] - SINCE_SCALE[f]["base"] for f in range(4)]
mo.md(f"**Causal since-bin scale correction, cross-fitted:** fold deltas ×1000 = {np.round(1e3*np.array(_d), 3).tolist()}, mean {1e3*np.mean(_d):+.3f}.  "
      f"Fitted scales (fold 0, rows: class, cols: since bins {POS_LABELS}):\n\n```\n{SINCE_SCALE[0]['scales']}\n```")
