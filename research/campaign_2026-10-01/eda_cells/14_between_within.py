# Pooled covariance decomposition per class: between-sequence (sequence means) vs within-sequence parts.
_rows = []
for _c in range(4):
    for _t in range(2):
        _s = SEQ_SOTA[CLS == _c][:, _t]                       # per-sequence raw sums [w, wy, wp, wyy, wpp, wyp]
        _n = _s[:, 0]; _my = _s[:, 1] / np.maximum(_n, 1e-9); _mp = _s[:, 2] / np.maximum(_n, 1e-9)
        _N = _n.sum(); _My = _s[:, 1].sum() / _N; _Mp = _s[:, 2].sum() / _N
        _cov_b = (_n * (_my - _My) * (_mp - _Mp)).sum() / _N
        _cov_w = (_s[:, 5] - _n * _my * _mp).sum() / _N
        _vy_b = (_n * (_my - _My) ** 2).sum() / _N; _vp_b = (_n * (_mp - _Mp) ** 2).sum() / _N
        _vy = _s[:, 3].sum() / _N - _My ** 2; _vp = _s[:, 4].sum() / _N - _Mp ** 2
        _rows.append({"class": _c, "target": f"t{_t}", "cov share between": round(_cov_b / (_cov_b + _cov_w), 3),
                      "y var share between": round(_vy_b / _vy, 3), "p var share between": round(_vp_b / _vp, 3),
                      "corr between": round(_cov_b / np.sqrt(_vy_b * _vp_b), 3),
                      "corr within": round(_cov_w / np.sqrt((_vy - _vy_b) * (_vp - _vp_b)), 3)})
BETWEEN_WITHIN = _rows
mo.ui.table(_rows, selection=None)
