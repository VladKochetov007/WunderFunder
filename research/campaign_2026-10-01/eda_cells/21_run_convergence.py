# F2 — Does the model learn the run's outcome while the run is in progress? Fixed cohorts avoid survivorship bias:
# take only runs that last at least L rows and track corr(prediction, run target D) at offsets 0..L inside those runs.
# Flat curve = no information accrues inside the run; rising curve = the model reads the queue race as it unfolds.
def _cohort_curve(pred, cls, min_len, offsets, max_seqs=400):
    _num = np.zeros((len(offsets), 6))
    for _i in np.flatnonzero(CLS == cls)[:max_seqs]:
        _e = np.flatnonzero(EVENT[_i, 99:]) + 99
        _len = np.diff(np.r_[_e, 20000]); _keep = _e[_len >= min_len]
        if len(_keep) == 0: continue
        _yD = (np.clip(Y[_i, _keep, 0], -2, 2) - np.clip(Y[_i, _keep, 1], -2, 2)) / 2
        _pD = (np.clip(pred[_i, :, 0], -2, 2) - np.clip(pred[_i, :, 1], -2, 2)) / 2
        _y = _yD.astype(np.float64); _w = np.abs(_y)
        for _k, _o in enumerate(offsets):
            _p = _pD[np.minimum(_keep + _o, 19999)].astype(np.float64)
            _num[_k] += [_w.sum(), (_w*_y).sum(), (_w*_p).sum(), (_w*_y*_y).sum(), (_w*_p*_p).sum(), (_w*_y*_p).sum()]
    n, sy, sp, syy, spp, syp = _num.T
    return (syp/n - sy*sp/n**2) / np.sqrt((syy/n - (sy/n)**2) * (spp/n - (sp/n)**2)), int(n[0] > 0)
_parent = np.zeros_like(SOTA)
for _f in range(4):
    _pp = np.load(ROOT / f"cache/pred_c20260929_since_s4_f{_f}_valid.npy", mmap_mode="r")
    for _i in np.flatnonzero(FOLD == _f): _parent[_i] = _pp[_i]
_fig, _ax = plt.subplots(2, 4, figsize=(17, 7), sharey="row")
CONV = {}
for _row, (_L, _offs) in enumerate(((64, np.array([0, 1, 2, 4, 8, 16, 32, 63])), (256, np.array([0, 2, 8, 16, 32, 64, 128, 255])))):
    for _c in range(4):
        for _name, _pred, _col in (("SOTA", SOTA, "#2a78d6"), ("reference", REF, "#eb6834"), ("since parent (no FT)", _parent, "#1baf7a")):
            _r, _ = _cohort_curve(_pred, _c, _L, _offs); CONV[(_L, _c, _name)] = _r
            _ax[_row, _c].plot(range(len(_offs)), _r, "o-", ms=3, color=_col, label=_name)
        _ax[_row, _c].set_xticks(range(len(_offs))); _ax[_row, _c].set_xticklabels(_offs, fontsize=7)
        _ax[_row, _c].set_title(f"class {_c}, runs lasting ≥ {_L} rows", fontsize=9)
        _ax[_row, _c].set_xlabel("rows after the run's starting quote move")
    _ax[_row, 0].set_ylabel("|y|-weighted corr(pred, run D)")
_ax[0, 0].legend(fontsize=7)
save(_fig, "F2_run_convergence")
