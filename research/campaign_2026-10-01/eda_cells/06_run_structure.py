def run_bounds(i):
    """Event rows (quote-move markers) from row 99 on; runs are [e_k, e_{k+1})."""
    return np.flatnonzero(EVENT[i, 99:]) + 99
_fig, _ax = plt.subplots(1, 2, figsize=(12, 3.4))
for _c in range(4):
    _seqs = np.flatnonzero(CLS == _c)[:150]
    _rl = np.concatenate([np.diff(run_bounds(i)) for i in _seqs])
    _ax[0].hist(np.log10(_rl), bins=60, histtype="step", density=True, color=CLASS_COLORS[_c], label=f"class {_c}: median {int(np.median(_rl))} rows")
_ax[0].set_title("run length between quote-move events (log10 rows)"); _ax[0].legend(fontsize=8)
_chg = (Y[:, 1:] != Y[:, :-1]).any(-1)
_on = (_chg & EVENT[:, 1:]).sum() / max(_chg.sum(), 1)
_ax[1].bar(range(4), [EVENT[CLS == c].mean() for c in range(4)], color=CLASS_COLORS); _ax[1].set_xticks(range(4))
_ax[1].set_xlabel("class"); _ax[1].set_title(f"event-row share (target changes on event rows: {_on:.4f})")
save(_fig, "02_run_structure")
