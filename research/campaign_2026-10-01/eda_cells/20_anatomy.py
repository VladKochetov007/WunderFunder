# F1 — Anatomy of real sequences: targets (steps) vs SOTA and reference predictions, quote-move events, scored rows.
# One window per class, chosen at a typical-WP sequence so the picture is representative, not cherry-picked.
def _seq_wp(i):
    s = SEQ_SOTA[i]
    return wp_from(s) if s[..., 0].min() > 0 else np.nan
_fig, _axes = plt.subplots(4, 1, figsize=(15, 13), sharex=False)
for _c, _ax in zip(range(4), _axes):
    _cand = np.flatnonzero(CLS == _c)
    _w = np.array([_seq_wp(i) for i in _cand])
    _i = int(_cand[np.nanargmin(np.abs(_w - np.nanmedian(_w)))])          # median-WP sequence of the class
    _scored = np.flatnonzero(MASK[_i])
    _start = int(np.clip(_scored[len(_scored) // 2] - 600, 99, 20000 - 1200)); _sl = slice(_start, _start + 1200)
    _t = np.arange(_sl.start, _sl.stop)
    _D = lambda a: (np.clip(a[_sl, 0], -2, 2) - np.clip(a[_sl, 1], -2, 2)) / 2
    _ax.fill_between(_t, -2.2, 2.2, where=MASK[_i, _sl], color="#999", alpha=0.12, step="mid", label="scored rows")
    _ax.step(_t, _D(Y[_i]), where="post", color="k", lw=1.6, label="target D=(t0−t1)/2")
    _ax.plot(_t, _D(REF[_i]), color="#eb6834", lw=0.9, alpha=0.9, label="reference (incumbent recipe)")
    _ax.plot(_t, _D(SOTA[_i]), color="#2a78d6", lw=1.1, label="SOTA (R4D analog)")
    _ev = _t[EVENT[_i, _sl]]
    _ax.vlines(_ev, -2.2, -1.95, color="#e34948", lw=0.6, label="quote-move event")
    _ax.set_ylim(-2.25, 2.25); _ax.set_xlim(_t[0], _t[-1])
    _ax.set_title(f"class {_c} — sequence row {_i} (fold {FOLD[_i]}), sequence WP {_seq_wp(_i):.3f}, rows {_sl.start}–{_sl.stop}", fontsize=10)
    _ax.set_ylabel("D")
_axes[0].legend(fontsize=8, ncol=5, loc="upper left")
_axes[-1].set_xlabel("row (step_in_seq)")
save(_fig, "F1_sequence_anatomy")
