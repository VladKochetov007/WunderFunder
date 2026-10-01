# Per-sequence picture: within-sequence WP distribution, SOTA vs reference gain per sequence, worst sequences.
_seq_wp = np.array([wp_from(s) if s[..., 0].min() > 0 else np.nan for s in SEQ_SOTA])
_seq_ref = np.array([wp_from(s) if s[..., 0].min() > 0 else np.nan for s in SEQ_REF])
_fig, _ax = plt.subplots(1, 3, figsize=(15, 3.6))
for _c in range(4):
    _v = _seq_wp[(CLS == _c) & np.isfinite(_seq_wp)]
    _ax[0].hist(_v, bins=40, range=(-0.5, 1), histtype="step", density=True, color=CLASS_COLORS[_c], label=f"class {_c} median {np.median(_v):.2f}")
    _d = (_seq_wp - _seq_ref)[(CLS == _c) & np.isfinite(_seq_wp)]
    _ax[1].hist(_d, bins=40, range=(-0.05, 0.05), histtype="step", color=CLASS_COLORS[_c])
_ax[0].set_title("within-sequence WP (SOTA OOF)"); _ax[0].legend(fontsize=7)
_ax[1].set_title("per-sequence WP gain: SOTA − reference"); _ax[1].axvline(0, color="k", lw=0.6)
_w = SEQ_SOTA[..., 0, 0]
_ax[2].scatter(np.log10(_w + 1), _seq_wp, s=4, c=[CLASS_COLORS[c] for c in CLS], alpha=0.5)
_ax[2].set_title("sequence WP vs log10 metric weight"); _ax[2].set_xlabel("log10(Σ|y| on scored rows)")
save(_fig, "08_per_sequence")
_worst = np.argsort(np.nan_to_num(_seq_wp, nan=9))[:12]
mo.ui.table([{"seq_row": int(i), "class": int(CLS[i]), "fold": int(FOLD[i]), "WP": round(float(_seq_wp[i]), 3),
              "ref WP": round(float(_seq_ref[i]), 3), "weight": round(float(_w[i]), 1)} for i in _worst], selection=None)
