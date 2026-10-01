# F5 — Why does distillation help? Fold 0 has α 0.5 and α 0.7 students. Compare WP of: since s4 (merged heads),
# distilled α 0.5, distilled α 0.7 — by class × rows until the next quote move. Where the α gain appears tells
# whether distillation denoises long-horizon (future-path) labels or sharpens near-event predictions.
_f0 = np.flatnonzero(FOLD == 0)
def _avg(names):
    _arrs = [np.load(ROOT / f"cache/pred_{n}_valid.npy", mmap_mode="r") for n in names]
    return np.stack([sum(np.asarray(a[i], np.float32) for a in _arrs) / len(_arrs) for i in _f0])
S4F0 = _avg(["c20260929_since_s4_f0", "c20260929_ft_since_s4_f0h"])
D5F0 = _avg(["c20261001_d256_s3_f0", "c20261001_ft_d256_s3_f0h"])
D7F0 = _avg(["c20261001_d256a07_s3_f0", "c20261001_ft_d256a07_s3_f0h"])
YF0, MF0, UF0, CF0 = Y[_f0], MASK[_f0], UNTIL[_f0], CLS[_f0]
_fig, _ax = plt.subplots(1, 4, figsize=(17, 3.8), sharey=False)
DISTILL = {}
_ref = {}
for _c in range(4):
    _sel = CF0 == _c
    for _name, _P, _col in (("since s4 (merged heads)", S4F0, "#1baf7a"), ("distilled α 0.5", D5F0, "#eb6834"), ("distilled α 0.7", D7F0, "#2a78d6")):
        _vals = []
        for _k in range(len(POS_LABELS)):
            _m = MF0[_sel] & (np.digitize(UF0[_sel], POS_BINS) - 1 == _k)
            _vals.append(wp_from(wsums(YF0[_sel], _P[_sel], _m)) if _m.sum() > 2000 else np.nan)
        DISTILL[(_c, _name)] = _vals
        if _name.startswith("since"): _ref[_c] = np.array(_vals); continue
        _ax[_c].plot(range(len(POS_LABELS)), 1e3 * (np.array(_vals) - _ref[_c]), "o-", ms=3, color=_col, label=f"{_name} − since s4")
    _ax[_c].set_xticks(range(len(POS_LABELS))); _ax[_c].set_xticklabels(POS_LABELS, fontsize=7, rotation=30)
    _ax[_c].axhline(0, color="k", lw=0.6)
    _ax[_c].set_title(f"class {_c} (fold 0): segment WP gain ×1000 vs since s4", fontsize=9)
_ax[0].legend(fontsize=7); _ax[0].set_ylabel("Δ segment WP × 1000")
save(_fig, "F5_distillation_mechanism")
