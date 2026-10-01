_rows = []
for _c in [0, 1, 2, 3, "all"]:
    _sel = np.ones(len(Y), bool) if _c == "all" else CLS == _c
    _w = SEQ_SOTA[_sel][..., 0].sum() / SEQ_SOTA[..., 0].sum()
    _rows.append({"class": _c, "sequences": int(_sel.sum()), "metric weight share": round(float(_w), 3),
                  "WP reference (incumbent recipe)": round(wp_from(SEQ_REF[_sel].sum(0)), 5),
                  "WP SOTA (R4D analog)": round(wp_from(SEQ_SOTA[_sel].sum(0)), 5),
                  "delta": round(wp_from(SEQ_SOTA[_sel].sum(0)) - wp_from(SEQ_REF[_sel].sum(0)), 5)})
SUMMARY = _rows
mo.vstack([mo.md("## Summary — clean out-of-fold, 4 folds, Global WP (pooled, |y|-weighted, clip ±2)"), mo.ui.table(_rows, selection=None)])
