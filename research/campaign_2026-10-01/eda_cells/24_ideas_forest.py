# F4 — What tonight's ideas did: every screened mechanism with its paired chain-bootstrap delta (± 1.96 SD),
# grouped by verdict. Values come from the experiment ledger / screen reports (clean folds unless noted).
IDEAS = [
    ("distilled GRU256 vs since head (folds 2/3, prereg)", 0.002248, 0.000646, "supported"),
    ("+ distilled GRU256 trunk in pair (folds 2/3, prereg)", 0.000985, 0.000175, "supported"),
    ("distill α 0.7 vs 0.5, single (fold 0)", 0.002331, 0.001255, "promising"),
    ("routed since+d256 R4D vs incumbent recipe (4 folds)", 0.002002, 0.000422, "promising"),
    ("parent+FT head mix (folds 0/1, post hoc)", 0.000983, 0.000623, "not confirmed"),
    ("parent+FT head mix (folds 2/3, prereg)", -0.000075, 0.000407, "not confirmed"),
    ("class routing RSP2 (folds 2/3, prereg)", 0.000306, 0.000332, "not confirmed"),
    ("head-only vs two-stage FT (folds 0/1)", 0.000635, 0.000431, "not confirmed"),
    ("spread-label aux (fold 0)", 0.0000196, 0.0000825, "rejected"),
    ("per-class output scale (4 folds)", -0.00006, 0.0001, "rejected"),
    ("pooled-memory gcorr loss (folds 0/1)", -0.001521, 0.000674, "rejected"),
    ("joint ensemble head FT vs solo (folds 0/1)", -0.000099, 0.000031, "rejected"),
    ("cross-fitted blend weights (folds 0/1)", -0.000379, 0.000173, "rejected"),
    ("extra GRU seeds (holdout)", 0.000051, 0.000232, "rejected"),
    ("extra FT-seed heads (folds 0/1)", 0.000081, 0.00012, "rejected"),
    ("sequence-level de-meaning (4 folds)", -0.000053, 0.0001, "rejected"),
    ("class-3 queue-race history (4 folds)", 0.00022, 0.00017, "rejected"),
    ("matured-label feedback, deployable (4 folds)", -0.00023, 0.0002, "rejected"),
]
_col = {"supported": "#1d7a4a", "promising": "#2a78d6", "not confirmed": "#eda100", "rejected": "#b4372f"}
_fig, _ax = plt.subplots(figsize=(10, 7.5))
for _k, (_n, _d, _s, _v) in enumerate(IDEAS[::-1]):
    _ax.errorbar(1e3 * _d, _k, xerr=1e3 * 1.96 * _s, fmt="o", color=_col[_v], ms=5, capsize=2)
_ax.set_yticks(range(len(IDEAS))); _ax.set_yticklabels([i[0] for i in IDEAS[::-1]], fontsize=8)
_ax.axvline(0, color="k", lw=0.8); _ax.axvline(0.5, color="#999", lw=0.8, ls="--")
_ax.set_xlabel("Δ Global WP × 1000 (± 1.96 bootstrap SD); dashed = +0.0005 screen gate")
for _v, _cc in _col.items(): _ax.plot([], [], "o", color=_cc, label=_v)
_ax.legend(fontsize=8, loc="lower right"); _ax.set_title("What each tested idea did (local, clean chain-excluded folds)", fontsize=10)
save(_fig, "F4_ideas_forest")
