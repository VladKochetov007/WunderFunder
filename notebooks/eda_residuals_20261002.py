import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")

with app.setup(hide_code=True):

    import json, sys
    from pathlib import Path
    import numpy as np
    import marimo as mo
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ROOT = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
    sys.path.insert(0, str(ROOT / "src"))
    from wnn.splits import load_campaign_split
    PLOTS = ROOT / "research/campaign_2026-10-01/plots/eda"
    PLOTS.mkdir(parents=True, exist_ok=True)
    CLASS_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#e34948"]
    plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
                         "axes.spines.right": False, "font.size": 9})
    def save(fig, name):
        """Persist every figure as PNG in the research folder (notebook state is not the record)."""
        fig.savefig(PLOTS / f"{name}.png", bbox_inches="tight")
        return fig



@app.cell(hide_code=True)
def data():

    _z = np.load(ROOT / "cache/valid_targets.npz")
    Y = _z["y"]; MASK = _z["mask"].copy(); MASK[:, :99] = False
    CLS = _z["market_class"].astype(int); ROWG = _z["row_group"].astype(int)
    _pos = {int(g): i for i, g in enumerate(ROWG)}
    FOLD = np.full(len(ROWG), -1); COMP = np.full(len(ROWG), -1)
    for _f in range(4):
        _sp = load_campaign_split(ROOT / f"research/campaign_2026-09-29/artifacts/splits/fold{_f}.json", ROOT)
        _cm = {g: c for s, g, c in _sp.components if s == "valid"}
        for _g in _sp.evaluation_groups:
            FOLD[_pos[_g]] = _f; COMP[_pos[_g]] = _cm[_g]
    def _load(name): return np.load(ROOT / f"cache/pred_{name}_valid.npy", mmap_mode="r")
    SOTA = np.zeros(Y.shape, np.float32); REF = np.zeros(Y.shape, np.float32)
    for _f in range(4):
        _rows = np.flatnonzero(FOLD == _f)
        _ft4 = _load(f"c20260929_ft_since_s4_f{_f}"); _ft3 = _load(f"c20260929_ft_since_s3_f{_f}")
        _p4 = _load(f"c20260929_since_s4_f{_f}"); _h4 = _load(f"c20260929_ft_since_s4_f{_f}h")
        _dp = _load(f"c20261001_d256_s3_f{_f}"); _dh = _load(f"c20261001_ft_d256_s3_f{_f}h")
        for _i in _rows:
            _d = 0.5 * (np.asarray(_dp[_i], np.float32) + np.asarray(_dh[_i], np.float32))
            if CLS[_i] == 3:
                _s = 0.5 * (np.asarray(_p4[_i], np.float32) + np.asarray(_h4[_i], np.float32))
            else:
                _s = np.asarray(_ft4[_i], np.float32)
            SOTA[_i] = 0.5 * (_s + _d)
            REF[_i] = 0.5 * (np.asarray(_ft3[_i], np.float32) + np.asarray(_ft4[_i], np.float32))
    import pyarrow.parquet as _pq
    _pf = _pq.ParquetFile(ROOT / "datasets/valid.parquet")
    A6 = _pf.read(columns=["a6"]).column(0).to_numpy().reshape(-1, 20000).astype(np.float32)
    A6_END = float(A6.min())
    EVENT = np.isclose(A6, A6_END, atol=1e-6)   # quote-move marker: targets change exactly on these rows
    print(Y.shape, SOTA.dtype, "a6 endpoint", A6_END, "event share", EVENT.mean().round(4), "folds", np.bincount(FOLD + 1))

    return CLS, EVENT, FOLD, MASK, REF, SOTA, Y


@app.cell(hide_code=True)
def metric(MASK, REF, SOTA, Y):
    def wsums(y, p, m):
        """|y|-weighted raw sums per target over selected rows (clip ±2): [w, wy, wp, wyy, wpp, wyp]."""
        yy = np.clip(y[m].astype(np.float64), -2, 2); pp = np.clip(p[m].astype(np.float64), -2, 2); w = np.abs(yy)
        return np.stack([w.sum(0), (w*yy).sum(0), (w*pp).sum(0), (w*yy*yy).sum(0), (w*pp*pp).sum(0), (w*yy*pp).sum(0)], -1)
    def wp_from(s):
        n, sy, sp, syy, spp, syp = np.moveaxis(np.asarray(s), -1, 0)
        r = (syp/n - sy*sp/n**2) / np.sqrt((syy/n - (sy/n)**2) * (spp/n - (sp/n)**2))
        return float(np.mean(r))
    SEQ_SOTA = np.stack([wsums(Y[i], SOTA[i], MASK[i]) for i in range(len(Y))])
    SEQ_REF = np.stack([wsums(Y[i], REF[i], MASK[i]) for i in range(len(Y))])

    return SEQ_REF, SEQ_SOTA, wp_from, wsums


@app.cell(hide_code=True)
def summary(CLS, SEQ_REF, SEQ_SOTA, Y, wp_from):
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

    return (SUMMARY,)


@app.cell(hide_code=True)
def findings(SUMMARY):
    mo.md(f"""
    # What limits the model, and what to do about it

    Clean out-of-fold predictions, 4 chain-excluded folds. **SOTA** = R4D analog (classes 0–2: fine-tuned since s4 + distilled
    GRU256; class 3: since s4 with merged parent+FT heads + distilled GRU256). **Reference** = incumbent recipe (two-stage FT
    since pair). Pooled Global WP: SOTA {SUMMARY[4]['WP SOTA (R4D analog)']:.4f} vs reference {SUMMARY[4]['WP reference (incumbent recipe)']:.4f}.

    ### 1. The loss is one place: long class-3 runs  (F3)
    Half of all lost metric sits in class-3 rows of runs lasting 64–1,024 rows (segment WP 0.25–0.44); by distance to the
    next quote move, rows 32–512 rows away hold ~44% of the loss at WP 0.09–0.28. Classes 0–2 together are under 40%.

    ### 2. In those runs the outcome is not knowable at the start  (F1, F2)
    For runs lasting ≥ 64 rows, corr(prediction, run target) at the run's first row is 0.02 in class 3 (0.07 class 2,
    ~0.2 classes 0/1) and rises only in the last ~32 rows before the next move. F1 shows it directly: the target sits
    at e.g. −1.3 for hundreds of rows while the prediction hovers near 0 until the book starts to move. This is the
    future-path part of the label, not a modelling bug — and why every post-hoc fix we tried failed.

    ### 3. Ensembling is saturated  (F6)
    The since member and the distilled GRU256 have residual correlation 0.985–0.999 in every segment. Averaging them
    gains ≤ 0.0005 and sometimes loses. Gains must come from better single models, not more members.

    ### 4. Why distillation works  (F5)
    Against the since member, the α 0.5 student is worse far from the next move (class 3, 128+ rows: −4 to −7 ×10⁻³)
    while the α 0.7 student is better everywhere (+0.5 to +4 ×10⁻³). Stronger teacher weight denoises labels that are
    dominated by unknowable future paths — exactly the high-loss region. Fine-tuning shows the opposite trade: the
    un-fine-tuned parent is better early in long runs for classes 1–2 (F2).

    ### 5. Dead ends confirmed here
    Prediction scale per segment looks misallocated (F7: far-from-move rows "want" 0.1–0.75× scale), but a causal
    24-parameter scale correction loses on every fold (E-322). Calibration is on the diagonal (06). Sequence-level
    dispersion and residual autocorrelation are future information (E-314, E-317).

    ### Next experiments this points to
    1. **Noise-adaptive distillation (H-059):** teacher weight rising with the training row's distance to label
       resolution (known at train time; inputs stay causal), against constant α 0.7. Targets the 44–50% loss region.
    2. **Push α (0.85 queued) and teacher quality** (GRU256 teachers in the team recipe).
    3. **Head-only FT or lighter FT for the since trunk** in classes 1–2, which keeps early-run information.

    Appendix (descriptive): target distributions, run structure, event-time ACF/PACF of target and residual, ARCH/GARCH
    (residual volatility persistence α+β ≈ 0.82–0.86, class 3 ≈ 0.99), calibration, per-sequence and between/within tables.
    """)

    return


@app.cell(hide_code=True)
def anatomy(CLS, EVENT, FOLD, MASK, REF, SEQ_SOTA, SOTA, Y, wp_from):
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

    return


@app.cell(hide_code=True)
def run_convergence(CLS, EVENT, FOLD, REF, SOTA, Y):
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

    return


@app.cell(hide_code=True)
def positions(EVENT, Y):
    # Shared per-row event coordinates: rows since the last quote move (causal) and rows until the next one (non-causal),
    # plus the length of the run each row belongs to. Used by the error budget and the mechanism plots.
    SINCE = np.zeros(Y.shape[:2], np.int32); UNTIL = np.zeros(Y.shape[:2], np.int32); RUNLEN = np.zeros(Y.shape[:2], np.int32)
    for _i in range(len(Y)):
        _e = np.flatnonzero(EVENT[_i]); _idx = np.arange(20000)
        _k = np.searchsorted(_e, _idx, side="right") - 1
        _prev = np.where(_k >= 0, _e[np.maximum(_k, 0)], 0)
        _nxt = np.where(_k + 1 < len(_e), _e[np.minimum(_k + 1, len(_e) - 1)], 20000)
        SINCE[_i] = _idx - _prev; UNTIL[_i] = _nxt - _idx; RUNLEN[_i] = _nxt - _prev
    POS_BINS = np.array([0, 2, 8, 32, 128, 512, 20001]); POS_LABELS = ["<2", "2–8", "8–32", "32–128", "128–512", "≥512"]

    return POS_BINS, POS_LABELS, RUNLEN, SINCE, UNTIL


@app.cell(hide_code=True)
def error_budget(
    BETA,
    CLS,
    MASK,
    POS_BINS,
    POS_LABELS,
    RUNLEN,
    SOTA,
    UNTIL,
    Y,
    wp_from,
    wsums,
):
    # F3 — Error budget: where is the lost metric? Loss = Σ|y|(y − βp)² with β the per-class OLS scale (metric geometry),
    # summed over scored rows of the SOTA OOF. Panels: share of total loss by class × run length, and by class ×
    # rows until the next quote move; annotated with the segment's WP so "big and bad" stands out.
    _rl_bins = np.array([0, 4, 16, 64, 256, 1024, 20001]); _rl_lab = ["<4", "4–16", "16–64", "64–256", "256–1024", "≥1024"]
    _loss = np.zeros((4, len(_rl_lab))); _loss2 = np.zeros((4, len(POS_LABELS)))
    _wp1 = np.full((4, len(_rl_lab)), np.nan); _wp2 = np.full((4, len(POS_LABELS)), np.nan)
    for _c in range(4):
        _sel = np.flatnonzero(CLS == _c)
        _m = MASK[_sel]; _y = np.clip(Y[_sel][_m], -2, 2).astype(np.float64); _p = np.clip(SOTA[_sel][_m], -2, 2).astype(np.float64)
        _l = (np.abs(_y) * (_y - BETA[_c] * _p) ** 2).sum(1)
        _b1 = np.digitize(RUNLEN[_sel][_m], _rl_bins) - 1; _b2 = np.digitize(UNTIL[_sel][_m], POS_BINS) - 1
        for _k in range(len(_rl_lab)):
            _loss[_c, _k] = _l[_b1 == _k].sum()
            if (_b1 == _k).sum() > 2000: _wp1[_c, _k] = wp_from(wsums(_y[_b1 == _k], _p[_b1 == _k], np.ones((_b1 == _k).sum(), bool)))
        for _k in range(len(POS_LABELS)):
            _loss2[_c, _k] = _l[_b2 == _k].sum()
            if (_b2 == _k).sum() > 2000: _wp2[_c, _k] = wp_from(wsums(_y[_b2 == _k], _p[_b2 == _k], np.ones((_b2 == _k).sum(), bool)))
    _tot = _loss.sum()
    _fig, _ax = plt.subplots(1, 2, figsize=(15, 4.2))
    for _j, (_L, _W, _lab, _title) in enumerate(((_loss, _wp1, _rl_lab, "run length (rows)"), (_loss2, _wp2, POS_LABELS, "rows until the next quote move"))):
        _im = _ax[_j].imshow(100 * _L / _tot, cmap="Reds", aspect="auto")
        for _c in range(4):
            for _k in range(len(_lab)):
                _ax[_j].text(_k, _c, f"{100*_L[_c,_k]/_tot:.1f}%\nWP {_W[_c,_k]:.2f}" if np.isfinite(_W[_c, _k]) else f"{100*_L[_c,_k]/_tot:.1f}%",
                             ha="center", va="center", fontsize=8)
        _ax[_j].set_xticks(range(len(_lab))); _ax[_j].set_xticklabels(_lab); _ax[_j].set_yticks(range(4)); _ax[_j].set_yticklabels([f"class {c}" for c in range(4)])
        _ax[_j].set_xlabel(_title); _ax[_j].grid(False)
        _ax[_j].set_title(f"share of total loss by class × {_title} (cell: loss share, segment WP)", fontsize=9)
    save(_fig, "F3_error_budget")

    return


@app.cell(hide_code=True)
def ideas_forest():
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

    return


@app.cell(hide_code=True)
def distill_mechanism(
    CLS,
    FOLD,
    MASK,
    POS_BINS,
    POS_LABELS,
    UNTIL,
    Y,
    wp_from,
    wsums,
):
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

    return CF0, D5F0, D7F0, MF0, S4F0, UF0, YF0


@app.cell(hide_code=True)
def diversity(
    CF0,
    D7F0,
    MF0,
    POS_BINS,
    POS_LABELS,
    S4F0,
    UF0,
    YF0,
    wp_from,
    wsums,
):
    # F6 — Ensemble diversity: correlation between the since member's and the distilled GRU256's residuals (lower = more
    # complementary) and the gain of their average over the better single member, by class × rows until next move.
    _fig, _ax = plt.subplots(1, 2, figsize=(13, 3.8))
    for _c in range(4):
        _sel = CF0 == _c; _rc, _gain = [], []
        for _k in range(len(POS_LABELS)):
            _m = MF0[_sel] & (np.digitize(UF0[_sel], POS_BINS) - 1 == _k)
            if _m.sum() < 2000: _rc.append(np.nan); _gain.append(np.nan); continue
            _y = np.clip(YF0[_sel][_m], -2, 2); _a = np.clip(S4F0[_sel][_m], -2, 2); _b = np.clip(D7F0[_sel][_m], -2, 2)
            _w = np.abs(_y).mean(1)
            _ra = ((_y - _a) * [1, -1]).sum(1) / 2; _rb = ((_y - _b) * [1, -1]).sum(1) / 2
            _rc.append(float(np.cov(_ra, _rb, aweights=_w)[0, 1] / np.sqrt(np.cov(_ra, aweights=_w) * np.cov(_rb, aweights=_w))))
            _one = np.ones(_m.sum(), bool)
            _gain.append(wp_from(wsums(_y, (_a + _b) / 2, _one)) - max(wp_from(wsums(_y, _a, _one)), wp_from(wsums(_y, _b, _one))))
        _ax[0].plot(range(len(POS_LABELS)), _rc, "o-", ms=3, color=CLASS_COLORS[_c], label=f"class {_c}")
        _ax[1].plot(range(len(POS_LABELS)), 1e3 * np.array(_gain), "o-", ms=3, color=CLASS_COLORS[_c])
    for _a in _ax: _a.set_xticks(range(len(POS_LABELS))); _a.set_xticklabels(POS_LABELS, fontsize=8); _a.set_xlabel("rows until next quote move")
    _ax[0].set_title("residual correlation: since s4 vs distilled GRU256 α0.7 (fold 0)", fontsize=9); _ax[0].legend(fontsize=8)
    _ax[1].set_title("WP gain ×1000 of their average over the better single member", fontsize=9); _ax[1].axhline(0, color="k", lw=0.6)
    save(_fig, "F6_ensemble_diversity")

    return


@app.cell(hide_code=True)
def magnitude(
    CF0,
    D5F0,
    D7F0,
    FOLD,
    MF0,
    POS_BINS,
    POS_LABELS,
    S4F0,
    SINCE,
    UF0,
    YF0,
):
    # F7 — Magnitude allocation. Pooled WP rewards predictions whose scale tracks how predictable each segment is.
    # For each segment, β_seg = cov_w(y,p)/var_w(p) is the scale that segment "wants"; β_seg/β_class = 1 everywhere means
    # the model spends its prediction magnitude where the signal is. Compare since s4, distilled α 0.5 and α 0.7 (fold 0),
    # by rows until the next move (explanatory, non-causal) and by rows since the last move (what a model can see).
    def _beta_seg(P, m, cls_sel):
        _y = np.clip(YF0[cls_sel][m], -2, 2).astype(np.float64); _p = np.clip(P[cls_sel][m], -2, 2).astype(np.float64)
        _b = []
        for _t in range(2):
            _w = np.abs(_y[:, _t])
            _mp = (_w * _p[:, _t]).sum() / _w.sum(); _my = (_w * _y[:, _t]).sum() / _w.sum()
            _b.append(((_w * (_p[:, _t] - _mp) * (_y[:, _t] - _my)).sum()) / ((_w * (_p[:, _t] - _mp) ** 2).sum()))
        return float(np.mean(_b))
    SINCEF0 = SINCE[np.flatnonzero(FOLD == 0)]
    _fig, _ax = plt.subplots(2, 4, figsize=(17, 7))
    MAGNITUDE = {}
    for _c in range(4):
        _sel = CF0 == _c
        for _row, (_arr, _lab) in enumerate(((UF0, "rows until next move"), (SINCEF0, "rows since last move"))):
            for _name, _P, _col in (("since s4", S4F0, "#1baf7a"), ("distilled α 0.5", D5F0, "#eb6834"), ("distilled α 0.7", D7F0, "#2a78d6")):
                _bc = _beta_seg(_P, MF0[_sel], _sel)
                _vals = []
                for _k in range(len(POS_LABELS)):
                    _m = MF0[_sel] & (np.digitize(_arr[_sel], POS_BINS) - 1 == _k)
                    _vals.append(_beta_seg(_P, _m, _sel) / _bc if _m.sum() > 2000 else np.nan)
                MAGNITUDE[(_c, _lab, _name)] = _vals
                _ax[_row, _c].plot(range(len(POS_LABELS)), _vals, "o-", ms=3, color=_col, label=_name)
            _ax[_row, _c].axhline(1, color="k", lw=0.6)
            _ax[_row, _c].set_xticks(range(len(POS_LABELS))); _ax[_row, _c].set_xticklabels(POS_LABELS, fontsize=7, rotation=30)
            _ax[_row, _c].set_title(f"class {_c}: wanted scale / class scale, by {_lab}", fontsize=9)
    _ax[0, 0].legend(fontsize=7); _ax[0, 0].set_ylabel("β_segment / β_class  (1 = well allocated)")
    _ax[1, 0].set_ylabel("β_segment / β_class")
    save(_fig, "F7_magnitude_allocation")

    return


@app.cell(hide_code=True)
def since_scale_test(
    CLS,
    FOLD,
    MASK,
    POS_BINS,
    POS_LABELS,
    SINCE,
    SOTA,
    Y,
    wsums,
):
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

    return


@app.cell(hide_code=True)
def target_dist(CLS, MASK, Y):
    _fig, _ax = plt.subplots(1, 3, figsize=(13, 3.4))
    for _c in range(4):
        _sel = CLS == _c
        _y = np.clip(Y[_sel][MASK[_sel]], -2, 2)
        _ax[0].hist(_y[:, 0], bins=81, range=(-2, 2), histtype="step", density=True, color=CLASS_COLORS[_c], label=f"class {_c}")
        _ax[1].hist(_y[:, 1], bins=81, range=(-2, 2), histtype="step", density=True, color=CLASS_COLORS[_c])
    _ax[0].set_title("t0 on scored rows (clipped)"); _ax[1].set_title("t1 on scored rows (clipped)"); _ax[0].legend(fontsize=8)
    _y3 = np.clip(Y[CLS == 3][MASK[CLS == 3]], -2, 2)
    _ax[2].hexbin(_y3[:, 0], _y3[:, 1], gridsize=40, bins="log", cmap="Blues"); _ax[2].set_title("class 3: t0 vs t1 (log count)")
    _ax[2].set_xlabel("t0"); _ax[2].set_ylabel("t1")
    save(_fig, "01_target_distributions")

    return


@app.cell(hide_code=True)
def run_structure(CLS, EVENT, Y):
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

    return


@app.cell(hide_code=True)
def event_series(CLS, EVENT, SEQ_SOTA, SOTA, Y):
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

    return BETA, RUNS


@app.cell(hide_code=True)
def target_acf(RUNS):
    from statsmodels.tsa.stattools import acf as sm_acf, pacf as sm_pacf
    def mean_acf(key, cls, nlags=20, fn="acf", max_seqs=250):
        """Length-weighted average of per-sequence ACF/PACF of an event-time series (sequence boundaries respected)."""
        _acc, _w = np.zeros(nlags + 1), 0.0
        for _r in [r for r in RUNS if r["cls"] == cls][:max_seqs]:
            _x = _r[key]
            if len(_x) < 4 * nlags or np.std(_x) < 1e-9: continue
            _v = sm_acf(_x, nlags=nlags, fft=True) if fn == "acf" else sm_pacf(_x, nlags=nlags, method="ywm")
            _acc += len(_x) * _v; _w += len(_x)
        return _acc / max(_w, 1)
    LAGS = np.arange(21)
    _fig, _ax = plt.subplots(2, 2, figsize=(13, 6.4), sharex=True)
    TARGET_ACF = {}
    for _c in range(4):
        for _j, _key in enumerate(["D", "S"]):
            _a = mean_acf(_key, _c); _p = mean_acf(_key, _c, fn="pacf")
            TARGET_ACF[(_c, _key)] = (_a, _p)
            _ax[0, _j].plot(LAGS[1:], _a[1:], "o-", ms=3, color=CLASS_COLORS[_c], label=f"class {_c}")
            _ax[1, _j].plot(LAGS[1:], _p[1:], "o-", ms=3, color=CLASS_COLORS[_c])
    for _j, _key in enumerate(["D (direction)", "S (t0+t1, spread-like)"]):
        _ax[0, _j].set_title(f"target ACF in event time — {_key}"); _ax[1, _j].set_title(f"target PACF in event time — {_key}")
        _ax[1, _j].set_xlabel("lag (quote-move runs)")
        for _r in range(2): _ax[_r, _j].axhline(0, color="k", lw=0.6)
    _ax[0, 0].legend(fontsize=8)
    save(_fig, "03_target_acf_pacf_event_time")

    return LAGS, mean_acf, sm_acf


@app.cell(hide_code=True)
def residual_acf(BETA, CLS, LAGS, SOTA, Y, mean_acf):
    _fig, _ax = plt.subplots(2, 3, figsize=(15, 6.4))
    RESID_ACF = {}
    for _c in range(4):
        for _j, _key in enumerate(["rD_start", "rD_end"]):
            _a = mean_acf(_key, _c); _p = mean_acf(_key, _c, fn="pacf")
            RESID_ACF[(_c, _key)] = (_a, _p)
            _ax[0, _j].plot(LAGS[1:], _a[1:], "o-", ms=3, color=CLASS_COLORS[_c], label=f"class {_c}")
            _ax[1, _j].plot(LAGS[1:], _p[1:], "o-", ms=3, color=CLASS_COLORS[_c])
    _ax[0, 0].set_title("residual ACF, p at run START (event time)"); _ax[1, 0].set_title("residual PACF, p at run START")
    _ax[0, 1].set_title("residual ACF, p at run END (event time)"); _ax[1, 1].set_title("residual PACF, p at run END")
    # row-time ACF of the scored-row residual (log-spaced lags), class means over sequences
    _lags = np.unique(np.logspace(0, 3.3, 28).astype(int))
    for _c in range(4):
        _acc = np.zeros(len(_lags)); _n = 0
        for _i in np.flatnonzero(CLS == _c)[:200]:
            _p = SOTA[_i, 99:].astype(np.float64); _y = np.clip(Y[_i, 99:].astype(np.float64), -2, 2)
            _r = ((_y[:, 0] - BETA[_c, 0]*_p[:, 0]) - (_y[:, 1] - BETA[_c, 1]*_p[:, 1])) / 2
            _r = _r - _r.mean(); _v = (_r * _r).mean()
            if _v < 1e-12: continue
            _acc += np.array([(_r[:-L] * _r[L:]).mean() / _v for L in _lags]); _n += 1
        _ax[0, 2].semilogx(_lags, _acc / max(_n, 1), "-", color=CLASS_COLORS[_c], label=f"class {_c}")
        _ys = np.clip(Y[np.flatnonzero(CLS == _c)[:200], 99:], -2, 2).astype(np.float64)
        _d = (_ys[..., 0] - _ys[..., 1]) / 2; _d = _d - _d.mean(1, keepdims=True); _vv = (_d*_d).mean(1, keepdims=True) + 1e-12
        _ax[1, 2].semilogx(_lags, [((_d[:, :-L]*_d[:, L:]).mean(1, keepdims=True) / _vv).mean() for L in _lags], "-", color=CLASS_COLORS[_c])
    _ax[0, 2].set_title("row-time ACF of residual D (all rows ≥ 99)"); _ax[1, 2].set_title("row-time ACF of target D")
    for _a in _ax.ravel(): _a.axhline(0, color="k", lw=0.6)
    _ax[0, 0].legend(fontsize=8); _ax[1, 2].set_xlabel("lag (rows)"); _ax[1, 0].set_xlabel("lag (runs)"); _ax[1, 1].set_xlabel("lag (runs)")
    save(_fig, "04_residual_acf_pacf")

    return


@app.cell(hide_code=True)
def arch_garch(LAGS, RUNS, sm_acf):
    # Volatility clustering in event time: ACF of squared (demeaned) series = ARCH effect; Engle ARCH-LM per sequence;
    # GARCH(1,1) fits on run-level D and on the residual for a sample of sequences per class.
    from statsmodels.stats.diagnostic import het_arch as _het_arch
    from arch import arch_model as _arch_model
    import warnings as _warnings
    _fig, _ax = plt.subplots(1, 3, figsize=(15, 3.6))
    GARCH = {}
    for _c in range(4):
        _sq = []
        for _key, _ls in (("D", "-"), ("rD_start", "--")):
            _acc, _w = np.zeros(21), 0
            for _r in [r for r in RUNS if r["cls"] == _c][:250]:
                _x = _r[_key] - _r[_key].mean(); _x2 = _x * _x - (_x * _x).mean()
                if _x2.std() < 1e-9 or len(_x2) < 100: continue
                _acc += len(_x2) * sm_acf(_x2, nlags=20, fft=True); _w += len(_x2)
            _ax[0].plot(LAGS[1:], (_acc / max(_w, 1))[1:], _ls, color=CLASS_COLORS[_c], label=f"class {_c} {'target D' if _key=='D' else 'residual'}")
        _pv, _ab = [], []
        for _r in [r for r in RUNS if r["cls"] == _c][:60]:
            _x = _r["rD_start"]
            if len(_x) < 300 or _x.std() < 1e-6: continue
            _pv.append(_het_arch(_x - _x.mean(), nlags=10)[1])
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore")
                try:
                    _res = _arch_model(100 * (_x - _x.mean()), mean="Zero", vol="GARCH", p=1, q=1).fit(disp="off")
                    _ab.append((_res.params["alpha[1]"], _res.params["beta[1]"]))
                except Exception:
                    pass
        _ab = np.array(_ab)
        GARCH[_c] = {"arch_lm_share_p<0.01": float(np.mean(np.array(_pv) < 0.01)), "median_alpha": float(np.median(_ab[:, 0])),
                     "median_beta": float(np.median(_ab[:, 1])), "median_persistence": float(np.median(_ab.sum(1))), "n": len(_ab)}
        _ax[1].hist(_ab.sum(1), bins=20, range=(0, 1.05), histtype="step", color=CLASS_COLORS[_c], label=f"class {_c}")
        _ax[2].bar(_c, GARCH[_c]["arch_lm_share_p<0.01"], color=CLASS_COLORS[_c])
    _ax[0].set_title("ACF of squared series (ARCH effect), event time"); _ax[0].legend(fontsize=7); _ax[0].set_xlabel("lag (runs)")
    _ax[1].set_title("GARCH(1,1) persistence α+β of residual (per sequence)"); _ax[1].legend(fontsize=8)
    _ax[2].set_title("share of sequences with ARCH-LM p<0.01 (residual)"); _ax[2].set_xticks(range(4))
    save(_fig, "05_arch_garch")
    mo.ui.table([{"class": c, **v} for c, v in GARCH.items()], selection=None)

    return


@app.cell(hide_code=True)
def calibration(CLS, MASK, SOTA, Y):
    # Reliability: |y|-weighted mean target vs prediction bins, per class and target (scored rows, OOF).
    _fig, _ax = plt.subplots(1, 2, figsize=(12, 3.8))
    for _t in range(2):
        for _c in range(4):
            _sel = CLS == _c
            _p = np.clip(SOTA[_sel][MASK[_sel]][:, _t], -2, 2); _y = np.clip(Y[_sel][MASK[_sel]][:, _t], -2, 2); _w = np.abs(_y)
            _q = np.quantile(_p, np.linspace(0, 1, 26)); _b = np.clip(np.searchsorted(_q, _p) - 1, 0, 24)
            _pm = np.bincount(_b, _w * _p, 25) / np.maximum(np.bincount(_b, _w, 25), 1e-9)
            _ym = np.bincount(_b, _w * _y, 25) / np.maximum(np.bincount(_b, _w, 25), 1e-9)
            _ax[_t].plot(_pm, _ym, "o-", ms=3, color=CLASS_COLORS[_c], label=f"class {_c}")
        _ax[_t].plot([-2, 2], [-2, 2], "k:", lw=0.8); _ax[_t].set_title(f"t{_t}: |y|-weighted mean target per prediction bin")
        _ax[_t].set_xlabel("prediction (bin mean)"); _ax[_t].set_ylabel("target (bin mean)")
    _ax[0].legend(fontsize=8)
    save(_fig, "06_calibration")

    return


@app.cell(hide_code=True)
def event_position(CLS, EVENT, MASK, SOTA, Y, wp_from, wsums):
    # Where inside a run is the metric lost? WP by rows since the last quote move (causal) and rows until the next one (not causal).
    _bins = np.array([0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 20001])
    _since = np.zeros(Y.shape[:2], np.int32); _until = np.zeros(Y.shape[:2], np.int32)
    for _i in range(len(Y)):
        _e = np.flatnonzero(EVENT[_i]); _idx = np.arange(20000)
        _k = np.searchsorted(_e, _idx, side="right") - 1
        _since[_i] = np.where(_k >= 0, _idx - _e[np.maximum(_k, 0)], 9999)
        _k2 = np.searchsorted(_e, _idx, side="right")
        _until[_i] = np.where(_k2 < len(_e), _e[np.minimum(_k2, len(_e) - 1)] - _idx, 9999)
    _fig, _ax = plt.subplots(1, 3, figsize=(15, 3.6))
    for _j, (_arr, _lab) in enumerate(((_since, "rows since last quote move (causal)"), (_until, "rows until next quote move (non-causal)"))):
        for _c in range(4):
            _sel = CLS == _c
            _bb = np.digitize(_arr[_sel], _bins) - 1
            _vals, _wts = [], []
            for _k in range(len(_bins) - 1):
                _m = MASK[_sel] & (_bb == _k)
                if _m.sum() < 500: _vals.append(np.nan); _wts.append(0); continue
                _vals.append(wp_from(wsums(Y[_sel], SOTA[_sel], _m))); _wts.append(np.abs(np.clip(Y[_sel][_m], -2, 2)).sum())
            _ax[_j].plot(range(len(_vals)), _vals, "o-", ms=3, color=CLASS_COLORS[_c], label=f"class {_c}")
            if _j == 1:
                _ax[2].plot(range(len(_wts)), np.array(_wts) / max(sum(_wts), 1), "o-", ms=3, color=CLASS_COLORS[_c])
        _ax[_j].set_xticks(range(len(_bins) - 1)); _ax[_j].set_xticklabels([f"{a}" for a in _bins[:-1]], rotation=45, fontsize=7)
        _ax[_j].set_title(f"WP by {_lab}"); _ax[_j].set_xlabel("bin lower edge (rows)")
    _ax[2].set_xticks(range(len(_bins) - 1)); _ax[2].set_xticklabels([f"{a}" for a in _bins[:-1]], rotation=45, fontsize=7)
    _ax[2].set_title("metric weight share by rows until next move"); _ax[0].legend(fontsize=8)
    save(_fig, "07_wp_by_event_position")

    return


@app.cell(hide_code=True)
def per_sequence(CLS, FOLD, SEQ_REF, SEQ_SOTA, wp_from):
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

    return


@app.cell(hide_code=True)
def between_within(CLS, SEQ_SOTA):
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

    return


if __name__ == "__main__":
    app.run()
