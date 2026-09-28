import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    mo.md("""# Connectome: causal headroom audit

    **Question.** Is the high-weight class-3 winning tier forecastable from information available at prediction time? Outcome anatomy uses validation labels. Model-based diagnostics use fold-specific out-of-fold pair predictions on pure chains; realized-outcome slices are descriptive only.""")
    return (mo,)


@app.cell
def _(mo):
    import numpy as np
    import polars as pl
    import matplotlib.pyplot as plt
    from pathlib import Path
    from wnn.splits import valid_row_groups
    from wnn.data import SequenceFile
    ROOT = mo.notebook_dir().parent
    V = np.load(ROOT / "cache" / "valid_targets.npz")
    Y, MASK, CLS, RG = (V[k] for k in ("y", "mask", "market_class", "row_group"))
    _chain_folds = np.load(ROOT / "cache" / "chain_folds.npz")
    _valid_fold = {int(g): int(f) for g, f, s in zip(_chain_folds['row_group'], _chain_folds['fold'], _chain_folds['split']) if s == 'valid'}
    FOLD = np.array([_valid_fold[int(g)] for g in RG], dtype=np.int8)
    class FoldPred:
        def __init__(self, prefix, fold):
            self.fold = fold
            self.arrays = {k: np.load(ROOT / 'cache' / f'pred_{prefix}_f{k}_valid.npy', mmap_mode='r') for k in (0, 1)}
        def __getitem__(self, index):
            seq_index = int(index[0])
            return self.arrays[int(self.fold[seq_index])][index]
    P0 = FoldPred('ab62_base_s4', FOLD)
    P1 = FoldPred('ab62_since', FOLD)
    DEV, HOLD = valid_row_groups(ROOT / "datasets" / "valid.parquet")
    Q = 0.65687
    C3 = np.flatnonzero(CLS == 3)
    print(f"valid: {len(RG)} sequences; class 3: {len(C3)}; scored class-3 rows: {int(MASK[C3,99:].sum()):,}; OOF folds: 0,1")
    return (
        C3,
        CLS,
        DEV,
        FOLD,
        MASK,
        P0,
        P1,
        Q,
        RG,
        ROOT,
        SequenceFile,
        Y,
        np,
        pl,
        plt,
    )


@app.cell
def _(CLS, DEV, RG, ROOT, mo, np, pl):
    import pyarrow.parquet as pq
    _pf = pq.ParquetFile(ROOT / "datasets" / "valid.parquet")
    _cols = [f"i0_p{j}" for j in range(22)] + ["t0", "t1", "is_scored"]
    _rows = []
    for _k in range(4):
        _choices = np.flatnonzero((CLS == _k) & np.isin(RG, DEV))[:24]
        _same_book = _label_changes = _same_and_change = _weight_changed = _weight_total = 0.0
        _seq_rates = []
        for _idx in _choices:
            _tbl = _pf.read_row_group(int(RG[_idx]), columns=_cols, use_threads=False)
            _p = np.stack([_tbl.column(f"i0_p{j}").to_numpy() for j in range(22)], 1)
            _t = np.stack([_tbl.column(_c).to_numpy() for _c in ("t0", "t1")], 1)
            _m = _tbl.column("is_scored").to_numpy().astype(bool)
            _book_changed = np.any(_p[1:] != _p[:-1], axis=1)
            _target_changed = np.any(np.abs(_t[1:] - _t[:-1]) > 1e-4, axis=1)
            _good = np.arange(1, len(_t)) >= 99
            _same_book += np.sum(_good & ~_book_changed)
            _label_changes += np.sum(_good & _target_changed)
            _same_and_change += np.sum(_good & ~_book_changed & _target_changed)
            _w = np.abs(np.clip(_t[1:], -2, 2)).sum(1)
            _weight_changed += _w[_good & _m[1:] & ~_book_changed & _target_changed].sum()
            _weight_total += _w[_good & _m[1:]].sum()
            _seq_rates.append(np.sum(_good & ~_book_changed & _target_changed) / max(np.sum(_good & _target_changed), 1))
        _rows.append({"class": _k, "sequences": len(_choices), "target changes": int(_label_changes),
                      "no-book share of target changes": _same_and_change / max(_label_changes, 1),
                      "no-book target change per quiet row": _same_and_change / max(_same_book, 1),
                      "scored weight at no-book target change": _weight_changed / max(_weight_total, 1),
                      "median sequence rate": float(np.median(_seq_rates))})
    table_change = pl.DataFrame(_rows)
    mo.vstack([mo.md("## Elephant 1: can targets change without a visible quote event?"), table_change])
    return (table_change,)


@app.cell
def _(C3, DEV, MASK, P0, P1, Q, RG, ROOT, SequenceFile, Y, np, pl):
    import pickle
    from wnn.ticks import reconstruct
    with (ROOT / "cache" / "class3_moves_full.pkl").open("rb") as _f:
        MOVES, SPREADS = pickle.load(_f)
    _cf = np.load(ROOT / "cache" / "chain_folds.npz")
    _valid = _cf["split"] == "valid"
    _chainmap = {int(g): (int(c), int(f)) for g, c, f in zip(_cf["row_group"][_valid], _cf["chain"][_valid], _cf["fold"][_valid])}
    _train_chains = set(map(int, _cf["chain"][_cf["split"] == "train"]))
    _lists = []
    _future = []
    _input = []
    _hist = []
    _rng = np.random.default_rng(41)
    _source = SequenceFile(ROOT / "datasets" / "valid.parquet")
    for _fold in (0, 1):
        _choices = [int(i) for i in C3 if int(RG[i]) in set(DEV) and _chainmap[int(RG[i])][1] == _fold and _chainmap[int(RG[i])][0] not in _train_chains]
        _choices = _rng.choice(_choices, size=min(64, len(_choices)), replace=False)
        for _i in _choices:
            _seq = _source.read(int(RG[_i]), targets=False, mask=False)
            _ask, _bid, _unk = reconstruct(_seq.features, MOVES, SPREADS)
            _events = np.r_[0, np.flatnonzero(np.any(_seq.features[1:,:22] != _seq.features[:-1,:22], axis=1)) + 1]
            _n = len(_events) - 64
            if _n <= 0:
                continue
            _start = _events[:_n]
            _end = _events[1:_n + 1]
            _ticks = np.rint(Y[_i, _start] / Q).astype(np.int16)
            _spread = _ask[_start] - _bid[_start]
            _stop = -(_spread + 2)
            _F0 = _ask[_start] + _ticks[:,0]
            _F1 = _bid[_start] - _ticks[:,1]
            _ftrue = np.where(_ticks[:,0] != _stop, _F0, _F1)
            _ambig = (_ticks[:,0] == _stop) & (_ticks[:,1] == _stop)
            _both_ok = (_ticks[:,0] != _stop) & (_ticks[:,1] != _stop)
            _mid = 0.5 * (_ask[_events] + _bid[_events])
            _path = np.stack([_mid[j + 1 : j + 1 + _n] for j in range(64)], axis=1)
            _weight = np.abs(np.clip(Y[_i], -2, 2)).sum(axis=1) * MASK[_i]
            _cum = np.r_[0.0, np.cumsum(_weight)]
            _mass = _cum[_end] - _cum[_start]
            _scored = np.flatnonzero(MASK[_i])
            if not len(_scored):
                continue
            _loc = np.searchsorted(_scored, _start)
            _rep = _scored[np.minimum(_loc, len(_scored) - 1)]
            _keep = (_mass > 0) & ~_ambig & (_loc < len(_scored)) & (_rep < _end)
            _lists.append(pl.DataFrame({"fold": np.full(_keep.sum(), _fold, np.int8),
                "seq": np.full(_keep.sum(), int(RG[_i]), np.int32), "start": _start[_keep],
                "F": _ftrue[_keep], "ask": _ask[_start[_keep]], "bid": _bid[_start[_keep]],
                "weight": _mass[_keep], "winner_tick": np.maximum(_ticks[_keep,0], _ticks[_keep,1]),
                "first_jump": _path[_keep,0] - _mid[:_n][_keep],
                "both_consistent": (_F0 == _F1)[_keep & _both_ok] if False else np.where(_both_ok[_keep], _F0[_keep] == _F1[_keep], True)}))
            _future.append(_path[_keep].astype(np.float32))
            _pred = 0.5 * (P0[_i, _rep[_keep]] + P1[_i, _rep[_keep]])
            _input.append(np.column_stack((_pred, _seq.features[_rep[_keep]], (_rep - _start)[_keep])).astype(np.float32))
            _event_index = np.arange(_n)
            _abs_path = np.r_[0.0, np.cumsum(np.abs(np.diff(_mid)))]
            _history_columns = []
            for _h in (1, 2, 4, 8, 16, 32, 64):
                _prev = np.maximum(0, _event_index - _h)
                _history_columns.extend((_mid[_event_index] - _mid[_prev],
                    _abs_path[_event_index] - _abs_path[_prev],
                    np.log1p(_start - _events[_prev])))
            _hist.append(np.stack(_history_columns, axis=1)[_keep].astype(np.float32))
    EDATA = pl.concat(_lists)
    FUT = np.concatenate(_future)
    EINPUT = np.concatenate(_input)
    EHIST = np.concatenate(_hist)
    print(f"episodes {len(EDATA):,}; sequences {EDATA['seq'].n_unique()}; folds {EDATA.group_by('fold').len().sort('fold')}; both-valid F agreement {EDATA['both_consistent'].mean():.5f}")
    return EDATA, EHIST, EINPUT, FUT


@app.cell
def _(EDATA, FUT, mo, np, pl):
    _F = EDATA["F"].to_numpy()
    _W = EDATA["weight"].to_numpy()
    _fold = EDATA["fold"].to_numpy()
    _compare = []
    for _n in range(1, 13):
        _candidates = {f"mean next {_n}": np.rint(FUT[:,:_n].mean(1)),
                       f"last event {_n}": np.rint(FUT[:,_n-1]),
                       f"median next {_n}": np.rint(np.median(FUT[:,:_n], axis=1))}
        for _name, _hat in _candidates.items():
            _match = _hat == _F
            _row = {"candidate": _name, "exact events": _match.mean(),
                    "exact metric weight": np.sum(_W * _match) / _W.sum(),
                    "weighted |tick error|": np.sum(_W * np.abs(_hat - _F)) / _W.sum()}
            for _k in (0, 1):
                _s = _fold == _k
                _row[f"fold {_k} exact weight"] = np.sum(_W[_s] * _match[_s]) / _W[_s].sum()
            _compare.append(_row)
    path_table = pl.DataFrame(_compare).sort("exact metric weight", descending=True)
    mo.vstack([mo.md("""## Elephant 2: which future-path statistic defines the label?
    These are future oracle reconstructions for mechanism discovery, never usable as features."""), path_table.head(20)])
    return


@app.cell
def _(EDATA, EINPUT, mo, np, pl):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score, brier_score_loss
    from threadpoolctl import threadpool_limits
    _evfold = EDATA["fold"].to_numpy()
    _evw = EDATA["weight"].to_numpy().astype(np.float64)
    _tier = EDATA["winner_tick"].to_numpy()
    _p = EINPUT[:,:2]
    _base = np.column_stack((_p, _p[:,0]-_p[:,1], np.abs(_p[:,0]-_p[:,1]), _p.sum(1), EINPUT[:,-1]))
    _full = np.column_stack((_base, EINPUT[:,2:-1]))
    _head_rows = []
    tier_preds = {}
    with threadpool_limits(limits=12):
        for _threshold in (1, 2):
            _label = (_tier >= _threshold).astype(np.int8)
            for _eval_fold in (0, 1):
                _train = _evfold != _eval_fold
                _eval = ~_train
                for _variant, _x in (("pair + age", _base), ("pair + raw state", _full)):
                    _model = HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=15,
                        min_samples_leaf=80, learning_rate=0.05, l2_regularization=20,
                        early_stopping=False, random_state=41)
                    _model.fit(_x[_train], _label[_train], sample_weight=np.minimum(_evw[_train],100))
                    _prob = _model.predict_proba(_x[_eval])[:,1]
                    tier_preds[(_threshold, _eval_fold, _variant)] = _prob
                    _head_rows.append({"tier threshold": _threshold, "eval fold": _eval_fold,
                        "model": _variant, "weighted AUC": roc_auc_score(_label[_eval], _prob, sample_weight=_evw[_eval]),
                        "weighted Brier": brier_score_loss(_label[_eval], _prob, sample_weight=_evw[_eval]),
                        "unweighted AUC": roc_auc_score(_label[_eval], _prob)})
    head_table = pl.DataFrame(_head_rows)
    mo.vstack([mo.md("""## Causal tier probe on separate price-chain folds
    One scored snapshot per episode; both models train on the other fold with the same capped episode weights. The added features are the 112 current market inputs. Weighted AUC is descriptive and does not equal the official submission score."""), head_table])
    return (
        HistGradientBoostingClassifier,
        brier_score_loss,
        head_table,
        roc_auc_score,
        threadpool_limits,
    )


@app.cell
def _(
    EDATA,
    EHIST,
    EINPUT,
    HistGradientBoostingClassifier,
    brier_score_loss,
    head_table,
    mo,
    np,
    pl,
    roc_auc_score,
    threadpool_limits,
):
    _fold2 = EDATA["fold"].to_numpy()
    _w2 = EDATA["weight"].to_numpy().astype(np.float64)
    _tier2 = EDATA["winner_tick"].to_numpy()
    _p2 = EINPUT[:,:2]
    _base2 = np.column_stack((_p2, _p2[:,0]-_p2[:,1], np.abs(_p2[:,0]-_p2[:,1]), _p2.sum(1), EINPUT[:,-1]))
    _hfull = np.column_stack((_base2, EHIST))
    _history_rows = []
    history_preds = {}
    with threadpool_limits(limits=12):
        for _threshold in (1, 2):
            _label = (_tier2 >= _threshold).astype(np.int8)
            for _eval_fold in (0, 1):
                _train = _fold2 != _eval_fold
                _eval = ~_train
                _model = HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=15,
                    min_samples_leaf=80, learning_rate=0.05, l2_regularization=20,
                    early_stopping=False, random_state=41)
                _model.fit(_hfull[_train], _label[_train], sample_weight=np.minimum(_w2[_train],100))
                _prob = _model.predict_proba(_hfull[_eval])[:,1]
                history_preds[(_threshold, _eval_fold)] = _prob
                _reference = head_table.filter((pl.col("tier threshold") == _threshold) &
                    (pl.col("eval fold") == _eval_fold) & (pl.col("model") == "pair + age"))
                _auc = roc_auc_score(_label[_eval], _prob, sample_weight=_w2[_eval])
                _brier = brier_score_loss(_label[_eval], _prob, sample_weight=_w2[_eval])
                _history_rows.append({"tier threshold": _threshold, "eval fold": _eval_fold,
                    "pair + causal path AUC": _auc, "AUC gain over pair + age": _auc - _reference["weighted AUC"][0],
                    "pair + causal path Brier": _brier,
                    "Brier improvement": _reference["weighted Brier"][0] - _brier})
    history_table = pl.DataFrame(_history_rows)
    mo.vstack([mo.md("""### Does the previous price-event path forecast the payoff tier?
    Current and previous decoded quote events only. Same train fold, weight cap, model size and evaluation as above. Positive AUC with negative Brier improvement means ranking rose but probability calibration fell."""), history_table])
    return (history_table,)


@app.cell
def _(EDATA, FUT, ROOT, mo, np, pl, plt):
    _yref = EDATA["F"].to_numpy()
    _mass = EDATA["weight"].to_numpy()
    _group = EDATA["fold"].to_numpy()
    _truth_tier = EDATA["winner_tick"].to_numpy()
    _cumulative = np.cumsum(FUT.astype(np.float64), axis=1)
    _ns = np.arange(1, FUT.shape[1] + 1)
    _plot_rows = []
    for _k in (0,1):
        _s = _group == _k
        for _n in _ns:
            _h = np.rint(_cumulative[_s,_n-1] / _n)
            _plot_rows.append((_k,_n, np.mean(_h == _yref[_s]),
                np.average(_h == _yref[_s], weights=_mass[_s]),
                np.average(np.abs(_h-_yref[_s]), weights=_mass[_s])))
    horizon_table = pl.DataFrame(_plot_rows, schema=["fold","horizon","episode exact","metric weight exact","weighted tick MAE"], orient="row")
    _fig, _axes = plt.subplots(1,3,figsize=(13.5,3.5))
    for _k, _color in ((0,"#2679c9"),(1,"#e16637")):
        _z = horizon_table.filter(pl.col("fold") == _k)
        _axes[0].plot(_z["horizon"],_z["episode exact"],label=f"fold {_k} episodes",color=_color,linestyle="--")
        _axes[0].plot(_z["horizon"],_z["metric weight exact"],label=f"fold {_k} weight",color=_color)
        _axes[1].plot(_z["horizon"],_z["weighted tick MAE"],label=f"fold {_k}",color=_color)
    for _ax in _axes[:2]:
        for _n in (4,8,24):
            _ax.axvline(_n,color="#888",linewidth=0.6)
        _ax.set_xlabel("number of future quote events (oracle)")
        _ax.grid(alpha=0.25)
    _axes[0].set_title("Reference-price exact match");_axes[0].set_ylabel("fraction")
    _axes[0].legend(fontsize=7)
    _axes[1].set_title("Payoff-weighted reference error");_axes[1].set_ylabel("ticks")
    _axes[1].legend(fontsize=7)
    _bins = ["touch / inside", "one tick", "two ticks", "three+ ticks"]
    _tbin = np.clip(_truth_tier,0,3)
    _evs = np.bincount(_tbin,minlength=4)/len(_tbin)
    _masses = np.bincount(_tbin,weights=_mass,minlength=4)/_mass.sum()
    _x = np.arange(4)
    _axes[2].bar(_x-0.18,_evs,0.36,label="book episodes",color="#2679c9")
    _axes[2].bar(_x+0.18,_masses,0.36,label="metric weight",color="#e16637")
    _axes[2].set_xticks(_x,_bins,rotation=18,ha="right")
    _axes[2].set_title("Rare continuations carry the score")
    _axes[2].set_ylabel("fraction");_axes[2].legend(fontsize=7)
    _fig.tight_layout()
    _chart_path = ROOT / "research" / "artifacts" / "brainteaser_event_horizon.png"
    _fig.savefig(_chart_path,dpi=150)
    mo.vstack([mo.md("""## Event-clock anatomy
    The 8-event average maximizes exact reference matches here, while longer averages reduce the weighted tick error on extreme moves. All future-path lines are **oracle diagnostics**. The right panel shows why row-weighted and independent-event analyses disagree."""),_fig])
    return (horizon_table,)


@app.cell
def _(EDATA, MASK, P0, P1, Q, RG, ROOT, Y, np, pl):
    _rtefile = np.load(ROOT / "cache" / "rte_valid.npz")
    _ageall, _futureall = _rtefile["since_last"], _rtefile["to_next"]
    _seqfold = {int(r["seq"]):int(r["fold"]) for r in EDATA.select("seq","fold").unique().to_dicts()}
    _posmap = {int(g):i for i,g in enumerate(RG)}
    _rowparts = []
    for _g, _fold in _seqfold.items():
        _i = _posmap[_g]
        _selected = MASK[_i].copy()
        _selected[:99] = False
        _target = Y[_i,_selected]
        _pred = 0.5*(P0[_i,_selected]+P1[_i,_selected])
        _rowparts.append(pl.DataFrame({"fold":np.full(len(_target),_fold,np.int8),
            "seq":np.full(len(_target),_g,np.int32),
            "age":_ageall[_i,_selected],"future distance":_futureall[_i,_selected],
            "tier":np.maximum(0,np.rint(np.max(_target,axis=1)/Q)).astype(np.int16),
            "weight":np.abs(np.clip(_target,-2,2)).sum(1),
                "weighted absolute error":(np.abs(np.clip(_target,-2,2))*np.abs(np.clip(_pred,-2,2)-np.clip(_target,-2,2))).sum(1),
            "pred direction magnitude":np.abs(_pred[:,0]-_pred[:,1]),
            "side correct":np.sign(_pred[:,0]-_pred[:,1])==np.sign(_target[:,0]-_target[:,1])}))
    ROWDATA=pl.concat(_rowparts)
    print(f"{len(ROWDATA):,} scored class-3 rows from {len(_seqfold)} pure-chain development sequences")
    return (ROWDATA,)


@app.cell(hide_code=True)
def _(mo):
    tier_threshold = mo.ui.dropdown({"at least one tick":1,"at least two ticks":2,"at least three ticks":3},value="at least two ticks",label="winning tier")
    tier_threshold
    return (tier_threshold,)


@app.cell(hide_code=True)
def _(ROOT, ROWDATA, mo, np, pl, plt, tier_threshold):
    _thr = tier_threshold.value
    _edges = np.array([0,1,2,4,8,16,32,64,128,100000])
    _labels = ["0","1","2–3","4–7","8–15","16–31","32–63","64–127","128+"]
    _fig2, _axs = plt.subplots(1,3,figsize=(13,3.5))
    _stats = []
    for _fold in (0,1):
        _data = ROWDATA.filter(pl.col("fold")==_fold)
        _a = _data["age"].to_numpy()
        _r = _data["future distance"].to_numpy()
        _y = (_data["tier"].to_numpy()>=_thr)
        _w = _data["weight"].to_numpy()
        _pd = _data["pred direction magnitude"].to_numpy()
        for _axis,_v,_causal in ((0,_a,True),(1,_r,False)):
            _bucket = np.minimum(np.searchsorted(_edges[1:],_v,side="right"),8)
            _rates=[]
            for _j in range(9):
                _s = _bucket==_j
                _rates.append(float(np.average(_y[_s],weights=_w[_s])) if _s.any() else np.nan)
                _stats.append({"fold":_fold,"axis":"causal age" if _causal else "future distance oracle",
                    "bin":_labels[_j],"tier threshold":_thr,"weighted rate":_rates[-1],"rows":int(_s.sum())})
            _axs[_axis].plot(range(9),_rates,marker="o",ms=3,label=f"fold {_fold}")
        _bucket = np.minimum(np.searchsorted(_edges[1:],_a,side="right"),8)
        _means=[float(np.average(_pd[_bucket==_j],weights=_w[_bucket==_j])) if (_bucket==_j).any() else np.nan for _j in range(9)]
        _axs[2].plot(range(9),_means,marker="o",ms=3,label=f"fold {_fold}")
    for _ax in _axs:
        _ax.set_xticks(range(9),_labels,rotation=40,ha="right")
        _ax.grid(alpha=0.25)
        _ax.legend(fontsize=7)
    _axs[0].set_title("Causal rows since last quote event")
    _axs[1].set_title("Oracle rows until next event")
    _axs[2].set_title("Pair |direction| by causal age")
    _axs[0].set_ylabel(f"weighted P(winner tier >= {_thr})")
    _axs[1].set_ylabel(f"weighted P(winner tier >= {_thr})")
    _axs[2].set_ylabel("mean |p0 - p1|")
    _fig2.tight_layout()
    if _thr==2:
        _fig2.savefig(ROOT / "research" / "artifacts" / "brainteaser_tier_age.png",dpi=150)
    age_table = pl.DataFrame(_stats)
    mo.vstack([mo.md("""### Survival and future timing
    Left: available age; middle: unknowable time to the next event, shown only to bound headroom; right: what the current pair already expresses. Curves use official target weight on scored rows but are **conditional outcome plots**, not incremental-score estimates."""),_fig2])
    return


@app.cell
def _(
    EDATA,
    EHIST,
    EINPUT,
    FUT,
    ROOT,
    ROWDATA,
    head_table,
    history_table,
    horizon_table,
    mo,
    np,
    table_change,
):
    import json
    _art = ROOT / "research" / "artifacts"
    _event_path = _art / "brainteaser_event_snapshot.npz"
    _rows_path = _art / "brainteaser_scored_rows.parquet"
    _metrics_path = _art / "brainteaser_metrics.json"
    np.savez_compressed(_event_path,
        fold=EDATA["fold"].to_numpy(), seq=EDATA["seq"].to_numpy(), start=EDATA["start"].to_numpy(),
        true_reference=EDATA["F"].to_numpy(), ask=EDATA["ask"].to_numpy(),bid=EDATA["bid"].to_numpy(),
        metric_weight=EDATA["weight"].to_numpy(), winner_tick=EDATA["winner_tick"].to_numpy(),
        future_mids=FUT, causal_snapshot=EINPUT, causal_event_history=EHIST)
    ROWDATA.write_parquet(_rows_path)
    _payload={"date":"2026-09-27","public_champion":0.6622,"public_mixed":0.6616,
        "design":{"sample":"64 randomly selected pure-chain development class-3 sequences per fold 0/1",
          "decoder":"train-fitted class3_moves_full.pkl","future_mids":"oracle diagnostic only",
          "episode_snapshot":"first scored row in each selected book-price episode",
              "pair_predictions":"ab62_base_s4_f{k} + ab62_since_f{k}, held-out fold k=0/1",
          "head_probe":"HGB, 120 iterations, 15 leaf nodes, min leaf 80, weight capped at 100; train other fold"},
        "sample":{"sequences":EDATA["seq"].n_unique(),"episodes":len(EDATA),"scored_rows":len(ROWDATA),
          "both_target_implied_reference_agreement":float(EDATA["both_consistent"].mean())},
        "no_book_target_change":table_change.to_dicts(),
        "future_path_horizons":horizon_table.to_dicts(),
        "causal_tier_snapshot_probes":head_table.to_dicts(),
        "causal_event_history_probes":history_table.to_dicts(),
        "limitations":["future path is unavailable at inference", "event probe is one snapshot per episode, not official pooled score", "decoder is train-fitted", "class-3 tail is rare per independent episode", "two exploratory folds are not a submission gate"]}
    _metrics_path.write_text(json.dumps(_payload,indent=2)+"\n")
    mo.md(f"Saved evidence snapshot: `{_event_path.name}`, `{_rows_path.name}`, `{_metrics_path.name}`. The figures above are saved in the same research/artifacts directory.")
    return (json,)


@app.cell
def _(
    EDATA,
    EHIST,
    EINPUT,
    HistGradientBoostingClassifier,
    brier_score_loss,
    head_table,
    mo,
    np,
    pl,
    roc_auc_score,
    threadpool_limits,
):
    _fold3 = EDATA["fold"].to_numpy()
    _w3 = EDATA["weight"].to_numpy().astype(float)
    _lab3 = (EDATA["winner_tick"].to_numpy() >= 2).astype(np.int8)
    _p3 = EINPUT[:,:2]
    _basex = np.column_stack((_p3, _p3[:,0]-_p3[:,1], np.abs(_p3[:,0]-_p3[:,1]), _p3.sum(1), EINPUT[:,-1]))
    _focused = np.column_stack((_basex, EHIST[:,[9,11,14]]))
    _focus_rows = []
    with threadpool_limits(limits=2):
        for _k in (0,1):
            _train, _eval = _fold3 != _k, _fold3 == _k
            _model = HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=15,
                min_samples_leaf=80,learning_rate=0.05,l2_regularization=20,
                early_stopping=False,random_state=41)
            _model.fit(_focused[_train],_lab3[_train],sample_weight=np.minimum(_w3[_train],100))
            _prob = _model.predict_proba(_focused[_eval])[:,1]
            _ref = head_table.filter((pl.col("tier threshold")==2)&(pl.col("eval fold")==_k)&(pl.col("model")=="pair + age"))
            _auc = roc_auc_score(_lab3[_eval],_prob,sample_weight=_w3[_eval])
            _brier = brier_score_loss(_lab3[_eval],_prob,sample_weight=_w3[_eval])
            _focus_rows.append({"fold":_k,"focused AUC":_auc,"AUC gain":_auc-_ref["weighted AUC"][0],
                "focused Brier":_brier,"Brier improvement":_ref["weighted Brier"][0]-_brier})
    focus_table = pl.DataFrame(_focus_rows)
    mo.vstack([mo.md("""### Exploratory focused history screen
    Only past 8-event price drift and elapsed time over the past 8/16 quote events were added. Features were selected after inspecting both folds, so a positive result here needs a new held-out-chain check."""),focus_table])
    return (focus_table,)


@app.cell
def _(ROOT, focus_table, json, mo):
    _focus_path = ROOT / "research" / "artifacts" / "brainteaser_focused_screen.json"
    _focus_path.write_text(json.dumps({"date":"2026-09-27","features":"past 8-event price drift and elapsed time over past 8/16 quote events; chosen after inspecting folds 0/1","metrics":focus_table.to_dicts(),"status":"exploratory; AUC small positive, weighted Brier worse in both folds"},indent=2)+"\n")
    mo.md(f"Saved focused-history follow-up: `{_focus_path.name}`")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## What the EDA changes about the scout ideas

    1. **Training-only tier supervision is the first bounded experiment.** The eight-event oracle reveals why a four-event continuation label is too short for high-weight outcomes. A snapshot classifier's two-tick failure keeps the prior modest; the live GPU arm tests representation learning, not a production tier gate.
    2. **Observable continuation, not another target-change horizon, is the next head candidate.** The side is mainly the first quote move; payoff size involves later quote motion. A train-only successor-event head should predict the later path from current causal state and must beat a side-only control.
    3. **Selection-aware training remains the best proven class of ideas.** High-tier outcomes are rare independent episodes but common in scorer weight. The mixed pair's local gain existed only under mask softening, so the exact visible mask is the primary promotion gate.
    4. **Broad raw-state additions and simple event history lack tail evidence.** Their two-tick weighted Brier and AUC worsened on both folds. The focused 8/16-event history follow-up gave only a small exploratory AUC lift and worse Brier.
    5. **Deployment is part of the model.** The mixed package can use one GRU on v2 and two on v3. Local fallback loss is material, but its public execution path is unobserved.

    Every future quote-path plot above is diagnostic: the submitted predictor only sees the current and past state.
    """)
    return


@app.cell
def _(MASK, P0, P1, Q, RG, ROOT, ROWDATA, Y, json, mo, np, pl, plt):
    _rg_pos = {int(g):i for i,g in enumerate(RG)}
    _eb = {(s,t):np.zeros(3,float) for s in (False,True) for t in (0,1,2)}
    for _g in ROWDATA["seq"].unique().to_numpy():
        _i = _rg_pos[int(_g)]
        _m = MASK[_i].copy();_m[:99]=False
        _y = np.clip(Y[_i,_m],-2,2)
        _p = np.clip(0.5*(P0[_i,_m]+P1[_i,_m]),-2,2)
        _side = np.sign(_p[:,0]-_p[:,1])==np.sign(_y[:,0]-_y[:,1])
        _tier = np.minimum(2,np.maximum(0,np.rint(np.max(Y[_i,_m],axis=1)/Q).astype(int)))
        _weight = np.abs(_y).sum(1)
        _error = (np.abs(_y)*np.abs(_y-_p)).sum(1)
        for _s in (False,True):
            for _t in (0,1,2):
                _sel = (_side==_s)&(_tier==_t)
                _eb[(_s,_t)] += [_sel.sum(),_weight[_sel].sum(),_error[_sel].sum()]
    _erows=[]
    _totals=np.sum(list(_eb.values()),axis=0)
    for (_s,_t),_v in _eb.items():
        _erows.append({"side correct":_s,"tier":"2+" if _t==2 else str(_t),
           "scored row share":_v[0]/_totals[0],"target weight share":_v[1]/_totals[1],
           "weighted abs error share":_v[2]/_totals[2]})
    error_budget=pl.DataFrame(_erows).sort("tier","side correct",descending=[False,True])
    _lbl=[f"{'correct' if r['side correct'] else 'wrong'} / {r['tier']}" for r in error_budget.to_dicts()]
    _x=np.arange(len(_lbl))
    _fig3,_ax=plt.subplots(figsize=(9.5,3.7))
    for _j,(_name,_color) in enumerate((("scored row share","#8aa3b5"),("target weight share","#2679c9"),("weighted abs error share","#e16637"))):
        _ax.bar(_x+(_j-1)*0.24,error_budget[_name].to_numpy(),0.23,label=_name,color=_color)
    _ax.set_xticks(_x,_lbl,rotation=25,ha="right")
    _ax.set_ylabel("fraction of class-3 total")
    _ax.set_title("Where the pair's class-3 absolute error lives")
    _ax.legend(fontsize=8);_ax.grid(axis="y",alpha=0.2)
    _fig3.tight_layout()
    _fig3.savefig(ROOT / "research" / "artifacts" / "brainteaser_error_budget.png",dpi=150)
    (ROOT / "research" / "artifacts" / "brainteaser_error_budget.json").write_text(json.dumps(error_budget.to_dicts(),indent=2)+"\n")
    mo.vstack([mo.md("""### Error budget by realized side and tier
    The orange bars are a descriptive weighted absolute error budget, not the competition's pooled Pearson decomposition. Outcome-defined buckets use future labels; they cannot be used for inference routing."""),_fig3,error_budget])
    return


@app.cell
def _(EDATA, MASK, Q, RG, ROOT, Y, json, mo, np, pl, plt):
    _agecache = np.load(ROOT / "cache" / "rte_valid.npz")["since_last"]
    _rg2idx = {int(g):i for i,g in enumerate(RG)}
    _sf = {int(r["seq"]):int(r["fold"]) for r in EDATA.select("seq","fold").unique().to_dicts()}
    _edges2=np.array([0,1,2,4,8,16,32,64,128,30000])
    _count=np.zeros((2,2,9,2),np.float64)
    _mass2=np.zeros_like(_count)
    for _g,_fold in _sf.items():
        _i=_rg2idx[_g]
        _a=_agecache[_i,99:]
        _m=MASK[_i,99:].astype(np.int8)
        _t=(np.rint(np.max(Y[_i,99:],axis=1)/Q)>=2).astype(np.int8)
        _b=np.minimum(np.searchsorted(_edges2[1:],_a,side="right"),8)
        _w=np.abs(np.clip(Y[_i,99:],-2,2)).sum(1)
        _id=((_fold*2+_m)*9+_b)*2+_t
        _count += np.bincount(_id,minlength=72).reshape(2,2,9,2)
        _mass2 += np.bincount(_id,weights=_w,minlength=72).reshape(2,2,9,2)
    _mask_rows=[]
    for _f in (0,1):
        for _m in (0,1):
            for _b,_label in enumerate(("0","1","2–3","4–7","8–15","16–31","32–63","64–127","128+")):
                _v=_count[_f,_m,_b]
                _wv=_mass2[_f,_m,_b]
                _mask_rows.append({"fold":_f,"scored":bool(_m),"age bin":_label,
                    "rows":int(_v.sum()),"P(tier >=2)":float(_v[1]/max(_v.sum(),1)),
                    "target-weighted P(tier >=2)":float(_wv[1]/max(_wv.sum(),1))})
    mask_tier_table=pl.DataFrame(_mask_rows)
    _fig4,_axes4=plt.subplots(1,2,figsize=(9.2,3.4),sharey=True)
    for _f in (0,1):
        for _m,_style in ((0,"--"),(1,"-")):
            _z=mask_tier_table.filter((pl.col("fold")==_f)&(pl.col("scored")==bool(_m)))
            _axes4[_f].plot(np.arange(9),_z["P(tier >=2)"],marker="o",ms=3,
                linestyle=_style,label="scored" if _m else "not scored")
        _axes4[_f].set_title(f"fold {_f}")
        _axes4[_f].set_xticks(np.arange(9),("0","1","2–3","4–7","8–15","16–31","32–63","64–127","128+"),rotation=45,ha="right")
        _axes4[_f].grid(alpha=0.2);_axes4[_f].legend(fontsize=8)
    _axes4[0].set_ylabel("P(winner tier ≥ 2 | current age, mask)")
    _fig4.tight_layout()
    _fig4.savefig(ROOT/"research"/"artifacts"/"brainteaser_mask_tier.png",dpi=150)
    (ROOT/"research"/"artifacts"/"brainteaser_mask_tier.json").write_text(json.dumps(mask_tier_table.to_dicts(),indent=2)+"\n")
    mo.vstack([mo.md("""### Does the scorer select high-tier states at the same causal age?
    Current age is available to the model. The scorer mask and realized tier are only visible during validation. A scored/unscored gap at equal age motivates better **training selection**, never use of the hidden mask at inference."""),_fig4])
    return


@app.cell
def _(EDATA, MASK, Q, RG, ROOT, Y, json, mo, np, pl):
    _prop=np.load(ROOT/"cache"/"prop_v1_valid.npy",mmap_mode="r")
    _age=np.load(ROOT/"cache"/"rte_valid.npz")["since_last"]
    _lookup={int(g):i for i,g in enumerate(RG)}
    _seqfold2={int(r["seq"]):int(r["fold"]) for r in EDATA.select("seq","fold").unique().to_dicts()}
    _counts=np.zeros((2,9,5,2,2),np.float64)
    for _g,_f in _seqfold2.items():
        _i=_lookup[_g]
        _ab=np.minimum(np.searchsorted([1,2,4,8,16,32,64,128],_age[_i,99:],side="right"),8)
        _pb=np.minimum(np.searchsorted([.02,.05,.1,.2],_prop[_i,99:],side="right"),4)
        _ms=MASK[_i,99:].astype(int)
        _t=(np.rint(np.max(Y[_i,99:],axis=1)/Q)>=2).astype(int)
        _code=((((_f*9+_ab)*5+_pb)*2+_ms)*2+_t)
        _counts+=np.bincount(_code,minlength=360).reshape(_counts.shape)
    _prop_rows=[]
    for _f in (0,1):
        _n0=_counts[_f,:,:,0].sum(-1);_n1=_counts[_f,:,:,1].sum(-1)
        _p0=_counts[_f,:,:,0,1]/np.maximum(_n0,1);_p1=_counts[_f,:,:,1,1]/np.maximum(_n1,1)
        _valid=(_n0>=100)&(_n1>=100)
        _match=np.minimum(_n0,_n1)*_valid
        for _j,_name in enumerate(("<.02",".02–.05",".05–.10",".10–.20","≥.20")):
            _wt=_match[:,_j]
            _prop_rows.append({"fold":_f,"propensity":_name,"matched row count":int(_wt.sum()),
                "age-matched scored minus unscored tier≥2":float(np.sum(_wt*(_p1[:,_j]-_p0[:,_j]))/max(_wt.sum(),1))})
        _prop_rows.append({"fold":_f,"propensity":"all","matched row count":int(_match.sum()),
            "age-matched scored minus unscored tier≥2":float(np.sum(_match*(_p1-_p0))/max(_match.sum(),1))})
    prop_gap_table=pl.DataFrame(_prop_rows)
    (ROOT/"research"/"artifacts"/"brainteaser_propensity_gap.json").write_text(json.dumps(prop_gap_table.to_dicts(),indent=2)+"\n")
    mo.vstack([mo.md("""### Conditioning on the existing scoring propensity
    Within causal age and prop_v1 bins (at least 100 rows from each mask group), the scored-minus-unscored two-tick prevalence gap is ~0 on fold 0 and +1.5 percentage points on fold 1. The visible propensity already explains much of the mask/tier association; this does not establish a new mask head gain."""),prop_gap_table])
    return


@app.cell(hide_code=True)
def _(EDATA):

    import json as _json
    import numpy as _np
    import matplotlib.pyplot as _plt
    from pathlib import Path as _Path

    # Weight concentration among independently triggered quote episodes.
    _ec_fold = EDATA['fold'].to_numpy()
    _ec_tier = EDATA['winner_tick'].to_numpy()
    _ec_weight = EDATA['weight'].to_numpy().astype(float)
    EPISODE_CONCENTRATION = []
    _fig_ec, _axes_ec = _plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for _f, _ax in enumerate(_axes_ec):
        for _label, _select in [('all', _np.ones(len(_ec_tier), dtype=bool)),
                                 ('touch', _ec_tier == 0),
                                 ('one tick', _ec_tier == 1),
                                 ('two+ ticks', _ec_tier >= 2)]:
            _w = _np.sort(_ec_weight[(_ec_fold == _f) & _select])[::-1]
            _total = float(_w.sum())
            _n = len(_w)
            _k1 = max(1, int(_np.ceil(.01 * _n)))
            _k10 = max(1, int(_np.ceil(.10 * _n)))
            EPISODE_CONCENTRATION.append({'fold': _f, 'tier': _label, 'episodes': _n,
                  'kish_n': round(_total ** 2 / float((_w ** 2).sum()), 1),
                  'top_1pct_share': round(float(_w[:_k1].sum() / _total), 4),
                  'top_10pct_share': round(float(_w[:_k10].sum() / _total), 4)})
            _x = _np.arange(1, _n + 1) / _n
            _ax.plot(_x, _np.cumsum(_w) / _total, label=_label, linewidth=1.8)
        _ax.plot([0, 1], [0, 1], color='black', alpha=.35, linestyle='--', label='equal')
        _ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='Fraction of episodes, highest weight first',
                title=f'Development fold {_f}')
        _ax.grid(alpha=.18)
    _axes_ec[0].set_ylabel('Fraction of target weight')
    _axes_ec[1].legend(loc='lower right', fontsize=8)
    _fig_ec.suptitle('Scorer weight is concentrated in a few quote episodes')
    _fig_ec.tight_layout()
    _Path('research/artifacts').mkdir(exist_ok=True, parents=True)
    _fig_ec.savefig('research/artifacts/brainteaser_episode_concentration.png', dpi=160)
    _Path('research/artifacts/brainteaser_episode_concentration.json').write_text(
        _json.dumps(EPISODE_CONCENTRATION, indent=2))
    _fig_ec

    return


@app.cell(hide_code=True)
def _(EDATA, pl):

    import json as _json
    import numpy as _np
    import matplotlib.pyplot as _plt
    from pathlib import Path as _Path

    # Correlation evaluation is global, but strong target weight can cluster by sequence.
    _seqw = EDATA.group_by(['fold', 'seq']).agg(pl.col('weight').sum()).sort(['fold', 'seq'])
    SEQUENCE_CONCENTRATION = []
    _fig_sc, _ax_sc = _plt.subplots(figsize=(6.5, 4))
    for _f in (0, 1):
        _w = _np.sort(_seqw.filter(pl.col('fold') == _f)['weight'].to_numpy().astype(float))[::-1]
        _total = _w.sum()
        _n = len(_w)
        SEQUENCE_CONCENTRATION.append({'fold': _f, 'sequences': _n,
            'kish_n': round(float(_total * _total / (_w * _w).sum()), 1),
            'top_10pct_share': round(float(_w[:max(1, int(_np.ceil(.1 * _n)))].sum() / _total), 4)})
        _ax_sc.plot(_np.arange(1, _n + 1) / _n, _np.cumsum(_w) / _total,
                    marker='.', ms=3, label=f'fold {_f}')
    _ax_sc.plot([0, 1], [0, 1], 'k--', alpha=.35, label='equal')
    _ax_sc.set(xlabel='Fraction of sequences, highest weight first',
               ylabel='Fraction of target weight', title='Scorer weight also concentrates by sequence')
    _ax_sc.grid(alpha=.18)
    _ax_sc.legend()
    _fig_sc.tight_layout()
    _fig_sc.savefig('research/artifacts/brainteaser_sequence_concentration.png', dpi=160)
    _Path('research/artifacts/brainteaser_sequence_concentration.json').write_text(
        _json.dumps(SEQUENCE_CONCENTRATION, indent=2))
    _fig_sc

    return


@app.cell(hide_code=True)
def _(EDATA, FOLD, MASK, P0, P1, Q, RG, ROOT, SequenceFile, Y, pl):

    import pickle as _pickle
    import json as _json
    import numpy as _np
    from wnn.gmetric import wp_from_sums as _wp_from_sums

    # Cheap structural correction; all inputs and the spread table are causal.
    with (ROOT / 'cache' / 'class3_moves_full.pkl').open('rb') as _handle:
        _, _spreads = _pickle.load(_handle)
    _source_sc = SequenceFile(ROOT / 'datasets' / 'valid.parquet')
    _rg_to_index_sc = {int(g): i for i, g in enumerate(RG)}
    _selected_sc = sorted({int(g) for g in EDATA['seq'].to_list()})
    _alphas_sc = [0., .25, .5, 1.]
    _mom_sc = _np.zeros((2, len(_alphas_sc), 2, 6), dtype=float)
    _known_sc = _total_sc = 0
    for _g in _selected_sc:
        _i = _rg_to_index_sc[_g]
        _k = int(FOLD[_i])
        _x = _source_sc.read(_g, targets=False, mask=False).features
        _events = _np.r_[0, _np.flatnonzero(_np.any(_x[1:, :22] != _x[:-1, :22], axis=1)) + 1]
        _codes = [tuple(v) for v in _np.round(_x[_events, :22], 4)]
        _known_sc += sum(c in _spreads for c in _codes)
        _total_sc += len(_codes)
        _se = _np.array([_spreads.get(c, 1) for c in _codes], dtype=float)
        _spread_rows = _np.repeat(_se, _np.diff(_np.r_[_events, len(_x)]))
        _p = .5 * (_np.asarray(P0[_i, :], dtype=float) + _np.asarray(P1[_i, :], dtype=float))
        _yc = _np.clip(Y[_i].astype(float), -2, 2)
        _mask = MASK[_i]
        _d = (-Q * _spread_rows - _p.sum(axis=1)) / 2
        for _j, _alpha in enumerate(_alphas_sc):
            _pc = _np.clip(_p + _alpha * _d[:, None], -2, 2)
            for _t in range(2):
                _y = _yc[:, _t]
                _z = _pc[:, _t]
                _w = _np.abs(_y) * _mask
                _mom_sc[_k, _j, _t] += [_w.sum(), (_w*_y).sum(), (_w*_z).sum(),
                                          (_w*_y*_y).sum(), (_w*_z*_z).sum(), (_w*_y*_z).sum()]
    SPREAD_PROJECTION = []
    for _k in (0, 1):
        _baseline = float(_np.mean([_wp_from_sums(_mom_sc[_k, 0, _t]) for _t in range(2)]))
        for _j, _alpha in enumerate(_alphas_sc):
            _score = float(_np.mean([_wp_from_sums(_mom_sc[_k, _j, _t]) for _t in range(2)]))
            SPREAD_PROJECTION.append({'fold': _k, 'alpha': _alpha, 'score': _score,
                                      'delta': _score - _baseline})
    _json_path = ROOT / 'research' / 'artifacts' / 'brainteaser_spread_projection_sample.json'
    _json_path.write_text(_json.dumps({'rows': SPREAD_PROJECTION,
        'sequences': len(_selected_sc), 'known_code_fraction': _known_sc / _total_sc,
        'warning': 'sample of 64 sequences per fold used for idea development; confirm on untouched folds'}, indent=2))
    print('known spread code fraction', _known_sc / _total_sc)
    pl.DataFrame(SPREAD_PROJECTION)

    return


@app.cell(hide_code=True)
def _(CLS, DEV, FOLD, MASK, P0, P1, RG, ROOT, SequenceFile, Y, pl):

    import pickle as _pickle
    import json as _json
    import numpy as _np
    from wnn.ticks import reconstruct as _reconstruct
    from pathlib import Path as _Path

    # Frozen, three-feature causal correction suggested by the tier-age EDA.
    # Train on one pure-chain fold, evaluate all scored rows of the other.
    with (ROOT/'cache'/'class3_moves_full.pkl').open('rb') as _file_hr:
        _moves_hr, _spreads_hr = _pickle.load(_file_hr)
    _source_hr = SequenceFile(ROOT/'datasets'/'valid.parquet')
    _lookup_hr = {int(g):i for i,g in enumerate(RG)}
    _cf_hr = _np.load(ROOT/'cache'/'chain_folds.npz')
    _train_chains_hr = set(map(int, _cf_hr['chain'][_cf_hr['split']=='train']))
    _chain_for_group_hr = {int(g):int(c) for g,c,s in zip(_cf_hr['row_group'],_cf_hr['chain'],_cf_hr['split']) if s=='valid'}
    _development_hr = set(map(int, DEV))
    _selected_hr = [int(g) for i,g in enumerate(RG) if CLS[i]==3 and int(FOLD[i]) in (0,1)
                    and int(g) in _development_hr and _chain_for_group_hr[int(g)] not in _train_chains_hr]
    _parts_hr = {0:[],1:[]}
    for _group in _selected_hr:
        _i = _lookup_hr[_group]
        _fold = int(FOLD[_i])
        _x = _source_hr.read(_group,targets=False,mask=False).features
        _ask,_bid,_unknown = _reconstruct(_x,_moves_hr,_spreads_hr)
        _events = _np.r_[0,_np.flatnonzero(_np.any(_x[1:,:22]!=_x[:-1,:22],axis=1))+1]
        _mid = .5*(_ask[_events]+_bid[_events])
        _j = _np.arange(len(_events))
        _p8 = _np.maximum(0,_j-8)
        _p16 = _np.maximum(0,_j-16)
        _features_event = _np.column_stack((_mid-_mid[_p8],
            _np.log1p(_events-_events[_p8]),
            _np.log1p(_events-_events[_p16])))
        _features = _np.repeat(_features_event,_np.diff(_np.r_[_events,len(_x)]),axis=0)
        _selected_rows = MASK[_i] & (_np.arange(len(_x))>=99)
        _target = _np.clip(Y[_i,_selected_rows].astype(float),-2,2)
        _pred = .5*(_np.asarray(P0[_i,_selected_rows],float)+_np.asarray(P1[_i,_selected_rows],float))
        _parts_hr[_fold].append((_target,_pred,_features[_selected_rows]))
    HISTORY_RESIDUAL=[]
    for _eval in (0,1):
        _train = 1-_eval
        _yt = _np.concatenate([v[0] for v in _parts_hr[_train]])
        _pt = _np.concatenate([v[1] for v in _parts_hr[_train]])
        _ht = _np.concatenate([v[2][:,:1] for v in _parts_hr[_train]])
        _weight = _np.abs(_yt).mean(axis=1)
        _mu = _np.average(_ht,weights=_weight,axis=0)
        _sd = _np.sqrt(_np.average((_ht-_mu)**2,weights=_weight,axis=0)).clip(1e-6)
        _zt = (_ht-_mu)/_sd
        _coef=[]
        for _t in (0,1):
            _w = _np.abs(_yt[:,_t])
            _gram = _zt.T@(_w[:,None]*_zt)+.5*_w.sum()*_np.eye(1)
            _rhs = _zt.T@(_w*(_yt[:,_t]-_pt[:,_t]))
            _coef.append(_np.linalg.solve(_gram,_rhs))
        _coef = _np.column_stack(_coef)
        _ye = _np.concatenate([v[0] for v in _parts_hr[_eval]])
        _pe = _np.concatenate([v[1] for v in _parts_hr[_eval]])
        _he = _np.concatenate([v[2][:,:1] for v in _parts_hr[_eval]])
        _corrected = _np.clip(_pe+((_he-_mu)/_sd)@_coef,-2,2)
        def _wp_hr(y,p):
            _w=_np.abs(y); _ay=_np.average(y,weights=_w); _ap=_np.average(p,weights=_w)
            return float((_w*(y-_ay)*(p-_ap)).sum()/_np.sqrt((_w*(y-_ay)**2).sum()*(_w*(p-_ap)**2).sum()))
        _before = _np.mean([_wp_hr(_ye[:,t],_np.clip(_pe[:,t],-2,2)) for t in (0,1)])
        _after = _np.mean([_wp_hr(_ye[:,t],_corrected[:,t]) for t in (0,1)])
        HISTORY_RESIDUAL.append({'fold':_eval,'sequences':len(_parts_hr[_eval]),
            'scored_rows':len(_ye),'baseline':float(_before),'corrected':float(_after),
            'delta':float(_after-_before),'coef':_coef.tolist(),
            'mu':_mu.tolist(),'sd':_sd.tolist()})
    _Path(ROOT/'research'/'artifacts'/'brainteaser_history_residual_drift_full.json').write_text(_json.dumps(HISTORY_RESIDUAL,indent=2))
    pl.DataFrame([{k:v for k,v in r.items() if k not in ('coef','mu','sd')} for r in HISTORY_RESIDUAL])

    return


@app.cell(hide_code=True)
def _(ROWDATA):

    import json as _json
    import numpy as _np
    import matplotlib.pyplot as _plt
    from pathlib import Path as _Path

    # First-scored-event snapshots discard the age distribution seen by the scorer.
    _ab_fold=ROWDATA['fold'].to_numpy()
    _ab_age=ROWDATA['age'].to_numpy()
    _ab_weight=ROWDATA['weight'].to_numpy().astype(float)
    _ab_error=ROWDATA['weighted absolute error'].to_numpy().astype(float)
    _ab_tier=ROWDATA['tier'].to_numpy()
    _ab_edges=_np.array([0,1,2,4,8,16,32,64,128,256,_np.inf])
    _ab_labels=['0','1','2–3','4–7','8–15','16–31','32–63','64–127','128–255','256+']
    AGE_ERROR_BUDGET=[]
    _ab_fig,_ab_axes=_plt.subplots(1,2,figsize=(12,4),sharey=True)
    for _fold,_ax in enumerate(_ab_axes):
        _in=_ab_fold==_fold
        _ids=_np.searchsorted(_ab_edges[1:],_ab_age,side='right')
        _shares=[]
        for _j,_label in enumerate(_ab_labels):
            _z=_in&(_ids==_j)
            _row=float(_z.sum()/_in.sum())
            _w=float(_ab_weight[_z].sum()/_ab_weight[_in].sum())
            _e=float(_ab_error[_z].sum()/_ab_error[_in].sum())
            AGE_ERROR_BUDGET.append({'fold':_fold,'age':_label,'rows':int(_z.sum()),
                'row_share':_row,'target_weight_share':_w,'weighted_error_share':_e,
                'tier2_weighted_rate':float(_ab_weight[_z&(_ab_tier>=2)].sum()/max(_ab_weight[_z].sum(),1e-8))})
        _tab=[r for r in AGE_ERROR_BUDGET if r['fold']==_fold]
        _xx=_np.arange(len(_tab))
        _ax.bar(_xx-.25,[r['row_share'] for r in _tab],width=.24,label='rows')
        _ax.bar(_xx,[r['target_weight_share'] for r in _tab],width=.24,label='target weight')
        _ax.bar(_xx+.25,[r['weighted_error_share'] for r in _tab],width=.24,label='weighted error')
        _ax.set(xticks=_xx,xticklabels=_ab_labels,title=f'Development fold {_fold}',xlabel='Rows since current price event')
        _ax.tick_params(axis='x',rotation=45)
        _ax.grid(axis='y',alpha=.2)
    _ab_axes[0].set_ylabel('Share within sampled scored rows')
    _ab_axes[1].legend(fontsize=8)
    _ab_fig.suptitle('Class-3 scorer mass across the age of a quote event')
    _ab_fig.tight_layout()
    _ab_fig.savefig('research/artifacts/brainteaser_age_error_budget.png',dpi=160)
    _Path('research/artifacts/brainteaser_age_error_budget.json').write_text(_json.dumps(AGE_ERROR_BUDGET,indent=2))
    _ab_fig

    return


@app.cell(hide_code=True)
def _(CLS, FOLD, MASK, P0, P1, RG, ROOT, Y, json, mo, np, pl):

    import decide as _decide
    _cf = np.load(ROOT / "cache/chain_folds.npz")
    _cv = _cf["split"] == "valid"
    _chain_by_rg = {int(g): int(c) for g,c in zip(_cf["row_group"][_cv], _cf["chain"][_cv])}
    _train_chains = set(map(int, _cf["chain"][_cf["split"] == "train"]))
    _CHAINS = np.array([_chain_by_rg[int(g)] for g in RG])
    _ORACLE_IDX = np.flatnonzero(np.isin(FOLD, [0,1]) & ~np.isin(_CHAINS, list(_train_chains)))
    _ORACLE_MODES = ("base", "true_sign", "true_magnitude", "true_common", "true_contrast")
    _ORACLE_SUMS = {name:np.zeros((len(_ORACLE_IDX),2,2,6),np.float64) for name in _ORACLE_MODES}
    _ORACLE_BUDGET = np.zeros((4,2,3),np.float64)
    for _j,_i in enumerate(_ORACLE_IDX):
        _sel=MASK[_i].copy(); _sel[:99]=False
        if not _sel.any(): continue
        _yy=np.clip(Y[_i,_sel].astype(np.float64),-2,2)
        _pp=np.clip(0.5*(P0[_i,_sel]+P1[_i,_sel]),-2,2).astype(np.float64)
        _cm=0.5*(_pp[:,0]+_pp[:,1]); _ct=0.5*(_pp[:,0]-_pp[:,1])
        _yc=0.5*(_yy[:,0]+_yy[:,1]); _yt=0.5*(_yy[:,0]-_yy[:,1])
        _variants={"base":_pp,"true_sign":np.sign(_yy)*np.abs(_pp),
          "true_magnitude":np.sign(_pp)*np.abs(_yy),
          "true_common":np.column_stack((_yc+_ct,_yc-_ct)),
          "true_contrast":np.column_stack((_cm+_yt,_cm-_yt))}
        for _name,_pred in _variants.items():
          _pred=np.clip(_pred,-2,2)
          for _k in (0,1):
            _w=np.abs(_yy[:,_k]); _y=_yy[:,_k]; _p=_pred[:,_k]
            _ORACLE_SUMS[_name][_j,0,_k]=(_w.sum(),np.dot(_w,_y),np.dot(_w,_p),np.dot(_w,_y*_y),np.dot(_w,_p*_p),np.dot(_w,_y*_p))
        for _k in (0,1):
          _w=np.abs(_yy[:,_k]); _wrong=(_yy[:,_k]*_pp[:,_k])<0
          _err=np.abs(_yy[:,_k]-_pp[:,_k]); _cl=CLS[_i]
          _ORACLE_BUDGET[_cl,_k,0]+=np.sum(_w*_err)
          _ORACLE_BUDGET[_cl,_k,1]+=np.sum(_w[_wrong]*_err[_wrong])
          _ORACLE_BUDGET[_cl,_k,2]+=np.sum(_w*_err**2)
    _ORACLE_TABLE=[]
    for _mode in _ORACLE_MODES:
      _row={"mode":_mode,"all_wp":_decide.wp(_ORACLE_SUMS[_mode])}
      for _class in range(4):
        _row[f"class{_class}_wp"]=_decide.wp(_ORACLE_SUMS[_mode][CLS[_ORACLE_IDX]==_class])
      _ORACLE_TABLE.append(_row)
    _ORACLE_REPORT={"n_sequences":len(_ORACLE_IDX),"scores":_ORACLE_TABLE,
      "wrong_sign_weighted_abs_error_share":(_ORACLE_BUDGET[:,:,1].sum(1)/_ORACLE_BUDGET[:,:,0].sum(1)).tolist(),
      "weighted_abs_error_by_class":_ORACLE_BUDGET[:,:,0].sum(1).tolist()}
    (ROOT/"research/artifacts/brainteaser_oracle_headroom.json").write_text(json.dumps(_ORACLE_REPORT,indent=2))
    mo.vstack([mo.md("## Direction, size, common movement: noncausal oracle headroom"),pl.DataFrame(_ORACLE_TABLE),mo.md(f"Wrong-sign weighted absolute error share by class: {_ORACLE_REPORT['wrong_sign_weighted_abs_error_share']}")])

    return


@app.cell(hide_code=True)
def _(CLS, FOLD, P0, P1, RG, ROOT, Y, mo, np, pl, plt):
    _cf=np.load(ROOT/"cache/chain_folds.npz")
    _valid=_cf["split"]=="valid"
    _chainmap={int(g):int(c) for g,c in zip(_cf["row_group"][_valid],_cf["chain"][_valid])}
    _train=set(map(int,_cf["chain"][_cf["split"]=="train"]))
    _RESID_IDX=np.flatnonzero(np.isin(FOLD,[0,1]) & ~np.isin(np.array([_chainmap[int(g)] for g in RG]),list(_train)))

    from scipy.signal import welch as _welch
    import json as _json
    _RTE = np.load(ROOT/"cache/rte_valid.npz")["since_last"]
    def _acf_fft(_v, _maxlag=256):
        _v=np.asarray(_v,np.float64)
        _v=_v-np.mean(_v)
        _n=len(_v)
        _m=1 << (2*_n-1).bit_length()
        _ft=np.fft.rfft(_v,n=_m)
        _c=np.fft.irfft(_ft*np.conj(_ft),n=_m)[:_maxlag+1]
        _c=_c/np.maximum(_n-np.arange(len(_c)),1)
        return _c/max(_c[0],1e-12)
    _RES_ACF_ROW=[]; _RES_ACF_EVENT=[]; _RES_ACF_SQ=[]; _RES_PSD=[]
    _RES_RUNS_EVENT=[]; _RES_RUNS_ROW=[]; _RES_HAAR=[]; _EVENT_INTERVAL=[]
    _RES_SEQ=[]
    for _i in _RESID_IDX[CLS[_RESID_IDX]==3]:
        _y=np.clip(Y[_i].astype(np.float64),-2,2)
        _p=np.clip(0.5*(P0[_i,np.arange(20000)]+P1[_i,np.arange(20000)]),-2,2).astype(np.float64)
        _e=0.5*((_y[:,0]-_y[:,1])-(_p[:,0]-_p[:,1]))
        _ev=np.flatnonzero(_RTE[_i]==0)
        _ev=_ev[_ev>=99]
        if len(_ev)<1024: continue
        _z=_e[_ev]
        _RES_SEQ.append(int(RG[_i]))
        _RES_ACF_ROW.append(_acf_fft(_e[99:],256))
        _RES_ACF_EVENT.append(_acf_fft(_z,256))
        _RES_ACF_SQ.append(_acf_fft((_z-np.mean(_z))**2,256))
        _f,_ps=_welch(_z-np.mean(_z),nperseg=512,noverlap=256)
        _RES_PSD.append(_ps/max(np.mean(_ps),1e-12))
        _EVENT_INTERVAL.extend(np.diff(_ev).tolist())
        _threshold=np.quantile(np.abs(_z),.75)
        _high=np.abs(_z)>=_threshold
        _starts=np.flatnonzero(_high & ~np.r_[False,_high[:-1]])
        _ends=np.flatnonzero(_high & ~np.r_[_high[1:],False])
        _RES_RUNS_EVENT.extend((_ends-_starts+1).tolist())
        _RES_RUNS_ROW.extend((_ev[_ends]-_ev[_starts]+1).tolist())
        _energies=[]
        for _scale in (1,2,4,8,16,32,64,128,256):
            _blocks=len(_z)//(2*_scale)
            _pair=_z[:_blocks*2*_scale].reshape(_blocks,2,_scale).mean(2)
            _energies.append(np.mean((_pair[:,1]-_pair[:,0])**2)/max(np.var(_z),1e-12))
        _RES_HAAR.append(_energies)
    _RES_ACF_ROW=np.mean(_RES_ACF_ROW,axis=0); _RES_ACF_EVENT=np.mean(_RES_ACF_EVENT,axis=0)
    _RES_ACF_SQ=np.mean(_RES_ACF_SQ,axis=0); _RES_PSD=np.mean(_RES_PSD,axis=0)
    _RES_HAAR=np.mean(_RES_HAAR,axis=0)
    def _lag_below(_a,_v):
        _j=np.flatnonzero(_a[1:]<_v)
        return int(_j[0]+1) if len(_j) else None
    _RES_REPORT={"n_sequences":len(_RES_SEQ),"acf_row_lag1":float(_RES_ACF_ROW[1]),
     "acf_event_lag1":float(_RES_ACF_EVENT[1]),"acf_sq_event_lag1":float(_RES_ACF_SQ[1]),
     "row_lag_below_half":_lag_below(_RES_ACF_ROW,.5),"event_lag_below_half":_lag_below(_RES_ACF_EVENT,.5),
     "row_lag_below_point1":_lag_below(_RES_ACF_ROW,.1),"event_lag_below_point1":_lag_below(_RES_ACF_EVENT,.1),
     "high_residual_run_events_p50_p90_p99":np.quantile(_RES_RUNS_EVENT,[.5,.9,.99]).tolist(),
     "high_residual_run_rows_p50_p90_p99":np.quantile(_RES_RUNS_ROW,[.5,.9,.99]).tolist(),
     "book_event_gap_rows_p50_p90_p99":np.quantile(_EVENT_INTERVAL,[.5,.9,.99]).tolist(),
     "haar_scales":[1,2,4,8,16,32,64,128,256],"haar_energy":_RES_HAAR.tolist()}
    (ROOT/"research/artifacts/brainteaser_residual_timescale.json").write_text(_json.dumps(_RES_REPORT,indent=2))
    _fig,_ax=plt.subplots(2,2,figsize=(12,7))
    _ax[0,0].plot(_RES_ACF_ROW,label="row"); _ax[0,0].plot(_RES_ACF_EVENT,label="book event")
    _ax[0,0].plot(_RES_ACF_SQ,label="squared residual, book event",alpha=.7)
    _ax[0,0].set_xlim(0,128); _ax[0,0].set_xlabel("lag"); _ax[0,0].set_title("Class-3 contrast residual ACF"); _ax[0,0].legend()
    _ax[0,1].loglog(_f[1:],_RES_PSD[1:]); _ax[0,1].set_xlabel("cycles / book event"); _ax[0,1].set_title("Welch spectrum (sequence mean)")
    _ax[1,0].plot([1,2,4,8,16,32,64,128,256],_RES_HAAR,marker="o"); _ax[1,0].set_xscale("log",base=2); _ax[1,0].set_xlabel("Haar block size, events"); _ax[1,0].set_title("Normalized residual difference energy")
    _ax[1,1].hist(np.minimum(_RES_RUNS_EVENT,40),bins=40,log=True); _ax[1,1].set_xlabel("top-quartile |residual| run, events (40+ capped)"); _ax[1,1].set_title("Peak lengths")
    _fig.tight_layout(); _fig.savefig(ROOT/"research/artifacts/brainteaser_residual_timescale.png",dpi=160); plt.close(_fig)
    mo.vstack([mo.md("## Error peaks and frequency structure"),pl.DataFrame([_RES_REPORT])])

    RES_FREQ=_f
    RES_PSD=_RES_PSD
    RES_ACF_EVENT=_RES_ACF_EVENT
    RES_ACF_SQ=_RES_ACF_SQ

    RTE=_RTE

    return RES_FREQ, RES_PSD, RTE


@app.cell(hide_code=True)
def _(
    CLS,
    FOLD,
    MASK,
    P0,
    P1,
    RES_FREQ,
    RES_PSD,
    ROOT,
    RTE,
    Y,
    mo,
    np,
    pl,
    plt,
):
    import json as _json

    from scipy.signal import find_peaks as _find_peaks, welch as _welch
    from scipy.ndimage import median_filter as _medfilt
    _peak,_prop=_find_peaks(RES_PSD,prominence=.2)
    _peak=sorted(_peak,key=lambda j:RES_PSD[j]/max(_medfilt(RES_PSD,size=13)[j],1e-9),reverse=True)
    _SPEC_TABLE=[{"frequency":float(RES_FREQ[j]),"period_events":float(1/RES_FREQ[j]),
     "power":float(RES_PSD[j]),"local_ratio":float(RES_PSD[j]/max(_medfilt(RES_PSD,size=13)[j],1e-9))} for j in _peak[:12]]
    _PSD_TARGET=[]; _PSD_PRED=[]; _PSD_RET=[]
    _PHASE=np.zeros((2,6,3),np.float64)
    for _i in np.flatnonzero(np.isin(FOLD,[0,1]) & (CLS==3)):
      _ev=np.flatnonzero(RTE[_i]==0); _ev=_ev[_ev>=99]
      if len(_ev)<1024:continue
      _y=np.clip(Y[_i,_ev].astype(np.float64),-2,2)
      _p=np.clip(.5*(P0[_i,_ev]+P1[_i,_ev]),-2,2).astype(np.float64)
      _dy=.5*(_y[:,0]-_y[:,1]); _dp=.5*(_p[:,0]-_p[:,1])
      _PSD_TARGET.append(_welch(_dy-_dy.mean(),nperseg=512,noverlap=256)[1])
      _PSD_PRED.append(_welch(_dp-_dp.mean(),nperseg=512,noverlap=256)[1])
      _sel=MASK[_i,_ev]
      _wt=np.abs(_y).sum(1)*_sel
      _ph=np.arange(len(_ev))%6
      _f=FOLD[_i]
      for _g in range(6):
        _s=_ph==_g; _w=_wt[_s]; _r=(_dy-_dp)[_s]
        _PHASE[_f,_g]+=[_w.sum(),np.dot(_w,_r),np.dot(_w,_r*_r)]
    _SPEC_COMPARE={"target":(np.mean(_PSD_TARGET,axis=0)/np.mean(_PSD_TARGET)).tolist(),
     "prediction":(np.mean(_PSD_PRED,axis=0)/np.mean(_PSD_PRED)).tolist(),
     "residual":RES_PSD.tolist(),"frequency":RES_FREQ.tolist(),"peaks":_SPEC_TABLE,
     "scored_event_phase_mean_residual":(_PHASE[:,:,1]/np.maximum(_PHASE[:,:,0],1)).tolist(),
     "scored_event_phase_weight":_PHASE[:,:,0].tolist()}
    (ROOT/"research/artifacts/brainteaser_spectral_phase.json").write_text(_json.dumps(_SPEC_COMPARE,indent=2))
    _fig,_ax=plt.subplots(1,2,figsize=(11,4))
    for _name in ("target","prediction","residual"):
      _ax[0].loglog(RES_FREQ[1:],_SPEC_COMPARE[_name][1:],label=_name)
    _ax[0].legend(); _ax[0].set_xlabel("cycles/book event"); _ax[0].set_title("Target vs prediction vs residual spectrum")
    for _f in (0,1):
      _ax[1].plot(range(6),_SPEC_COMPARE["scored_event_phase_mean_residual"][_f],marker="o",label=f"fold {_f}")
    _ax[1].legend(); _ax[1].set_xlabel("event index modulo 6"); _ax[1].set_title("Scored residual by causal event count")
    _fig.tight_layout();_fig.savefig(ROOT/"research/artifacts/brainteaser_spectral_phase.png",dpi=160);plt.close(_fig)
    mo.vstack([mo.md("## Is the six-event spectral peak actionable?"),pl.DataFrame(_SPEC_TABLE),pl.DataFrame([{"fold":f,**{f"phase{j}":v for j,v in enumerate(_SPEC_COMPARE["scored_event_phase_mean_residual"][f])}} for f in (0,1)])])

    return


@app.cell(hide_code=True)
def _(CLS, FOLD, MASK, P0, P1, RG, ROOT, RTE, Y, mo, np, pl):

    import decide as _decide
    import json as _json
    _cf=np.load(ROOT/"cache/chain_folds.npz")
    _cv=_cf["split"]=="valid"
    _chains={int(g):int(c) for g,c in zip(_cf["row_group"][_cv],_cf["chain"][_cv])}
    _train=set(map(int,_cf["chain"][_cf["split"]=="train"]))
    _idx=np.flatnonzero(np.isin(FOLD,[0,1]) & ~np.isin(np.array([_chains[int(g)] for g in RG]),list(_train)))
    _phase_stats=np.zeros((2,6,2),np.float64)
    for _i in _idx[CLS[_idx]==3]:
      _t=np.flatnonzero(MASK[_i] & (np.arange(20000)>=99))
      if not len(_t):continue
      _ph=(np.cumsum(RTE[_i]==0)-1)[_t]%6
      _y=np.clip(Y[_i,_t].astype(np.float64),-2,2)
      _p=np.clip(.5*(P0[_i,_t]+P1[_i,_t]),-2,2).astype(np.float64)
      _e=.5*((_y[:,0]-_y[:,1])-(_p[:,0]-_p[:,1]))
      _w=np.abs(_y).sum(1)
      for _g in range(6):
        _s=_ph==_g
        _phase_stats[FOLD[_i],_g]+=[_w[_s].sum(),np.dot(_w[_s],_e[_s])]
    _phase_bias=_phase_stats[:,:,1]/np.maximum(_phase_stats[:,:,0],1)
    _phase_bias-=np.sum(_phase_bias*_phase_stats[:,:,0],axis=1)[:,None]/np.sum(_phase_stats[:,:,0],axis=1)[:,None]
    _PHASE_SCORES={_a:np.zeros((len(_idx),2,2,6),np.float64) for _a in (0.,.25,.5,1.)}
    for _j,_i in enumerate(_idx):
      _t=np.arange(20000)
      _y=np.clip(Y[_i].astype(np.float64),-2,2)
      _p=np.clip(.5*(P0[_i,_t]+P1[_i,_t]),-2,2).astype(np.float64)
      _m=MASK[_i].copy();_m[:99]=False
      _u=(~MASK[_i]) & (_t>=99)
      if CLS[_i]==3:
        _ph=(np.cumsum(RTE[_i]==0)-1)%6
        _b=_phase_bias[1-FOLD[_i],_ph]
      else:
        _b=0.
      for _a in _PHASE_SCORES:
        _q=_p.copy()
        if CLS[_i]==3:
          _q[:,0]+=_a*_b;_q[:,1]-=_a*_b
        _q=np.clip(_q,-2,2)
        for _k in (0,1):
          _PHASE_SCORES[_a][_j,0,_k]=_decide._sums6(_y[:,_k][None],_q[:,_k][None],_m[None])[0]
          _PHASE_SCORES[_a][_j,1,_k]=_decide._sums6(_y[:,_k][None],_q[:,_k][None],_u[None])[0]
    _PHASE_TABLE=[]
    for _a,_sum in _PHASE_SCORES.items():
      _row={"alpha":_a,"all_lam0":_decide.wp(_sum),"all_lam02":_decide.wp(_sum,.02)}
      for _k in (0,1):
        _ss=_sum[FOLD[_idx]==_k]
        _row[f"fold{_k}_lam0"]=_decide.wp(_ss)
        _row[f"fold{_k}_class3_lam0"]=_decide.wp(_sum[(FOLD[_idx]==_k)&(CLS[_idx]==3)])
      _PHASE_TABLE.append(_row)
    _PHASE_REPORT={"phase_fit_mean_residual":_phase_bias.tolist(),"phase_weight":_phase_stats[:,:,0].tolist(),"scores":_PHASE_TABLE}
    (ROOT/"research/artifacts/brainteaser_phase6_gate.json").write_text(_json.dumps(_PHASE_REPORT,indent=2))
    mo.vstack([mo.md("## Cross-fitted six-event phase correction, exact scored rows"),pl.DataFrame(_PHASE_TABLE),pl.DataFrame(_phase_bias)])

    PAIR_SUMS=_PHASE_SCORES[0.0]
    PAIR_IDX=_idx

    return PAIR_IDX, PAIR_SUMS


@app.cell(hide_code=True)
def _(CLS, FOLD, P0, P1, RG, ROOT, RTE, Y, mo, np, pl):

    def _acf_lag1(v):
      centered=v-np.mean(v)
      return float(np.dot(centered[1:],centered[:-1])/max(np.dot(centered,centered),1e-12)*len(v)/(len(v)-1))
    import numba as _nb
    from scipy.optimize import minimize as _minimize
    import json as _json
    _cf=np.load(ROOT/"cache/chain_folds.npz");_cv=_cf["split"]=="valid"
    _chain={int(g):int(c) for g,c in zip(_cf["row_group"][_cv],_cf["chain"][_cv])}
    _train=set(map(int,_cf["chain"][_cf["split"]=="train"]))
    _idx=np.flatnonzero((CLS==3)&np.isin(FOLD,[0,1])&~np.isin(np.array([_chain[int(g)] for g in RG]),list(_train)))
    _series={0:[],1:[]}
    for _i in _idx:
      _ev=np.flatnonzero(RTE[_i]==0);_ev=_ev[_ev>=99]
      if len(_ev)<256:continue
      _y=np.clip(Y[_i,_ev].astype(np.float64),-2,2)
      _p=np.clip(.5*(P0[_i,_ev]+P1[_i,_ev]),-2,2).astype(np.float64)
      _series[FOLD[_i]].append(.5*((_y[:,0]-_y[:,1])-(_p[:,0]-_p[:,1])))
    @_nb.njit
    def _arma_error(_flat,_ends,_intercept,_ar,_ma):
      _out=np.empty_like(_flat)
      _start=0
      for _end in _ends:
        _last_x=0.;_last_e=0.
        for _t in range(_start,_end):
          _err=_flat[_t]-_intercept-_ar*_last_x-_ma*_last_e
          _out[_t]=_err
          _last_x=_flat[_t];_last_e=_err
        _start=_end
      return _out
    @_nb.njit
    def _ewma_nll(_err,_ends,_initial,_decay):
      _total=0.;_start=0
      for _end in _ends:
        _var=_initial
        for _t in range(_start,_end):
          _e=_err[_t]
          _total+=.5*(np.log(_var)+_e*_e/_var)
          _var=max(1e-6,_decay*_var+(1-_decay)*_e*_e)
        _start=_end
      return _total/len(_err)
    _flat={};_ends={}
    for _fold in (0,1):
      _flat[_fold]=np.concatenate(_series[_fold]).astype(np.float64)
      _ends[_fold]=np.cumsum([len(v) for v in _series[_fold]]).astype(np.int64)
    _ARMA_FITS={};_ARMA_EVAL=[]
    for _train_fold in (0,1):
      _a=_flat[_train_fold];_ends_train=_ends[_train_fold]
      _fun=lambda par:np.mean(_arma_error(_a,_ends_train,*par)**2)
      _fit=_minimize(_fun,[0.,.65,.1],method="L-BFGS-B",bounds=[(-.2,.2),(-.98,.98),(-.98,.98)],options={"maxiter":100,"ftol":1e-10})
      _ARMA_FITS[_train_fold]=_fit.x.tolist()
      _eval_fold=1-_train_fold;_b=_flat[_eval_fold]
      _innov=_arma_error(_b,_ends[_eval_fold],*_fit.x)
      _train_innov=_arma_error(_a,_ends_train,*_fit.x)
      _variance=max(np.mean(_train_innov**2),1e-6)
      _const=.5*(np.log(_variance)+np.mean(_innov**2)/_variance)
      _ewma=_ewma_nll(_innov,_ends[_eval_fold],_variance,.94)
      _ARMA_EVAL.append({"eval_fold":_eval_fold,"train_fold":_train_fold,
        "n_events":len(_b),"train_parameters":_fit.x.tolist(),
        "one_step_mse":float(np.mean(_innov**2)),"zero_forecast_mse":float(np.mean(_b**2)),
        "one_step_r2":float(1-np.mean(_innov**2)/np.mean(_b**2)),
        "constant_variance_nll":float(_const),"ewma94_variance_nll":float(_ewma),
        "innovation_acf_lag1":float(np.mean([_acf_lag1(v) for v in np.split(_innov,_ends[_eval_fold][:-1]) if len(v)>32])),
        "innovation_sq_acf_lag1":float(np.mean([_acf_lag1(v*v) for v in np.split(_innov,_ends[_eval_fold][:-1]) if len(v)>32]))})
    _ARMA_REPORT={"fits":_ARMA_FITS,"heldout":_ARMA_EVAL,
     "note":"Forecasts here use past realized target residuals, which are unavailable in test streaming; diagnostics only."}
    (ROOT/"research/artifacts/brainteaser_arma_garch_residual.json").write_text(_json.dumps(_ARMA_REPORT,indent=2))
    mo.vstack([mo.md("## ARMA(1,1) and EWMA variance on model errors (diagnostic only)"),pl.DataFrame(_ARMA_EVAL)])

    ARMA_FLAT=_flat
    ARMA_ENDS=_ends
    ARMA_FITS=_ARMA_FITS

    ARMA_ERROR=_arma_error

    return ARMA_ENDS, ARMA_ERROR, ARMA_FITS, ARMA_FLAT


@app.cell(hide_code=True)
def _(EDATA, FOLD, MASK, P0, P1, RG, ROOT, RTE, SequenceFile, Y, mo, np, pl):

    import pickle as _pickle
    import decide as _decide
    import json as _json
    from wnn.ticks import reconstruct as _reconstruct
    with (ROOT/"cache/class3_moves_full.pkl").open("rb") as _file:
      _moves,_spreads=_pickle.load(_file)
    _source=SequenceFile(ROOT/"datasets/valid.parquet")
    _by_rg={int(g):i for i,g in enumerate(RG)}
    _VOLS=[]
    BOOK_RETURNS=[]
    for _g in EDATA["seq"].unique().to_numpy():
      _i=_by_rg[int(_g)]
      _x=_source.read(int(_g),targets=False,mask=False).features
      _ask,_bid,_unknown=_reconstruct(_x,_moves,_spreads)
      _ev=np.flatnonzero(RTE[_i]==0)
      _ev=_ev[_ev>=1]
      if len(_ev)<64:continue
      _mid=.5*(_ask[_ev]+_bid[_ev])
      _ret=np.diff(_mid,prepend=_mid[0])
      BOOK_RETURNS.append((int(FOLD[_i]),int(_g),_ret.astype(np.float32)))
      _abs=np.r_[0.,np.cumsum(np.abs(_ret))]
      _j=np.searchsorted(_ev,np.arange(20000),side="right")-1
      _j=np.maximum(_j,0)
      _lag=np.maximum(_j-16,0)
      _vol=(_abs[_j+1]-_abs[_lag+1])/np.maximum(_j-_lag,1)
      _y=np.clip(Y[_i].astype(np.float64),-2,2)
      _p=np.clip(.5*(P0[_i,np.arange(20000)]+P1[_i,np.arange(20000)]),-2,2).astype(np.float64)
      _VOLS.append((int(FOLD[_i]),int(_g),_vol.astype(np.float32),_y.astype(np.float32),_p.astype(np.float32),MASK[_i]))
    _VOL_STATS=[]
    for _fold in (0,1):
      _v=np.concatenate([v[m & (np.arange(20000)>=99)] for f,g,v,y,p,m in _VOLS if f==_fold])
      _e=np.concatenate([(.5*(y[:,0]-y[:,1]-p[:,0]+p[:,1]))[m & (np.arange(20000)>=99)] for f,g,v,y,p,m in _VOLS if f==_fold])
      _w=np.concatenate([(np.abs(y[:,0])+np.abs(y[:,1]))[m & (np.arange(20000)>=99)] for f,g,v,y,p,m in _VOLS if f==_fold])
      _x=np.log1p(_v)
      _mx=np.dot(_w,_x)/_w.sum();_me=np.dot(_w,_e*_e)/_w.sum()
      _rho=np.dot(_w,(_x-_mx)*(_e*_e-_me))/np.sqrt(np.dot(_w,(_x-_mx)**2)*np.dot(_w,(_e*_e-_me)**2))
      _cuts=np.quantile(_v,[0,.25,.5,.75,1])
      _bins=[]
      for _b in range(4):
        _s=(_v>=_cuts[_b]) & (_v<=_cuts[_b+1] if _b==3 else _v<_cuts[_b+1])
        _bins.append({"bin":_b,"rows":int(_s.sum()),"weighted_mse":float(np.dot(_w[_s],_e[_s]**2)/max(_w[_s].sum(),1))})
      _VOL_STATS.append({"fold":_fold,"weighted_corr_log_vol_error_sq":float(_rho),"vol_quantiles":_cuts.tolist(),"bins":_bins})
    _VOL_REPORT={"n_sequences":len(_VOLS),"folds":_VOL_STATS}
    (ROOT/"research/artifacts/brainteaser_causal_volatility.json").write_text(_json.dumps(_VOL_REPORT,indent=2))
    mo.vstack([mo.md("## Does past quote volatility predict model-error bursts?"),pl.DataFrame([{"fold":x["fold"],"weighted_corr":x["weighted_corr_log_vol_error_sq"],"vol_quartiles":x["vol_quantiles"]} for x in _VOL_STATS]),pl.DataFrame([{"fold":x["fold"],**b} for x in _VOL_STATS for b in x["bins"]])])

    VOLS=_VOLS
    BY_RG=_by_rg

    return BOOK_RETURNS, BY_RG, VOLS


@app.cell(hide_code=True)
def _(ARMA_ENDS, ARMA_ERROR, ARMA_FITS, ARMA_FLAT, ROOT, mo, np, pl, plt):

    from scipy.signal import welch as _welch
    from scipy.ndimage import median_filter as _median_filter
    import json as _json
    _RAW_PS=[];_INNOV_PS=[];_D1_ACF=[];_INNOV_ACF=[];_INNOV_SQ_ACF=[];_HAAR_INNOV=[]
    _ARIMA_COMPARE=[]
    for _fold in (0,1):
      _x=ARMA_FLAT[_fold];_ends=ARMA_ENDS[_fold]
      _innov=ARMA_ERROR(_x,_ends,*ARMA_FITS[1-_fold])
      _starts=np.r_[0,_ends[:-1]]
      _diff=[]
      for _start,_end in zip(_starts,_ends):
        _raw=_x[_start:_end];_error=_innov[_start:_end]
        _r,_p0=_welch(_raw-_raw.mean(),nperseg=256,noverlap=128)
        _,_p1=_welch(_error-_error.mean(),nperseg=256,noverlap=128)
        _RAW_PS.append(_p0/max(_p0.mean(),1e-12));_INNOV_PS.append(_p1/max(_p1.mean(),1e-12))
        _dc=np.diff(_raw)
        _D1_ACF.append(float(np.corrcoef(_dc[1:],_dc[:-1])[0,1]))
        _INNOV_ACF.append(float(np.corrcoef(_error[1:],_error[:-1])[0,1]))
        _INNOV_SQ_ACF.append(float(np.corrcoef(_error[1:]**2,_error[:-1]**2)[0,1]))
        _diff.extend(_dc.tolist())
        _energy=[]
        for _scale in (1,2,4,8,16,32,64):
          _n=len(_error)//(2*_scale)
          _blocks=_error[:_n*2*_scale].reshape(_n,2,_scale).mean(2)
          _energy.append(float(np.mean(np.diff(_blocks,axis=1)**2)/max(np.var(_error),1e-12)))
        _HAAR_INNOV.append(_energy)
      _ARIMA_COMPARE.append({"fold":_fold,"arma_mse":float(np.mean(_innov**2)),
        "random_walk_d1_mse":float(np.mean(np.array(_diff)**2)),"raw_zero_mse":float(np.mean(_x**2))})
    _RAW_PS=np.mean(_RAW_PS,axis=0);_INNOV_PS=np.mean(_INNOV_PS,axis=0)
    _MID_F=int(np.argmin(np.abs(_r-1/6)))
    _report={"arma_vs_arima_random_walk":_ARIMA_COMPARE,
     "first_difference_acf_lag1":float(np.mean(_D1_ACF)),
     "arma_innovation_acf_lag1":float(np.mean(_INNOV_ACF)),
     "arma_innovation_squared_acf_lag1":float(np.mean(_INNOV_SQ_ACF)),
     "six_event_raw_spectral_peak_ratio":float(_RAW_PS[_MID_F]/_median_filter(_RAW_PS,size=13)[_MID_F]),
     "six_event_innovation_spectral_peak_ratio":float(_INNOV_PS[_MID_F]/_median_filter(_INNOV_PS,size=13)[_MID_F]),
     "innovation_haar_scales":[1,2,4,8,16,32,64],"innovation_haar_energy":np.mean(_HAAR_INNOV,axis=0).tolist()}
    (ROOT/"research/artifacts/brainteaser_arma_innovation_spectrum.json").write_text(_json.dumps(_report,indent=2))
    _fig,_ax=plt.subplots(1,2,figsize=(11,4))
    _ax[0].loglog(_r[1:],_RAW_PS[1:],label="raw model residual")
    _ax[0].loglog(_r[1:],_INNOV_PS[1:],label="ARMA innovation")
    _ax[0].set_xlabel("cycles/book event");_ax[0].legend();_ax[0].set_title("Event-clock error spectra")
    _ax[1].plot([1,2,4,8,16,32,64],np.mean(_HAAR_INNOV,axis=0),marker="o")
    _ax[1].set_xscale("log",base=2);_ax[1].set_xlabel("Haar block size, events")
    _ax[1].set_title("ARMA innovation scale energy")
    _fig.tight_layout();_fig.savefig(ROOT/"research/artifacts/brainteaser_arma_innovation_spectrum.png",dpi=160);plt.close(_fig)
    mo.vstack([mo.md("## What ARMA removes, and what it leaves"),pl.DataFrame(_ARIMA_COMPARE),pl.DataFrame([_report])])

    return


@app.cell(hide_code=True)
def _(BOOK_RETURNS, ROOT, mo, np, pl, plt):

    from scipy.signal import welch as _welch
    from scipy.ndimage import median_filter as _median_filter
    import json as _json
    def _acf_ret(_v):
      _v=_v.astype(np.float64)-np.mean(_v)
      _n=len(_v);_fft=np.fft.rfft(_v,n=1<<(2*_n-1).bit_length())
      _c=np.fft.irfft(_fft*np.conj(_fft))[:65]
      _c=_c/(_n-np.arange(65))
      return _c/max(_c[0],1e-12)
    _RETS_ACF=[];_RETS_SQ_ACF=[];_RETS_PSD=[];_COUNTS=[]
    for _fold,_g,_ret in BOOK_RETURNS:
      if len(_ret)<512:continue
      _r=_ret.astype(np.float64)
      _RETS_ACF.append(_acf_ret(_r))
      _RETS_SQ_ACF.append(_acf_ret(_r*_r))
      _f,_p=_welch(_r-_r.mean(),nperseg=512,noverlap=256)
      _RETS_PSD.append(_p/max(_p.mean(),1e-12))
      _COUNTS.append(int(len(_ret)))
    _RETS_ACF=np.mean(_RETS_ACF,axis=0);_RETS_SQ_ACF=np.mean(_RETS_SQ_ACF,axis=0)
    _RETS_PSD=np.mean(_RETS_PSD,axis=0)
    _j=int(np.argmin(np.abs(_f-1/6)))
    _RETURN_REPORT={"n_sequences":len(_COUNTS),"event_returns_acf_lag1":float(_RETS_ACF[1]),
     "event_returns_acf_lag6":float(_RETS_ACF[6]),"squared_returns_acf_lag1":float(_RETS_SQ_ACF[1]),
     "six_event_spectral_peak_ratio":float(_RETS_PSD[_j]/_median_filter(_RETS_PSD,size=13)[_j]),
     "event_counts_p10_p50_p90":np.quantile(_COUNTS,[.1,.5,.9]).tolist()}
    (ROOT/"research/artifacts/brainteaser_book_return_acf.json").write_text(_json.dumps(_RETURN_REPORT,indent=2))
    _fig,_ax=plt.subplots(1,2,figsize=(11,4))
    _ax[0].plot(_RETS_ACF,label="return");_ax[0].plot(_RETS_SQ_ACF,label="squared return")
    _ax[0].set_xlim(0,32);_ax[0].legend();_ax[0].set_xlabel("book-event lag");_ax[0].set_title("Observable quote returns")
    _ax[1].loglog(_f[1:],_RETS_PSD[1:]);_ax[1].axvline(1/6,color="red",ls="--",alpha=.6)
    _ax[1].set_xlabel("cycles/book event");_ax[1].set_title("Observable return spectrum")
    _fig.tight_layout();_fig.savefig(ROOT/"research/artifacts/brainteaser_book_return_acf.png",dpi=160);plt.close(_fig)
    mo.vstack([mo.md("## Is the error cycle visible in past quote returns?"),pl.DataFrame([_RETURN_REPORT])])

    return


@app.cell(hide_code=True)
def _(BOOK_RETURNS, BY_RG, ROOT, RTE, VOLS, mo, np, pl):

    import decide as _decide
    import json as _json
    _retmap={int(g):ret for fold,g,ret in BOOK_RETURNS}
    def _seasonal_x(_g,_time):
      _i=BY_RG[int(_g)]
      _ev=np.flatnonzero(RTE[_i]==0);_ev=_ev[_ev>=1]
      _ret=_retmap[int(_g)].astype(np.float64)
      _j=np.maximum(np.searchsorted(_ev,_time,side="right")-1,0)
      return np.column_stack([np.ones(len(_time))]+[_ret[np.maximum(_j-lag,0)] for lag in range(12)])
    _G=np.zeros((2,13,13));_B=np.zeros((2,13));_TOTAL=np.zeros(2)
    _BREAK=np.zeros((2,2,3))
    for _fold,_g,_vol,_y,_p,_m in VOLS:
      _t=np.flatnonzero(_m & (np.arange(20000)>=99))
      _x=_seasonal_x(_g,_t)
      _yc=.5*(_y[_t,0]-_y[_t,1]);_pc=.5*(_p[_t,0]-_p[_t,1]);_e=_yc-_pc
      _w=np.abs(_y[_t,0])+np.abs(_y[_t,1])
      _G[_fold]+=_x.T@(_w[:,None]*_x);_B[_fold]+=_x.T@(_w*_e);_TOTAL[_fold]+=_w.sum()
      _changed=np.abs(_x[:,1]-_x[:,7])>1e-5
      for _k,_sel in enumerate((~_changed,_changed)):
        _BREAK[_fold,_k]+=[_sel.sum(),_w[_sel].sum(),np.dot(_w[_sel],np.abs(_e[_sel]))]
    _SEASON_BETA=np.zeros((2,13))
    for _fold in (0,1):
      _pen=np.eye(13)*(.01*_TOTAL[_fold]);_pen[0,0]=.01*_TOTAL[_fold]*.01
      _SEASON_BETA[_fold]=np.linalg.solve(_G[_fold]+_pen,_B[_fold])
    _SEASON_SUMS={a:np.zeros((len(VOLS),2,2,6),np.float64) for a in (0.,.25,.5,1.)}
    for _i,(_fold,_g,_vol,_y,_p,_m) in enumerate(VOLS):
      _x=_seasonal_x(_g,np.arange(20000))
      _correction=_x@_SEASON_BETA[1-_fold]
      _scored=_m.copy();_scored[:99]=False
      _unscored=(~_m)&(np.arange(20000)>=99)
      for _a,_sum in _SEASON_SUMS.items():
        _q=_p.astype(np.float64).copy();_q[:,0]+=_a*_correction;_q[:,1]-=_a*_correction
        _q=np.clip(_q,-2,2)
        for _k in (0,1):
          _sum[_i,0,_k]=_decide._sums6(_y[:,_k][None],_q[:,_k][None],_scored[None])[0]
          _sum[_i,1,_k]=_decide._sums6(_y[:,_k][None],_q[:,_k][None],_unscored[None])[0]
    _SEASON_TABLE=[]
    _SAMPLE_FOLD=np.array([v[0] for v in VOLS])
    for _a,_sum in _SEASON_SUMS.items():
      _row={"alpha":_a,"sample_class3_lam0":_decide.wp(_sum),"sample_class3_lam02":_decide.wp(_sum,.02)}
      for _fold in (0,1):
        _row[f"fold{_fold}_lam0"]=_decide.wp(_sum[_SAMPLE_FOLD==_fold])
      _SEASON_TABLE.append(_row)
    _SEASON_REPORT={"n_sequences":len(VOLS),"fit_coefficients":_SEASON_BETA.tolist(),
     "scores":_SEASON_TABLE,"break_rows_weight_error":_BREAK.tolist()}
    (ROOT/"research/artifacts/brainteaser_seasonal_lag_readout.json").write_text(_json.dumps(_SEASON_REPORT,indent=2))
    mo.vstack([mo.md("## Seasonal AR lags against the GRU: opposite-fold residual fit"),pl.DataFrame(_SEASON_TABLE),pl.DataFrame(_SEASON_BETA)])

    return


@app.cell(hide_code=True)
def _(CLS, FOLD, MASK, P0, P1, RG, ROOT, RTE, Y, mo, np, pl, plt):

    from sklearn.metrics import roc_auc_score as _roc_auc_score
    import json as _json
    _cf=np.load(ROOT/"cache/chain_folds.npz");_cv=_cf["split"]=="valid"
    _chain={int(g):int(c) for g,c in zip(_cf["row_group"][_cv],_cf["chain"][_cv])}
    _train=set(map(int,_cf["chain"][_cf["split"]=="train"]))
    _idx=np.flatnonzero((CLS==3)&np.isin(FOLD,[0,1])&~np.isin(np.array([_chain[int(g)] for g in RG]),list(_train)))
    _DDATA=[]
    for _i in _idx:
      _t=np.flatnonzero(MASK[_i] & (np.arange(20000)>=99))
      if not len(_t):continue
      _y=np.clip(Y[_i,_t].astype(np.float64),-2,2)
      _p=np.clip(.5*(P0[_i,_t]+P1[_i,_t]),-2,2).astype(np.float64)
      _DDATA.append((int(FOLD[_i]),.5*(_y[:,0]-_y[:,1]),.5*(_p[:,0]-_p[:,1]),np.abs(_y).sum(1),RTE[_i,_t]))
    _DIR_STATS=[];_CAL_ROWS=[]
    for _fold in (0,1):
      _d=np.concatenate([d for f,d,p,w,a in _DDATA if f==_fold])
      _p=np.concatenate([p for f,d,p,w,a in _DDATA if f==_fold])
      _w=np.concatenate([w for f,d,p,w,a in _DDATA if f==_fold])
      _a=np.concatenate([a for f,d,p,w,a in _DDATA if f==_fold])
      _nonzero=np.abs(_d)>1e-6
      _good=(_d*_p)>0
      _acc=float(np.dot(_w[_nonzero],_good[_nonzero])/_w[_nonzero].sum())
      _auc=float(_roc_auc_score(_d[_nonzero]>0,_p[_nonzero],sample_weight=_w[_nonzero]))
      _abscorr=float(np.corrcoef(np.abs(_d),np.abs(_p))[0,1])
      _DIR_STATS.append({"fold":_fold,"n_scored_rows":len(_d),"nonzero_contrast_share":float(_nonzero.mean()),
       "weighted_direction_accuracy":_acc,"weighted_direction_auc":_auc,
       "absolute_movement_correlation":_abscorr,"weighted_direction_accuracy_age32plus":float(np.dot(_w[_nonzero&(_a>=32)],_good[_nonzero&(_a>=32)])/max(_w[_nonzero&(_a>=32)].sum(),1))})
      _edges=np.quantile(_p,np.linspace(0,1,11))
      for _k in range(10):
        _s=(_p>=_edges[_k]) & (_p<=_edges[_k+1] if _k==9 else _p<_edges[_k+1]) & _nonzero
        _CAL_ROWS.append({"fold":_fold,"prediction_decile":_k,"mean_predicted_contrast":float(np.dot(_w[_s],_p[_s])/max(_w[_s].sum(),1)),
           "realized_positive_rate":float(np.dot(_w[_s],(_d[_s]>0))/max(_w[_s].sum(),1)),"weighted_rows":float(_w[_s].sum())})
    _DIR_REPORT={"folds":_DIR_STATS,"calibration":_CAL_ROWS}
    (ROOT/"research/artifacts/brainteaser_direction_vs_size.json").write_text(_json.dumps(_DIR_REPORT,indent=2))
    _fig,_ax=plt.subplots(1,2,figsize=(11,4))
    for _fold in (0,1):
      _rr=[r for r in _CAL_ROWS if r["fold"]==_fold]
      _ax[0].plot([r["mean_predicted_contrast"] for r in _rr],[r["realized_positive_rate"] for r in _rr],marker="o",label=f"fold {_fold}")
    _ax[0].axhline(.5,color="gray",ls="--");_ax[0].legend();_ax[0].set_xlabel("weighted predicted contrast");_ax[0].set_ylabel("weighted P(actual contrast > 0)");_ax[0].set_title("Class-3 direction calibration")
    _ax[1].bar([0,1],[r["weighted_direction_accuracy"] for r in _DIR_STATS]);_ax[1].set_xticks([0,1],["fold 0","fold 1"]);_ax[1].set_ylim(0,1);_ax[1].set_title("Weighted direction accuracy")
    _fig.tight_layout();_fig.savefig(ROOT/"research/artifacts/brainteaser_direction_vs_size.png",dpi=160);plt.close(_fig)
    mo.vstack([mo.md("## Which head has real headroom?"),pl.DataFrame(_DIR_STATS)])

    DDATA=_DDATA

    return (DDATA,)


@app.cell(hide_code=True)
def _(CLS, FOLD, RG, ROOT, RTE, SequenceFile, mo, np, pl, plt):

    import json as _json
    _source=SequenceFile(ROOT/"datasets/valid.parquet")
    _rg_to_idx={int(g):i for i,g in enumerate(RG)}
    _rng=np.random.default_rng(17)
    _RECURRENCE=[]
    for _class in range(4):
      _choose=np.flatnonzero((CLS==_class)&np.isin(FOLD,[0,1]))
      _choose=_rng.choice(_choose,min(20,len(_choose)),replace=False)
      for _i in _choose:
        _x=_source.read(int(RG[_i]),targets=False,mask=False).features
        _ev=np.flatnonzero(RTE[_i]==0)
        _ev=_ev[_ev>=1]
        _code=_x[_ev,:22]
        if len(_code)<256:continue
        _match=[]
        for _lag in range(1,33):
          _match.append(float(np.mean(np.all(_code[_lag:]==_code[:-_lag],axis=1))))
        _RECURRENCE.append({"class":_class,"row_group":int(RG[_i]),"n_events":len(_ev),"match":_match})
    _REC_TABLE=[]
    for _class in range(4):
      _a=np.array([r["match"] for r in _RECURRENCE if r["class"]==_class])
      _REC_TABLE.append({"class":_class,"n_sequences":len(_a),"lag6_match":float(_a[:,5].mean()),
        "lag12_match":float(_a[:,11].mean()),"best_lag_1_to_16":int(np.argmax(_a.mean(0)[:16])+1),
        "lag1_match":float(_a[:,0].mean())})
    _REC_REPORT={"summary":_REC_TABLE,"sequences":_RECURRENCE}
    (ROOT/"research/artifacts/brainteaser_raw_code_recurrence.json").write_text(_json.dumps(_REC_REPORT,indent=2))
    _fig,_ax=plt.subplots(figsize=(9,4))
    for _class in range(4):
      _a=np.array([r["match"] for r in _RECURRENCE if r["class"]==_class])
      _ax.plot(range(1,33),_a.mean(0),label=f"class {_class}")
    _ax.set_xlabel("book-event lag");_ax.set_ylabel("exact raw price-code recurrence")
    _ax.legend();_ax.set_title("Raw feature recurrence, without tick reconstruction")
    _fig.tight_layout();_fig.savefig(ROOT/"research/artifacts/brainteaser_raw_code_recurrence.png",dpi=160);plt.close(_fig)
    mo.vstack([mo.md("## Six-event oscillator in raw price codes?"),pl.DataFrame(_REC_TABLE)])

    return


@app.cell(hide_code=True)
def _(CLS, FOLD, RG, ROOT, RTE, mo, np, pl, plt):

    from scipy.signal import welch as _welch
    import json as _json
    _cf=np.load(ROOT/"cache/chain_folds.npz");_cv=_cf["split"]=="valid"
    _chain={int(g):int(c) for g,c in zip(_cf["row_group"][_cv],_cf["chain"][_cv])}
    _train=set(map(int,_cf["chain"][_cf["split"]=="train"]))
    _idx=np.flatnonzero((CLS==3)&np.isin(FOLD,[0,1])&~np.isin(np.array([_chain[int(g)] for g in RG]),list(_train)))
    _GAPS={0:[],1:[]}
    for _i in _idx:
      _ev=np.flatnonzero(RTE[_i]==0);_ev=_ev[_ev>=1]
      if len(_ev)<256:continue
      _GAPS[FOLD[_i]].append(np.log1p(np.diff(_ev).astype(np.float64)))
    _GAP_ACF=[];_GAP_PSD=[]
    for _fold in (0,1):
      for _g in _GAPS[_fold]:
        _x=_g-_g.mean()
        _ft=np.fft.rfft(_x,n=1<<(2*len(_x)-1).bit_length())
        _c=np.fft.irfft(_ft*np.conj(_ft))[:33]
        _GAP_ACF.append(_c/max(_c[0],1e-12))
        _,_s=_welch(_x,nperseg=256,noverlap=128)
        _GAP_PSD.append(_s/max(_s.mean(),1e-12))
    _GAP_ACF=np.mean(_GAP_ACF,axis=0);_GAP_PSD=np.mean(_GAP_PSD,axis=0)
    _GAP_MODELS=[]
    for _train_fold in (0,1):
      _x=np.concatenate([v[:-1] for v in _GAPS[_train_fold]])
      _y=np.concatenate([v[1:] for v in _GAPS[_train_fold]])
      _center=_x.mean();_mean=_y.mean()
      _beta=np.dot(_x-_center,_y-_mean)/np.dot(_x-_center,_x-_center)
      _eval=1-_train_fold
      _xe=np.concatenate([v[:-1] for v in _GAPS[_eval]])
      _ye=np.concatenate([v[1:] for v in _GAPS[_eval]])
      _p=_mean+_beta*(_xe-_center)
      _r2=1-np.mean((_ye-_p)**2)/np.mean((_ye-_mean)**2)
      _GAP_MODELS.append({"eval_fold":_eval,"train_ar1_coefficient":float(_beta),"heldout_r2":float(_r2)})
    _GAP_REPORT={"n_sequences":sum(map(len,_GAPS.values())),"acf_lag1":float(_GAP_ACF[1]),
     "acf_lag6":float(_GAP_ACF[6]),"heldout_ar1":_GAP_MODELS}
    (ROOT/"research/artifacts/brainteaser_event_duration_ar.json").write_text(_json.dumps(_GAP_REPORT,indent=2))
    _fig,_ax=plt.subplots(1,2,figsize=(11,4))
    _ax[0].plot(_GAP_ACF);_ax[0].set_xlabel("event lag");_ax[0].set_title("Log price-event duration ACF")
    _ax[1].loglog(np.fft.rfftfreq(256)[1:],_GAP_PSD[1:]);_ax[1].set_xlabel("cycles/event");_ax[1].set_title("Duration spectrum")
    _fig.tight_layout();_fig.savefig(ROOT/"research/artifacts/brainteaser_event_duration_ar.png",dpi=160);plt.close(_fig)
    mo.vstack([mo.md("## Can elapsed quote-event timing be forecast causally?"),pl.DataFrame(_GAP_MODELS),pl.DataFrame([_GAP_REPORT])])

    GAPS=_GAPS

    return (GAPS,)


@app.cell(hide_code=True)
def _(GAPS, ROOT, mo, np, pl):

    import json as _json
    _DURATION_MODELS=[]
    for _lags in (1,6,12):
      for _fit_fold in (0,1):
        _gram=np.zeros((_lags+1,_lags+1));_rhs=np.zeros(_lags+1)
        for _g in GAPS[_fit_fold]:
          _n=len(_g)
          _x=np.column_stack([np.ones(_n-_lags)]+[_g[_lags-j-1:_n-j-1] for j in range(_lags)])
          _y=_g[_lags:]
          _gram+=_x.T@_x;_rhs+=_x.T@_y
        _pen=np.eye(_lags+1)*(.001*_gram[0,0]);_pen[0,0]=0
        _beta=np.linalg.solve(_gram+_pen,_rhs)
        _train_mean=_rhs[0]/_gram[0,0]
        _sse=_base=0.
        for _g in GAPS[1-_fit_fold]:
          _n=len(_g)
          _x=np.column_stack([np.ones(_n-_lags)]+[_g[_lags-j-1:_n-j-1] for j in range(_lags)])
          _y=_g[_lags:]
          _sse+=np.sum((_y-_x@_beta)**2)
          _base+=np.sum((_y-_train_mean)**2)
        _DURATION_MODELS.append({"lags":_lags,"eval_fold":1-_fit_fold,"heldout_r2":float(1-_sse/_base),"coefficients":_beta.tolist()})
    (ROOT/"research/artifacts/brainteaser_duration_seasonal_ar.json").write_text(_json.dumps(_DURATION_MODELS,indent=2))
    mo.vstack([mo.md("## Seasonal AR timing forecast, train on opposite fold"),pl.DataFrame([{"lags":v["lags"],"eval_fold":v["eval_fold"],"heldout_r2":v["heldout_r2"]} for v in _DURATION_MODELS])])

    return


@app.cell(hide_code=True)
def _(CLS, FOLD, GAPS, MASK, RG, ROOT, RTE, Y, mo, np, pl):

    import json as _json
    _models=_json.loads((ROOT/"research/artifacts/brainteaser_duration_seasonal_ar.json").read_text())
    _beta={int(x["eval_fold"]):np.asarray(x["coefficients"]) for x in _models if x["lags"]==12}
    _cf=np.load(ROOT/"cache/chain_folds.npz");_cv=_cf["split"]=="valid"
    _chain={int(g):int(c) for g,c in zip(_cf["row_group"][_cv],_cf["chain"][_cv])}
    _train=set(map(int,_cf["chain"][_cf["split"]=="train"]))
    _idx=np.flatnonzero((CLS==3)&np.isin(FOLD,[0,1])&~np.isin(np.array([_chain[int(g)] for g in RG]),list(_train)))
    _global_mean={k:float(np.mean(np.concatenate(GAPS[k]))) for k in (0,1)}
    _TIMING={0:[],1:[]}
    for _i in _idx:
      _ev=np.flatnonzero(RTE[_i]==0);_ev=_ev[_ev>=1]
      if len(_ev)<20:continue
      _g=np.log1p(np.diff(_ev).astype(np.float64))
      _coeff=_beta[int(FOLD[_i])]
      _mu=np.full(len(_ev),_global_mean[1-FOLD[_i]])
      _mu[12:len(_g)]=_coeff[0]
      for _lag in range(12):
        _mu[12:len(_g)]+=_coeff[_lag+1]*_g[11-_lag:len(_g)-1-_lag]
      _t=np.flatnonzero(MASK[_i] & (np.arange(20000)>=99))
      _j=np.searchsorted(_ev,_t,side="right")-1
      _ok=(_j>=12)&(_j<len(_g))
      _t=_t[_ok];_j=_j[_ok]
      _w=np.abs(np.clip(Y[_i,_t],-2,2)).sum(1)
      _TIMING[FOLD[_i]].append(np.column_stack((_mu[_j],_g[_j],np.log1p(_t-_ev[_j]),_w)))
    def _weighted_corr(_x,_y,_w):
      _mx=np.dot(_w,_x)/_w.sum();_my=np.dot(_w,_y)/_w.sum()
      return float(np.dot(_w,(_x-_mx)*(_y-_my))/np.sqrt(np.dot(_w,(_x-_mx)**2)*np.dot(_w,(_y-_my)**2)))
    _TIMING_STATS=[]
    for _fold in (0,1):
      _a=np.concatenate(_TIMING[_fold]);_p,_y,_age,_w=_a.T
      _basis=np.column_stack((np.ones(len(_a)),_age))
      _gram=_basis.T@(_w[:,None]*_basis)
      _xp=_p-_basis@np.linalg.solve(_gram,_basis.T@(_w*_p))
      _yp=_y-_basis@np.linalg.solve(_gram,_basis.T@(_w*_y))
      _TIMING_STATS.append({"fold":_fold,"scored_rows":len(_a),"raw_weighted_corr":_weighted_corr(_p,_y,_w),
        "partial_corr_after_age":_weighted_corr(_xp,_yp,_w),
        "weighted_loggap_r2_vs_fold_mean":float(1-np.dot(_w,(_y-_p)**2)/np.dot(_w,(_y-np.dot(_w,_y)/_w.sum())**2))})
    (ROOT/"research/artifacts/brainteaser_scored_timing_forecast.json").write_text(_json.dumps(_TIMING_STATS,indent=2))
    mo.vstack([mo.md("## Does seasonal gap forecasting survive scorer selection?"),pl.DataFrame(_TIMING_STATS)])

    return


@app.cell(hide_code=True)
def _(CLS, FOLD, GAPS, MASK, P0, P1, RG, ROOT, RTE, Y, mo, np, pl):

    import json as _json
    import decide as _decide
    _models=_json.loads((ROOT/"research/artifacts/brainteaser_duration_seasonal_ar.json").read_text())
    _duration_beta={int(x["eval_fold"]):np.asarray(x["coefficients"]) for x in _models if x["lags"]==12}
    _duration_mean={k:float(np.mean(np.concatenate(GAPS[k]))) for k in (0,1)}
    _cf=np.load(ROOT/"cache/chain_folds.npz");_cv=_cf["split"]=="valid"
    _chain={int(g):int(c) for g,c in zip(_cf["row_group"][_cv],_cf["chain"][_cv])}
    _train=set(map(int,_cf["chain"][_cf["split"]=="train"]))
    _idx=np.flatnonzero((CLS==3)&np.isin(FOLD,[0,1])&~np.isin(np.array([_chain[int(g)] for g in RG]),list(_train)))
    def _forecast_gap(_i):
      _ev=np.flatnonzero(RTE[_i]==0);_ev=_ev[_ev>=1]
      _g=np.log1p(np.diff(_ev).astype(np.float64))
      _fit=1-int(FOLD[_i]);_coef=_duration_beta[int(FOLD[_i])]
      _mu=np.full(len(_ev),_duration_mean[_fit])
      if len(_g)>=12:
        _mu[12:]=_coef[0]
        for _lag in range(12):
          _mu[12:]+=_coef[_lag+1]*_g[11-_lag:len(_g)-_lag]
      _j=np.maximum(np.searchsorted(_ev,np.arange(20000),side="right")-1,0)
      return _mu[_j]-_duration_mean[_fit]
    _G=np.zeros((2,2,2));_B=np.zeros((2,2));_W=np.zeros(2)
    for _i in _idx:
      _sel=MASK[_i].copy();_sel[:99]=False
      _p=np.clip(.5*(P0[_i,np.arange(20000)]+P1[_i,np.arange(20000)]),-2,2).astype(np.float64)
      _y=np.clip(Y[_i].astype(np.float64),-2,2)
      _x=_forecast_gap(_i)[_sel]
      _pd=.5*(_p[_sel,0]-_p[_sel,1]);_yd=.5*(_y[_sel,0]-_y[_sel,1])
      _X=np.column_stack((_x,_x*_pd));_w=np.abs(_y[_sel]).sum(1)
      _f=FOLD[_i]
      _G[_f]+=_X.T@(_w[:,None]*_X);_B[_f]+=_X.T@(_w*(_yd-_pd));_W[_f]+=_w.sum()
    _TIMING_BETA=np.array([np.linalg.solve(_G[f]+np.eye(2)*(.01*_W[f]),_B[f]) for f in (0,1)])
    _TIMING_SUMS={a:np.zeros((len(_idx),2,2,6)) for a in (0.,.25,.5,1.)}
    for _j,_i in enumerate(_idx):
      _t=np.arange(20000)
      _p=np.clip(.5*(P0[_i,_t]+P1[_i,_t]),-2,2).astype(np.float64)
      _y=np.clip(Y[_i].astype(np.float64),-2,2)
      _x=_forecast_gap(_i);_pd=.5*(_p[:,0]-_p[:,1])
      _delta=_TIMING_BETA[1-FOLD[_i],0]*_x+_TIMING_BETA[1-FOLD[_i],1]*_x*_pd
      _scored=MASK[_i].copy();_scored[:99]=False
      _unscored=(~MASK[_i])&(_t>=99)
      for _alpha,_sum in _TIMING_SUMS.items():
        _q=_p.copy();_q[:,0]+=_alpha*_delta;_q[:,1]-=_alpha*_delta
        _q=np.clip(_q,-2,2)
        for _k in (0,1):
          _sum[_j,0,_k]=_decide._sums6(_y[:,_k][None],_q[:,_k][None],_scored[None])[0]
          _sum[_j,1,_k]=_decide._sums6(_y[:,_k][None],_q[:,_k][None],_unscored[None])[0]
    _TIMING_TABLE=[]
    for _alpha,_sum in _TIMING_SUMS.items():
      _row={"alpha":_alpha,"class3_wp_lam0":_decide.wp(_sum),"class3_wp_lam02":_decide.wp(_sum,.02)}
      for _f in (0,1):
        _row[f"fold{_f}_lam0"]=_decide.wp(_sum[FOLD[_idx]==_f])
      _TIMING_TABLE.append(_row)
    _TIMING_REPORT={"n_class3_sequences":len(_idx),"fit_residual_coefficients":_TIMING_BETA.tolist(),"scores":_TIMING_TABLE}
    (ROOT/"research/artifacts/brainteaser_timing_residual_gate.json").write_text(_json.dumps(_TIMING_REPORT,indent=2))
    mo.vstack([mo.md("## Can seasonal duration forecast repair the current pair?"),pl.DataFrame(_TIMING_TABLE),pl.DataFrame(_TIMING_BETA)])

    TIMING_SUMS=_TIMING_SUMS
    TIMING_IDX=_idx

    return TIMING_IDX, TIMING_SUMS


@app.cell(hide_code=True)
def _(FOLD, PAIR_IDX, PAIR_SUMS, ROOT, TIMING_IDX, TIMING_SUMS, mo, np, pl):

    import decide as _decide
    import json as _json
    _pos={int(i):j for j,i in enumerate(PAIR_IDX)}
    _where=np.array([_pos[int(i)] for i in TIMING_IDX])
    _GLOBAL_TIMING=[]
    for _alpha in (0.,.25,.5,1.):
      _sum=PAIR_SUMS.copy()
      _sum[_where]=TIMING_SUMS[_alpha]
      _row={"alpha":_alpha,"all_lam0":_decide.wp(_sum),"all_lam02":_decide.wp(_sum,.02)}
      for _fold in (0,1):
        _s=_sum[FOLD[PAIR_IDX]==_fold]
        _row[f"fold{_fold}_lam0"]=_decide.wp(_s)
        _row[f"fold{_fold}_lam02"]=_decide.wp(_s,.02)
      _GLOBAL_TIMING.append(_row)
    (ROOT/"research/artifacts/brainteaser_timing_pair_global.json").write_text(_json.dumps(_GLOBAL_TIMING,indent=2))
    mo.vstack([mo.md("## Seasonal event-timing correction: exact global pooled metric"),pl.DataFrame(_GLOBAL_TIMING)])

    return


@app.cell(hide_code=True)
def _(BOOK_RETURNS, BY_RG, ROOT, RTE, VOLS, mo, np, pl, plt):

    import json as _json
    _RMAP={int(g):ret for fold,g,ret in BOOK_RETURNS}
    _VMAP={int(g):(fold,y,p,m) for fold,g,v,y,p,m in VOLS}
    _RANK=[]
    for _g,(_fold,_y,_p,_m) in _VMAP.items():
      _w=np.abs(_y);_err=np.abs(_y-_p)
      _mae=float(np.sum((_w*_err)[_m])/max(np.sum(_w[_m]),1))
      _RANK.append((_mae,int(_g)))
    _RANK.sort()
    _SELECTED=[_RANK[len(_RANK)//2],_RANK[int(.9*(len(_RANK)-1))]]
    _fig,_ax=plt.subplots(2,3,figsize=(15,7))
    _TRACE_META=[]
    for _row,(_mae,_g) in enumerate(_SELECTED):
      _fold,_y,_p,_m=_VMAP[_g];_i=BY_RG[_g]
      _ev=np.flatnonzero(RTE[_i]==0);_ev=_ev[_ev>=1]
      _ret=_RMAP[_g]
      _yd=.5*(_y[:,0]-_y[:,1]);_pd=.5*(_p[:,0]-_p[:,1])
      _w=np.abs(_y).sum(1)
      _risk=_w*np.abs(_yd-_pd)*_m
      _center=int(np.argmax(_risk));_j=int(np.searchsorted(_ev,_center,side="right")-1)
      _a=max(0,_j-35);_b=min(len(_ev),_j+36);_s=slice(_a,_b)
      _t=np.arange(max(99,_center-256),min(20000,_center+257))
      _ax[_row,0].plot(np.arange(_a,_b),np.cumsum(_ret)[_s],label="decoded mid",lw=1)
      _ax[_row,0].axvline(_j,color="red",ls="--",alpha=.5)
      _ax[_row,0].set_title(f"rg {_g}, fold {_fold}: quote path in event time")
      _ax[_row,0].set_xlabel("book-event index");_ax[_row,0].set_ylabel("relative ticks")
      _ax[_row,1].plot(_t,_yd[_t],label="target contrast",lw=1.4)
      _ax[_row,1].plot(_t,_pd[_t],label="OOF pair contrast",lw=1.2)
      _ax[_row,1].scatter(_t[_m[_t]],_yd[_t][_m[_t]],s=6,c="red",alpha=.5,label="scored")
      _ax[_row,1].axvline(_center,color="red",ls="--",alpha=.5)
      _ax[_row,1].set_title(f"row-time peak, weighted MAE {_mae:.2f}")
      _ax[_row,1].set_xlabel("row");_ax[_row,1].set_ylabel("future movement")
      _ax[_row,2].plot(np.arange(_a,_b),_ret[_s],label="event return",lw=1)
      _ax[_row,2].plot(np.arange(_a,_b),np.diff(_ev,prepend=_ev[0])[_s],label="event gap (rows)",alpha=.7)
      _ax[_row,2].axvline(_j,color="red",ls="--",alpha=.5)
      _ax[_row,2].set_title("event returns and durations");_ax[_row,2].set_xlabel("book-event index")
      _TRACE_META.append({"row_group":_g,"fold":int(_fold),"weighted_mae":_mae,"worst_scored_row":_center,"event_index":_j})
    _ax[0,1].legend(fontsize=8);_ax[0,2].legend(fontsize=8)
    _fig.tight_layout();_fig.savefig(ROOT/"research/artifacts/brainteaser_example_traces.png",dpi=160);plt.close(_fig)
    (ROOT/"research/artifacts/brainteaser_example_traces.json").write_text(_json.dumps(_TRACE_META,indent=2))
    mo.vstack([mo.md("## Actual trajectories around scored error peaks"),pl.DataFrame(_TRACE_META)])

    return


@app.cell(hide_code=True)
def _(FOLD, MASK, P0, P1, ROOT, RTE, TIMING_IDX, Y, mo, np, pl, plt):

    import json as _json
    _RUN_TABLE=[]
    _TOTAL_ERROR=np.zeros(2)
    for _i in TIMING_IDX:
      _ev=np.flatnonzero(RTE[_i]==0);_ev=_ev[_ev>=99]
      if len(_ev)<256:continue
      _y=np.clip(Y[_i,_ev].astype(np.float64),-2,2)
      _p=np.clip(.5*(P0[_i,_ev]+P1[_i,_ev]),-2,2).astype(np.float64)
      _e=.5*(_y[:,0]-_y[:,1]-_p[:,0]+_p[:,1])
      _hi=np.abs(_e)>=np.quantile(np.abs(_e),.75)
      _st=np.flatnonzero(_hi & ~np.r_[False,_hi[:-1]])
      _en=np.flatnonzero(_hi & ~np.r_[_hi[1:],False])
      _t=np.flatnonzero(MASK[_i] & (np.arange(20000)>=99))
      _j=np.searchsorted(_ev,_t,side="right")-1
      _ok=_j>=0;_t=_t[_ok];_j=_j[_ok]
      _yr=np.clip(Y[_i,_t].astype(np.float64),-2,2)
      _pr=np.clip(.5*(P0[_i,_t]+P1[_i,_t]),-2,2).astype(np.float64)
      _er=.5*(_yr[:,0]-_yr[:,1]-_pr[:,0]+_pr[:,1])
      _w=np.abs(_yr).sum(1)
      _mass=np.bincount(_j,weights=_w*np.abs(_er),minlength=len(_ev))
      _TOTAL_ERROR[FOLD[_i]]+=_mass.sum()
      _cum=np.r_[0.,np.cumsum(_mass)]
      _end_rows=np.r_[_ev[1:],20000]
      for _s,_eidx in zip(_st,_en):
        _RUN_TABLE.append((int(FOLD[_i]),int(_eidx-_s+1),int(_end_rows[_eidx]-_ev[_s]),float(_cum[_eidx+1]-_cum[_s])))
    _RUNS=np.asarray(_RUN_TABLE)
    def _wquant(_x,_w,_q):
      _o=np.argsort(_x);_xx=_x[_o];_ww=_w[_o]
      return float(_xx[min(np.searchsorted(np.cumsum(_ww),_q*_ww.sum()),len(_xx)-1)])
    _PEAK_STATS=[]
    for _fold in (0,1):
      _a=_RUNS[_RUNS[:,0]==_fold];_length=_a[:,1];_row=_a[:,2];_mass=_a[:,3]
      _PEAK_STATS.append({"fold":_fold,"n_runs":len(_a),"n_runs_6plus":int((_length>=6).sum()),"fraction_runs_6plus":float((_length>=6).mean()),"run_events_p50_p90_p99":np.quantile(_length,[.5,.9,.99]).tolist(),
        "run_rows_p50_p90_p99":np.quantile(_row,[.5,.9,.99]).tolist(),
        "error_weighted_run_events_p50_p90":[_wquant(_length,_mass,.5),_wquant(_length,_mass,.9)],
        "error_weighted_run_rows_p50_p90":[_wquant(_row,_mass,.5),_wquant(_row,_mass,.9)],
        "high_residual_runs_share_of_total_error":float(_mass.sum()/max(_TOTAL_ERROR[_fold],1)),
        "long_runs_6plus_share_of_total_error":float(_mass[_length>=6].sum()/max(_TOTAL_ERROR[_fold],1)),
        "long_runs_6plus_share_within_high_runs":float(_mass[_length>=6].sum()/max(_mass.sum(),1))})
    (ROOT/"research/artifacts/brainteaser_peak_length_scored.json").write_text(_json.dumps(_PEAK_STATS,indent=2))
    _fig,_ax=plt.subplots(figsize=(9,4))
    _bins=[1,2,3,6,12,24,100000]
    for _fold in (0,1):
      _a=_RUNS[_RUNS[:,0]==_fold];_share=[]
      for _lo,_hi in zip(_bins[:-1],_bins[1:]):
        _share.append(float(_a[(_a[:,1]>=_lo)&(_a[:,1]<_hi),3].sum()/max(_a[:,3].sum(),1)))
      _ax.plot(range(len(_share)),_share,marker="o",label=f"fold {_fold}")
    _ax.set_xticks(range(6),["1","2","3–5","6–11","12–23","24+"])
    _ax.set_xlabel("contiguous high-residual run, book events");_ax.set_ylabel("share of scored weighted contrast error")
    _ax.legend();_ax.set_title("Where high-error peaks cost score")
    _fig.tight_layout();_fig.savefig(ROOT/"research/artifacts/brainteaser_peak_length_scored.png",dpi=160);plt.close(_fig)
    mo.vstack([mo.md("## Error-peak duration weighted by the actual scoring rows"),pl.DataFrame(_PEAK_STATS)])

    return


@app.cell(hide_code=True)
def _(BOOK_RETURNS, BY_RG, ROOT, RTE, VOLS, mo, np, pl):

    import json as _json
    _RMAP={int(g):ret.astype(np.float64) for fold,g,ret in BOOK_RETURNS}
    _CYCLE_FOLDS={0:[],1:[]}
    for _fold,_g,_vol,_y,_p,_m in VOLS:
      _ret=_RMAP[_g];_n=len(_ret)
      _u=_ret-_ret[np.maximum(np.arange(_n)-6,0)]
      _s=np.convolve(_u,np.ones(3),mode="full")[:_n]
      _ev=np.flatnonzero(RTE[BY_RG[_g]]==0);_ev=_ev[_ev>=1]
      _t=np.flatnonzero(_m & (np.arange(20000)>=99))
      _j=np.maximum(np.searchsorted(_ev,_t,side="right")-1,0)
      _ycontrast=.5*(_y[_t,0]-_y[_t,1]);_pcontrast=.5*(_p[_t,0]-_p[_t,1])
      _w=np.abs(_y[_t]).sum(1)
      _CYCLE_FOLDS[_fold].append(np.column_stack((_s[_j],_ycontrast-_pcontrast,_w,RTE[BY_RG[_g],_t])))
    _CYCLE_STATS=[]
    for _fold in (0,1):
      _a=np.concatenate(_CYCLE_FOLDS[_fold]);_s,_e,_w,_age=_a.T
      _row={"fold":_fold,"n_scored_rows":len(_a)}
      _ms=np.dot(_w,_s)/_w.sum();_me=np.dot(_w,_e)/_w.sum()
      _row["weighted_partial_residual_corr"]=float(np.dot(_w,(_s-_ms)*(_e-_me))/np.sqrt(np.dot(_w,(_s-_ms)**2)*np.dot(_w,(_e-_me)**2)))
      _row["anomaly_abs1_weight_share"]=float(_w[np.abs(_s)>=1].sum()/_w.sum())
      _row["anomaly_abs1_error_share"]=float(np.dot(_w[np.abs(_s)>=1],np.abs(_e[np.abs(_s)>=1]))/np.dot(_w,np.abs(_e)))
      _row["anomaly_abs1_signed_residual_corr"]=float(np.corrcoef(_s[np.abs(_s)>=1],_e[np.abs(_s)>=1])[0,1])
      _bins=[]
      for _lo,_hi in ((0,.5),(.5,1),(1,2),(2,4),(4,100)):
        _b=(np.abs(_s)>=_lo)&(np.abs(_s)<_hi)
        _bins.append({"range":f"{_lo}–{_hi}","rows":int(_b.sum()),"weighted_mae":float(np.dot(_w[_b],np.abs(_e[_b]))/max(_w[_b].sum(),1))})
      _row["bins"]=_bins
      _CYCLE_STATS.append(_row)
    (ROOT/"research/artifacts/brainteaser_cycle_break_signal.json").write_text(_json.dumps(_CYCLE_STATS,indent=2))
    mo.vstack([mo.md("## Is a broken six-event cycle a causal warning signal?"),pl.DataFrame([{k:v for k,v in r.items() if k!="bins"} for r in _CYCLE_STATS]),pl.DataFrame([{"fold":r["fold"],**b} for r in _CYCLE_STATS for b in r["bins"]])])

    return


@app.cell(hide_code=True)
def _(ARMA_ENDS, ARMA_ERROR, ARMA_FITS, ARMA_FLAT, ROOT, mo, np, pl, plt):

    import numba as _nb
    import json as _json
    @_nb.njit
    def _garch11(_eps,_ends,_v0,_alpha,_beta,_keep_trace=False):
      _sum=0.;_trace=np.empty(len(_eps))
      _start=0;_omega=max(1e-8,(1-_alpha-_beta)*_v0)
      for _end in _ends:
        _h=_v0
        for _j in range(_start,_end):
          _v=max(_h,1e-8)
          _trace[_j]=_v
          _sum+=.5*(np.log(_v)+_eps[_j]*_eps[_j]/_v)
          _h=_omega+_alpha*_eps[_j]*_eps[_j]+_beta*_v
        _start=_end
      return _sum/len(_eps),_trace
    _GARCH_GRID=[]
    for _a in (.03,.06,.10,.16,.25):
      for _b in (.65,.75,.85,.90,.94):
        if _a+_b<.99:_GARCH_GRID.append((_a,_b))
    _GARCH_TABLE=[];_GARCH_EXAMPLE=None
    for _fit in (0,1):
      _train_eps=ARMA_ERROR(ARMA_FLAT[_fit],ARMA_ENDS[_fit],*ARMA_FITS[_fit])
      _v0=max(np.mean(_train_eps**2),1e-6)
      _fit_scores=[(_garch11(_train_eps,ARMA_ENDS[_fit],_v0,_a,_b)[0],_a,_b) for _a,_b in _GARCH_GRID]
      _best=min(_fit_scores)
      _eval=1-_fit
      _eps=ARMA_ERROR(ARMA_FLAT[_eval],ARMA_ENDS[_eval],*ARMA_FITS[_fit])
      _nll,_var=_garch11(_eps,ARMA_ENDS[_eval],_v0,_best[1],_best[2])
      _const=float(.5*(np.log(_v0)+np.mean(_eps**2)/_v0))
      _ewma=float(_garch11(_eps,ARMA_ENDS[_eval],_v0,.06,.94)[0])
      _std_sq=[];_start=0
      for _end in ARMA_ENDS[_eval]:
        _z=_eps[_start:_end]/np.sqrt(_var[_start:_end])
        if len(_z)>32:_std_sq.append(float(np.corrcoef((_z[1:]**2),(_z[:-1]**2))[0,1]))
        _start=_end
      _GARCH_TABLE.append({"eval_fold":_eval,"train_alpha":_best[1],"train_beta":_best[2],
        "heldout_garch_nll":float(_nll),"constant_nll":_const,"ewma94_nll":_ewma,
        "standardized_squared_acf_lag1":float(np.nanmean(_std_sq))})
      if _eval==0:_GARCH_EXAMPLE=(_eps[:800],_var[:800])
    (ROOT/"research/artifacts/brainteaser_garch11_residual.json").write_text(_json.dumps(_GARCH_TABLE,indent=2))
    _fig,_ax=plt.subplots(figsize=(11,3))
    _ax.plot(_GARCH_EXAMPLE[0]**2,label="ARMA innovation squared",alpha=.45,lw=.7)
    _ax.plot(_GARCH_EXAMPLE[1],label="GARCH(1,1) conditional variance",lw=1.5)
    _ax.set_xlabel("book event");_ax.set_title("Held-out model-error variance clustering (diagnostic only)")
    _ax.legend();_fig.tight_layout();_fig.savefig(ROOT/"research/artifacts/brainteaser_garch11_residual.png",dpi=160);plt.close(_fig)
    mo.vstack([mo.md("## GARCH(1,1) on ARMA error innovations"),pl.DataFrame(_GARCH_TABLE)])

    return


@app.cell(hide_code=True)
def _(BY_RG, ROOT, RTE, SequenceFile, VOLS, mo, np, pl, plt):

    import pickle as _pickle
    import json as _json
    from wnn.ticks import reconstruct as _reconstruct
    with (ROOT/"cache/class3_moves_full.pkl").open("rb") as _file:
      _moves,_spreads=_pickle.load(_file)
    _source=SequenceFile(ROOT/"datasets/valid.parquet")
    _LAGS=[1,2,4,6,12,24,48,96,192]
    _DVAR=[];_SPREAD_ACF=[]
    for _g in list(VOLS[j][1] for j in range(0,min(len(VOLS),80),4)):
      _x=_source.read(int(_g),targets=False,mask=False).features
      _ask,_bid,_unknown=_reconstruct(_x,_moves,_spreads)
      _i=BY_RG[int(_g)];_ev=np.flatnonzero(RTE[_i]==0);_ev=_ev[_ev>=1]
      _mid=.5*(_ask[_ev]+_bid[_ev]);_spread=(_ask[_ev]-_bid[_ev]).astype(float)
      _DVAR.append([[float(np.var(_mid[lag:]-_mid[:-lag])),float(np.var(_spread[lag:]-_spread[:-lag]))] for lag in _LAGS])
      _SPREAD_ACF.append(float(np.corrcoef(_spread[6:],_spread[:-6])[0,1]))
    _DVAR=np.mean(_DVAR,axis=0)
    _VECM_REPORT={"n_sequences":len(_SPREAD_ACF),"event_lags":_LAGS,
     "mid_displacement_variance":_DVAR[:,0].tolist(),"spread_displacement_variance":_DVAR[:,1].tolist(),
     "spread_level_acf_lag6":float(np.nanmean(_SPREAD_ACF))}
    (ROOT/"research/artifacts/brainteaser_vecm_diagnostic.json").write_text(_json.dumps(_VECM_REPORT,indent=2))
    _fig,_ax=plt.subplots(figsize=(8,4))
    _ax.plot(_LAGS,_DVAR[:,0],marker="o",label="common mid")
    _ax.plot(_LAGS,_DVAR[:,1],marker="o",label="ask-bid spread")
    _ax.set_xscale("log",base=2);_ax.set_yscale("log");_ax.set_xlabel("book-event lag")
    _ax.set_ylabel("displacement variance, tick²");_ax.legend()
    _ax.set_title("Two book prices: common movement and bounded spread")
    _fig.tight_layout();_fig.savefig(ROOT/"research/artifacts/brainteaser_vecm_diagnostic.png",dpi=160);plt.close(_fig)
    mo.vstack([mo.md("## VECM question: do two prices add two trends?"),pl.DataFrame([_VECM_REPORT])])

    return


@app.cell(hide_code=True)
def _(DDATA, ROOT, mo, np, pl, plt):

    import json as _json
    _CONF=[]
    for _fold in (0,1):
      _d=np.concatenate([d for f,d,p,w,a in DDATA if f==_fold])
      _p=np.concatenate([p for f,d,p,w,a in DDATA if f==_fold])
      _w=np.concatenate([w for f,d,p,w,a in DDATA if f==_fold])
      _err=np.abs(_d-_p);_wrong=(_d*_p)<0
      for _lo,_hi in ((0,.1),(.1,.2),(.2,.4),(.4,.8),(.8,2.1)):
        _s=(np.abs(_p)>=_lo)&(np.abs(_p)<_hi)
        _CONF.append({"fold":_fold,"predicted_abs_contrast":f"{_lo}–{_hi}",
          "row_share":float(_s.mean()),"weight_share":float(_w[_s].sum()/_w.sum()),
          "error_share":float(np.dot(_w[_s],_err[_s])/np.dot(_w,_err)),
          "weighted_direction_accuracy":float(np.dot(_w[_s],(~_wrong[_s]))/max(_w[_s].sum(),1)),
          "wrong_side_error_share_within_bin":float(np.dot(_w[_s&_wrong],_err[_s&_wrong])/max(np.dot(_w[_s],_err[_s]),1))})
    (ROOT/"research/artifacts/brainteaser_confidence_error.json").write_text(_json.dumps(_CONF,indent=2))
    _fig,_ax=plt.subplots(1,2,figsize=(11,4))
    for _fold in (0,1):
      _rows=[r for r in _CONF if r["fold"]==_fold]
      _ax[0].plot(range(5),[r["error_share"] for r in _rows],marker="o",label=f"fold {_fold}")
      _ax[1].plot(range(5),[r["weighted_direction_accuracy"] for r in _rows],marker="o",label=f"fold {_fold}")
    for _a in _ax:
      _a.set_xticks(range(5),["<.1",".1–.2",".2–.4",".4–.8",".8+"])
      _a.set_xlabel("absolute predicted future movement");_a.legend()
    _ax[0].set_title("Share of scored weighted contrast error");_ax[1].set_title("Weighted direction accuracy")
    _fig.tight_layout();_fig.savefig(ROOT/"research/artifacts/brainteaser_confidence_error.png",dpi=160);plt.close(_fig)
    mo.vstack([mo.md("## Are expensive mistakes uncertain or confident?"),pl.DataFrame(_CONF)])

    return


@app.cell(hide_code=True)
def _(
    CLS,
    FOLD,
    MASK,
    P0,
    P1,
    RG,
    ROOT,
    RTE,
    SequenceFile,
    Y,
    mo,
    np,
    pl,
    plt,
):
    import json as _json
    _cf = np.load(ROOT/"cache/chain_folds.npz")
    _train_chains = _cf["chain"][_cf["split"]=="train"]
    _valid = _cf["split"]=="valid"
    _cmap={int(g):int(c) for g,c in zip(_cf["row_group"][_valid],_cf["chain"][_valid])}
    _pure=np.array([_cmap[int(g)] not in set(_train_chains) for g in RG])
    _sel=np.flatnonzero((CLS==3)&np.isin(FOLD,[0,1])&_pure)
    _source=SequenceFile(ROOT/"datasets/valid.parquet")
    _store={0:[],1:[]}
    for _i in _sel:
        _ev=np.flatnonzero(RTE[_i]==0)
        _x=_source.read(int(RG[_i]),targets=False,mask=False).features
        _code=_x[_ev,:22]
        _fail=np.zeros(len(_ev),np.int8)
        _fail[6:]=np.any(_code[6:]!=_code[:-6],axis=1)
        _roll=np.convolve(_fail,np.ones(6,np.int8),"full")[:len(_ev)]
        _prior=np.r_[0,_roll[:-1]]
        _rows=np.flatnonzero(MASK[_i]&(np.arange(Y.shape[1])>=99))
        _event_id=np.searchsorted(_ev,_rows,side="right")-1
        _ok=_event_id>=12
        _rows=_rows[_ok];_event_id=_event_id[_ok]
        if not len(_rows):continue
        _target=np.clip(Y[_i,_rows],-2,2)
        _pred=.5*(P0[_i,_rows]+P1[_i,_rows])
        _truth=.5*(_target[:,0]-_target[:,1])
        _p=.5*(_pred[:,0]-_pred[:,1])
        _weight=np.abs(_target).sum(axis=1)
        _store[int(FOLD[_i])].append(((_roll[_event_id]).astype(np.float32),(_prior[_event_id]).astype(np.float32),
            np.abs(_truth-_p).astype(np.float32),_weight.astype(np.float32),np.abs(_p).astype(np.float32),RTE[_i,_rows].astype(np.float32)))
    _CYCLE_BREAK_STATS=[]
    _CYCLE_BINS=[]
    for _fold in (0,1):
        _a=[np.concatenate([r[j] for r in _store[_fold]]) for j in range(6)]
        _count,_prior_count,_error,_w,_pmag,_age=_a
        _valid_w=_w>0
        _controls=np.stack([np.ones_like(_pmag),np.log1p(_age),_pmag,_pmag**2],axis=1)
        _sw=np.sqrt(_w[_valid_w])
        _wc=_controls[_valid_w]*_sw[:,None]
        _er=_error[_valid_w]
        _er_res=_er-_controls[_valid_w]@np.linalg.lstsq(_wc,_er*_sw,rcond=None)[0]
        _stat={"fold":_fold,"sequences":len(_store[_fold]),"weighted_rows":float(_w.sum())}
        for _name,_signal in (("current",_count),("prior",_prior_count)):
            _ss=_signal[_valid_w]
            _res=_ss-_controls[_valid_w]@np.linalg.lstsq(_wc,_ss*_sw,rcond=None)[0]
            _center=lambda v: v-np.average(v,weights=_w[_valid_w])
            _part=float(np.sum(_w[_valid_w]*_res*_er_res)/np.sqrt(np.sum(_w[_valid_w]*_res**2)*np.sum(_w[_valid_w]*_er_res**2)))
            _stat[_name+"_partial_corr_abs_error"]=_part
            for _n in range(7):
                _hit=(_signal==_n)&_valid_w
                _CYCLE_BINS.append({"fold":_fold,"signal":_name,"mismatches":_n,
                    "weight_share":float(_w[_hit].sum()/_w[_valid_w].sum()),
                    "weighted_abs_error":float(np.average(_error[_hit],weights=_w[_hit])) if _hit.any() else None})
        _CYCLE_BREAK_STATS.append(_stat)
    _BREAK_REPORT={"summary":_CYCLE_BREAK_STATS,"bins":_CYCLE_BINS}
    (ROOT/"research/artifacts/brainteaser_raw_cycle_break.json").write_text(_json.dumps(_BREAK_REPORT,indent=2))
    _fig,_ax=plt.subplots(1,2,figsize=(10,4))
    for _j,_name in enumerate(("current","prior")):
        for _fold in (0,1):
            _rr=[r for r in _CYCLE_BINS if r["fold"]==_fold and r["signal"]==_name]
            _ax[_j].plot([r["mismatches"] for r in _rr],[r["weighted_abs_error"] for r in _rr],marker="o",label=f"fold {_fold}")
        _ax[_j].set_title(f"{_name} six-event cycle breaks")
        _ax[_j].set_xlabel("raw-code mismatches in previous 6 events")
        _ax[_j].set_ylabel("weighted |contrast residual|")
        _ax[_j].legend()
    _fig.tight_layout();_fig.savefig(ROOT/"research/artifacts/brainteaser_raw_cycle_break.png",dpi=160);plt.close(_fig)
    mo.vstack([mo.md("## Can raw-code cycle breaks warn of model error?"),pl.DataFrame(_CYCLE_BREAK_STATS)])

    return


@app.cell(hide_code=True)
def _(CLS, FOLD, MASK, P0, P1, RG, ROOT, Y, mo, np, pl, plt):
    import json as _json
    from wnn.gmetric import wp_from_sums as _wp_sums
    _cf=np.load(ROOT/"cache/chain_folds.npz")
    _v=_cf["split"]=="valid"
    _cm={int(g):int(c) for g,c in zip(_cf["row_group"][_v],_cf["chain"][_v])}
    _tr=set(map(int,_cf["chain"][_cf["split"]=="train"]))
    _idx=np.flatnonzero(np.isin(FOLD,[0,1]) & ~np.isin(np.array([_cm[int(g)] for g in RG]),list(_tr)))
    _modes=("pair","sign","magnitude","contrast")
    _mom=np.zeros((2,4,len(_modes),2,6),np.float64)
    _rows=np.zeros((2,4),np.int64)
    _mass=np.zeros((2,4,2),np.float64)
    for _i in _idx:
        _s=MASK[_i].copy();_s[:99]=False
        if not _s.any():continue
        _y=np.clip(Y[_i,_s].astype(np.float64),-2,2)
        _p=np.clip(.5*(P0[_i,_s]+P1[_i,_s]),-2,2).astype(np.float64)
        _com=.5*(_p[:,0]+_p[:,1]);_tcon=.5*(_y[:,0]-_y[:,1])
        _variants=(_p,np.sign(_y)*np.abs(_p),np.sign(_p)*np.abs(_y),np.column_stack((_com+_tcon,_com-_tcon)))
        _f=int(FOLD[_i]);_c=int(CLS[_i]);_rows[_f,_c]+=len(_y)
        for _target in (0,1):
            _yy=_y[:,_target];_w=np.abs(_yy);_mass[_f,_c,_target]+=_w.sum()
            for _mode,_pred in enumerate(_variants):
                _pp=np.clip(_pred[:,_target],-2,2)
                _mom[_f,_c,_mode,_target]+=(_w.sum(),np.dot(_w,_yy),np.dot(_w,_pp),
                    np.dot(_w,_yy*_yy),np.dot(_w,_pp*_pp),np.dot(_w,_yy*_pp))
    def _wp(_sum):
        return float(np.mean([_wp_sums(_sum[k]) for k in (0,1)]))
    _baseline=_wp(_mom[:,:,0].sum((0,1)))
    _head=[]
    for _f in (0,1):
        _base_f=_mom[_f,:,0].sum(0)
        for _c in range(4):
            _item={"fold":_f,"class":_c,"scored_rows":int(_rows[_f,_c]),
                "target_weight_share":float(_mass[_f,_c].sum()/_mass[_f].sum()),
                "class_wp":_wp(_mom[_f,_c,0]),"fold_wp":_wp(_base_f)}
            for _name in ("sign","magnitude","contrast"):
                _j=_modes.index(_name)
                _part=_base_f-_mom[_f,_c,0]+_mom[_f,_c,_j]
                _item[_name+"_global_wp_gain"]=_wp(_part)-_wp(_base_f)
            _head.append(_item)
    _HEAD_REPORT={"pure_sequences":len(_idx),"pooled_pair_wp":_baseline,"by_fold_class":_head}
    (ROOT/"research/artifacts/brainteaser_class_oracle_allocation.json").write_text(_json.dumps(_HEAD_REPORT,indent=2))
    _fig,_ax=plt.subplots(1,3,figsize=(12,4))
    for _j,_name in enumerate(("sign","magnitude","contrast")):
        for _fold in (0,1):
            _r=[r for r in _head if r["fold"]==_fold]
            _ax[_j].plot(range(4),[r[_name+"_global_wp_gain"] for r in _r],marker="o",label=f"fold {_fold}")
        _ax[_j].set_xticks(range(4));_ax[_j].set_xlabel("market class")
        _ax[_j].set_ylabel("global WP gain (oracle)")
        _ax[_j].set_title(_name)
        _ax[_j].legend()
    _fig.tight_layout();_fig.savefig(ROOT/"research/artifacts/brainteaser_class_oracle_allocation.png",dpi=160);plt.close(_fig)
    mo.vstack([mo.md("## Which class offers global score leverage? Noncausal oracles"),pl.DataFrame(_head)])

    return


@app.cell(hide_code=True)
def _(CLS, FOLD, MASK, P0, P1, RG, ROOT, RTE, Y, mo, np, pl, plt):
    import json as _json
    _cf=np.load(ROOT/"cache/chain_folds.npz")
    _v=_cf["split"]=="valid"
    _cm={int(g):int(c) for g,c in zip(_cf["row_group"][_v],_cf["chain"][_v])}
    _tr=set(map(int,_cf["chain"][_cf["split"]=="train"]))
    _sel=np.flatnonzero((CLS==3)&np.isin(FOLD,[0,1])&~np.isin(np.array([_cm[int(g)] for g in RG]),list(_tr)))
    _lags=np.arange(-8,9)
    _mom=np.zeros((2,len(_lags),6),np.float64)
    _n=np.zeros(2,np.int64)
    for _i in _sel:
        _ev=np.flatnonzero(RTE[_i]==0)
        if len(_ev)<40:continue
        _j=np.arange(8,len(_ev)-8)
        _j=_j[_ev[_j]>=99]
        if not len(_j):continue
        _yc=np.clip(Y[_i],-2,2)
        _mass=(np.abs(_yc).sum(1)*MASK[_i]).astype(np.float64)
        _cum=np.r_[0.,np.cumsum(_mass)]
        _start=_ev[_j];_end=_ev[_j+1]
        _w=_cum[_end]-_cum[_start]
        _keep=_w>0;_j=_j[_keep];_w=_w[_keep]
        if not len(_j):continue
        _truth=.5*(_yc[_ev[_j],0]-_yc[_ev[_j],1])
        _f=int(FOLD[_i]);_n[_f]+=len(_j)
        for _k,_lag in enumerate(_lags):
            _pos=_ev[_j+_lag]
            _pred=.5*(P0[_i,_pos]+P1[_i,_pos])
            _p=.5*(_pred[:,0]-_pred[:,1])
            _m=_mom[_f,_k]
            _m+=(np.sum(_w),np.dot(_w,_truth),np.dot(_w,_p),np.dot(_w,_truth**2),np.dot(_w,_p**2),np.dot(_w,_truth*_p))
    def _wcorr(_m):
        _w,_y,_p,_yy,_pp,_yp=_m
        return float((_yp-_y*_p/_w)/np.sqrt((_yy-_y*_y/_w)*(_pp-_p*_p/_w)))
    _LAG_REPORT={"sequences":len(_sel),"episode_counts":_n.tolist(),
        "lag_definition":"positive: future model output at current target's later book event (noncausal)",
        "folds":[{"fold":f,"correlation":[_wcorr(m) for m in _mom[f]],
                  "best_lag":int(_lags[np.argmax([_wcorr(m) for m in _mom[f]])])} for f in (0,1)]}
    (ROOT/"research/artifacts/brainteaser_event_prediction_lag.json").write_text(_json.dumps(_LAG_REPORT,indent=2))
    _fig,_ax=plt.subplots(figsize=(8,4))
    for _item in _LAG_REPORT["folds"]:
        _ax.plot(_lags,_item["correlation"],marker="o",label=f"fold {_item['fold']}")
    _ax.axvline(0,color="gray",ls="--")
    _ax.set_xlabel("prediction event shift (+ uses future model output)")
    _ax.set_ylabel("episode-weighted target/pair contrast correlation")
    _ax.set_title("Does the model lag class-3 target changes?")
    _ax.legend();_fig.tight_layout()
    _fig.savefig(ROOT/"research/artifacts/brainteaser_event_prediction_lag.png",dpi=160);plt.close(_fig)
    mo.vstack([mo.md("## Event-clock forecast lag diagnostic"),pl.DataFrame(_LAG_REPORT["folds"])])

    return


@app.cell(hide_code=True)
def _(CLS, FOLD, MASK, P0, P1, RG, ROOT, RTE, Y, mo, np, pl, plt):
    import json as _json
    from wnn.gmetric import wp_from_sums as _wp_sums
    _cf=np.load(ROOT/"cache/chain_folds.npz")
    _v=_cf["split"]=="valid";_cm={int(g):int(c) for g,c in zip(_cf["row_group"][_v],_cf["chain"][_v])}
    _tr=set(map(int,_cf["chain"][_cf["split"]=="train"]))
    _sel=np.flatnonzero((CLS==3)&np.isin(FOLD,[0,1])&~np.isin(np.array([_cm[int(g)] for g in RG]),list(_tr)))
    _names=("actual_row","entry","previous_event","next_event")
    _mom=np.zeros((2,len(_names),2,6),np.float64)
    _ct=np.zeros((2,len(_names),6),np.float64)
    _agebin=np.array([0,1,4,16,64,256,20001])
    _age_mom=np.zeros((2,len(_names),len(_agebin)-1,6),np.float64)
    _n=np.zeros(2,np.int64)
    for _i in _sel:
        _ev=np.flatnonzero(RTE[_i]==0)
        if len(_ev)<3:continue
        _rows=np.flatnonzero(MASK[_i]&(np.arange(Y.shape[1])>=99))
        _j=np.searchsorted(_ev,_rows,side="right")-1
        _good=(_j>=1)&(_j<len(_ev)-1)
        _rows=_rows[_good];_j=_j[_good]
        if not len(_rows):continue
        _yy=np.clip(Y[_i,_rows].astype(np.float64),-2,2)
        _target=.5*(_yy[:,0]-_yy[:,1])
        _w=np.abs(_yy).sum(axis=1)
        _age=RTE[_i,_rows]
        _positions=(_rows,_ev[_j],_ev[_j-1],_ev[_j+1])
        _f=int(FOLD[_i]);_n[_f]+=len(_rows)
        for _k,_pos in enumerate(_positions):
            _p=.5*(P0[_i,_pos]+P1[_i,_pos])
            _p=np.clip(_p.astype(np.float64),-2,2)
            for _t in (0,1):
                _y=_yy[:,_t];_q=_p[:,_t];_ww=np.abs(_y)
                _mom[_f,_k,_t]+=(_ww.sum(),np.dot(_ww,_y),np.dot(_ww,_q),np.dot(_ww,_y*_y),np.dot(_ww,_q*_q),np.dot(_ww,_y*_q))
            _q=.5*(_p[:,0]-_p[:,1])
            def _sums(_m):
                _wv=_w[_m];_yv=_target[_m];_pv=_q[_m]
                return np.array((_wv.sum(),np.dot(_wv,_yv),np.dot(_wv,_pv),
                    np.dot(_wv,_yv*_yv),np.dot(_wv,_pv*_pv),np.dot(_wv,_yv*_pv)))
            _ct[_f,_k]+=_sums(np.ones(len(_w),bool))
            for _b in range(len(_agebin)-1):
                _hit=(_age>=_agebin[_b])&(_age<_agebin[_b+1])
                if _hit.any():_age_mom[_f,_k,_b]+=_sums(_hit)
    def _corr(_m):
        _wt,_y,_p,_yy,_pp,_yp=_m
        return float((_yp-_y*_p/_wt)/np.sqrt((_yy-_y*_y/_wt)*(_pp-_p*_p/_wt)))
    _REAL_REPORT=[]
    for _f in (0,1):
        for _k,_name in enumerate(_names):
            _REAL_REPORT.append({"fold":_f,"source":_name,"rows":int(_n[_f]),
                "class_wp":float(np.mean([_wp_sums(_mom[_f,_k,_t]) for _t in (0,1)])),
                "contrast_corr":_corr(_ct[_f,_k]),
                "age_corr":[_corr(_age_mom[_f,_k,b]) if _age_mom[_f,_k,b,0]>0 else None for b in range(len(_agebin)-1)]})
    _REAL_DATA={"summary":_REAL_REPORT,"age_bins":_agebin.tolist(),
        "caveat":"next_event uses future inputs and is only a noncausal diagnostic"}
    (ROOT/"research/artifacts/brainteaser_scored_next_event_oracle.json").write_text(_json.dumps(_REAL_DATA,indent=2))
    _fig,_ax=plt.subplots(figsize=(9,4))
    for _f in (0,1):
        for _name in ("actual_row","next_event"):
            _r=next(x for x in _REAL_REPORT if x["fold"]==_f and x["source"]==_name)
            _ax.plot(range(len(_agebin)-1),_r["age_corr"],marker="o",label=f"fold {_f}, {_name}")
    _ax.set_xticks(range(len(_agebin)-1),["0","1–3","4–15","16–63","64–255","256+"])
    _ax.set_xlabel("rows since last book event");_ax.set_ylabel("weighted target/contrast correlation")
    _ax.set_title("Future-event model state as a noncausal information oracle")
    _ax.legend();_fig.tight_layout()
    _fig.savefig(ROOT/"research/artifacts/brainteaser_scored_next_event_oracle.png",dpi=160);plt.close(_fig)
    mo.vstack([mo.md("## Is the forecast gap an information gap on scored rows?"),pl.DataFrame(_REAL_REPORT)])

    return


@app.cell(hide_code=True)
def _(EDATA, FUT, mo, np, pl):
    _jump=FUT[:,0]-.5*(EDATA["ask"].to_numpy()+EDATA["bid"].to_numpy())
    _w=EDATA["weight"].to_numpy()
    _frac=[]
    for _f in (0,1):
        _s=EDATA["fold"].to_numpy()==_f
        _frac.append({"fold":_f,"episodes":int(_s.sum()),
            "up":float(np.mean(_jump[_s]>0)),"down":float(np.mean(_jump[_s]<0)),
            "zero":float(np.mean(_jump[_s]==0)),
            "weighted_nonzero":float(np.dot(_w[_s],_jump[_s]!=0)/_w[_s].sum())})
    mo.vstack([mo.md("## Next observable price event: directional label balance"),pl.DataFrame(_frac)])

    return


@app.cell(hide_code=True)
def _(EDATA, EHIST, EINPUT, FUT, ROOT, mo, np, pl, plt):
    import json as _json
    from sklearn.ensemble import HistGradientBoostingClassifier as _HGB
    from sklearn.metrics import roc_auc_score as _auc
    _fold=EDATA["fold"].to_numpy()
    _weight=EDATA["weight"].to_numpy().astype(np.float64)
    _jump=FUT[:,0]-.5*(EDATA["ask"].to_numpy()+EDATA["bid"].to_numpy())
    _valid_jump=_jump!=0
    _yjump=(_jump>0).astype(np.int8)
    _x=np.column_stack((EINPUT,EHIST)).astype(np.float32)
    _pair=.5*(EINPUT[:,0]-EINPUT[:,1])
    _prob=np.full(len(_fold),np.nan,np.float32)
    _HGB_REPORT=[]
    for _fit,_eval in ((0,1),(1,0)):
        _tr=(_fold==_fit)&_valid_jump
        _va=(_fold==_eval)&_valid_jump
        _model=_HGB(max_iter=80,max_leaf_nodes=15,learning_rate=.05,min_samples_leaf=100,
            l2_regularization=1.,random_state=41)
        _model.fit(_x[_tr],_yjump[_tr],sample_weight=np.minimum(_weight[_tr],100.))
        _prob[_fold==_eval]=_model.predict_proba(_x[_fold==_eval])[:,1]
        _sub=_prob[_va]
        _HGB_REPORT.append({"train_fold":_fit,"eval_fold":_eval,
            "n_eval_events":int(_va.sum()),
            "weighted_next_jump_auc_pair":float(_auc(_yjump[_va],_pair[_va],sample_weight=_weight[_va])),
            "weighted_next_jump_auc_hgb":float(_auc(_yjump[_va],_sub,sample_weight=_weight[_va])),
            "next_jump_weighted_brier":float(np.average((_sub-_yjump[_va])**2,weights=_weight[_va]))})
    np.savez_compressed(ROOT/"research/artifacts/brainteaser_next_event_hgb_oof.npz",
        fold=_fold,seq=EDATA["seq"].to_numpy(),prob=_prob,pair=_pair,weight=_weight,
        next_jump=_jump)
    _HGB_RESULT={"folds":_HGB_REPORT,
        "input":"first-scored causal pair outputs, raw features, age and 1/2/4/8/16/32/64-event decoded history",
        "train_weight":"min(scored episode target mass,100)",
        "limit":"128 sampled pure class-3 sequences; validation folds used as opposite-fold probe"}
    (ROOT/"research/artifacts/brainteaser_next_event_hgb.json").write_text(_json.dumps(_HGB_RESULT,indent=2))
    _fig,_ax=plt.subplots(figsize=(6,4))
    _xp=np.arange(2)
    _ax.bar(_xp-.18,[r["weighted_next_jump_auc_pair"] for r in _HGB_REPORT],width=.36,label="incumbent pair")
    _ax.bar(_xp+.18,[r["weighted_next_jump_auc_hgb"] for r in _HGB_REPORT],width=.36,label="causal snapshot + history HGB")
    _ax.set_xticks(_xp,["fold 1","fold 0"]);_ax.set_ylim(.5,1)
    _ax.set_ylabel("weighted AUC for next quote move");_ax.legend()
    _fig.tight_layout();_fig.savefig(ROOT/"research/artifacts/brainteaser_next_event_hgb.png",dpi=160);plt.close(_fig)
    mo.vstack([mo.md("## Is the next quote move predictable from causal snapshots?"),pl.DataFrame(_HGB_REPORT)])

    return


@app.cell(hide_code=True)
def _(EDATA, ROOT, mo, np, pl):
    import json as _json
    from sklearn.metrics import roc_auc_score as _auc
    _saved=np.load(ROOT/"research/artifacts/brainteaser_next_event_hgb_oof.npz")
    _mid=.5*(EDATA["ask"].to_numpy()+EDATA["bid"].to_numpy())
    _target_delta=EDATA["F"].to_numpy()-_mid
    _jump=_saved["next_jump"];_pair=_saved["pair"];_prob=_saved["prob"]
    _fold=_saved["fold"];_w=_saved["weight"]
    _MATCH=[]
    for _f in (0,1):
        _s=(_fold==_f)&(_target_delta!=0)&(_jump!=0)
        _label=(_target_delta[_s]>0).astype(np.int8)
        _MATCH.append({"fold":_f,"events":int(_s.sum()),
            "weighted_next_jump_target_side_agreement":float(np.average((_jump[_s]>0)==_label,weights=_w[_s])),
            "target_side_auc_pair":float(_auc(_label,_pair[_s],sample_weight=_w[_s])),
            "target_side_auc_next_jump_hgb":float(_auc(_label,_prob[_s],sample_weight=_w[_s]))})
    (ROOT/"research/artifacts/brainteaser_next_event_target_link.json").write_text(_json.dumps(_MATCH,indent=2))
    mo.vstack([mo.md("## Does next-event side supervision track the actual target?"),pl.DataFrame(_MATCH)])

    return


@app.cell(hide_code=True)
def _(CLS, FOLD, MASK, RG, ROOT, Y, mo, np, pl):
    import json as _json
    _q=np.load(ROOT/"cache/quote_path_valid.npy",mmap_mode="r")
    _cf=np.load(ROOT/"cache/chain_folds.npz")
    _v=_cf["split"]=="valid";_cm={int(g):int(c) for g,c in zip(_cf["row_group"][_v],_cf["chain"][_v])}
    _tr=set(map(int,_cf["chain"][_cf["split"]=="train"]))
    _sel=np.flatnonzero((CLS==3)&np.isin(FOLD,[0,1])&~np.isin(np.array([_cm[int(g)] for g in RG]),list(_tr)))
    _agg=np.zeros((2,2,8),np.float64)
    for _i in _sel:
        _mask=MASK[_i]&(np.arange(Y.shape[1])>=99)&np.isfinite(_q[_i,:,1])
        if not _mask.any():continue
        _yy=np.clip(Y[_i,_mask].astype(np.float64),-2,2)
        _target=.5*(_yy[:,0]-_yy[:,1]);_w=np.abs(_yy).sum(1)
        _f=int(FOLD[_i])
        for _h in (0,1):
            _pred=_q[_i,_mask,_h].astype(np.float64)
            _a=_agg[_f,_h]
            _a+=(len(_w),_w.sum(),np.dot(_w,_target),np.dot(_w,_pred),
                 np.dot(_w,_target**2),np.dot(_w,_pred**2),np.dot(_w,_target*_pred),
                 np.dot(_w,np.abs(_target-_pred)))
    _QUOTE_REPORT=[]
    for _f in (0,1):
        for _h,_label in enumerate((8,24)):
            _n,_w,_y,_p,_yy,_pp,_yp,_ae=_agg[_f,_h]
            _corr=(_yp-_y*_p/_w)/np.sqrt((_yy-_y*_y/_w)*(_pp-_p*_p/_w))
            _QUOTE_REPORT.append({"fold":_f,"horizon_events":_label,"scored_rows":int(_n),
                "weighted_target_contrast_corr":float(_corr),
                "weighted_abs_error_target_units":float(_ae/_w)})
    (ROOT/"research/artifacts/brainteaser_quote_path_label_quality.json").write_text(_json.dumps(_QUOTE_REPORT,indent=2))
    mo.vstack([mo.md("## Noncausal quote-path auxiliary label fidelity"),pl.DataFrame(_QUOTE_REPORT)])

    return


if __name__ == "__main__":
    app.run()
