import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import polars as pl
    import altair as alt

    ROOT = mo.notebook_dir().parent
    MODELS = ["base", "ab9", "ab16", "ab18"]
    scored = (
        pl.read_parquet(ROOT / "cache" / "valid_scored.parquet")
        .with_columns(
            ay0=pl.col("y0").clip(-2, 2).abs(), ay1=pl.col("y1").clip(-2, 2).abs(),
            log_since_book=(pl.col("since_book") + 1).log10(), log_since_trade=(pl.col("since_trade") + 1).log10(),
            log_since_i1=(pl.col("since_i1") + 1).log10(), log_n_book=(pl.col("n_book") + 1).log10(),
        )
    )
    clean = scored.filter(pl.col("split") != "A")


    def wp_expr(model: str) -> list[pl.Expr]:
        """Polars aggregations giving the six WP sums per target for one model's predictions."""
        out = []
        for k in (0, 1):
            y = pl.col(f"y{k}").clip(-2, 2)
            p = pl.col(f"{model}{k}").clip(-2, 2)
            w = y.abs()
            out += [w.sum().alias(f"{model}_W{k}"), (w * y).sum().alias(f"{model}_WY{k}"), (w * p).sum().alias(f"{model}_WP{k}"),
                    (w * y * y).sum().alias(f"{model}_WYY{k}"), (w * p * p).sum().alias(f"{model}_WPP{k}"), (w * y * p).sum().alias(f"{model}_WYP{k}")]
        return out


    def wp_from_cols(df: pl.DataFrame, model: str) -> np.ndarray:
        res = []
        for k in (0, 1):
            W = df[f"{model}_W{k}"].to_numpy()
            my, mp = df[f"{model}_WY{k}"].to_numpy() / W, df[f"{model}_WP{k}"].to_numpy() / W
            cov = df[f"{model}_WYP{k}"].to_numpy() / W - my * mp
            vy = df[f"{model}_WYY{k}"].to_numpy() / W - my * my
            vp = df[f"{model}_WPP{k}"].to_numpy() / W - mp * mp
            res.append(cov / np.sqrt(np.maximum(vy * vp, 1e-18)))
        return 0.5 * (res[0] + res[1])


    _tot = clean.select(sum((wp_expr(m) for m in MODELS), []))
    mo.md("# EDA 2: where the score lives\n\nClean split (devB + holdout, never used for fitting). Global WP: "
          + ", ".join(f"**{m}** {wp_from_cols(_tot, m)[0]:.4f}" for m in MODELS))
    return MODELS, alt, clean, mo, np, pl, wp_expr, wp_from_cols


@app.cell(hide_code=True)
def _(mo):
    bin_var = mo.ui.dropdown(["step", "log_since_book", "log_since_trade", "log_since_i1", "log_n_book", "ay0", "ay1", "cls", "base0", "ab180"],
                             value="log_since_book", label="bin variable")
    n_bins = mo.ui.slider(4, 20, value=10, label="quantile bins", show_value=True)
    bin_class = mo.ui.dropdown({"all": -1, "class 0": 0, "class 1": 1, "class 2": 2, "class 3": 3}, value="all", label="class filter")
    mo.hstack([bin_var, n_bins, bin_class], justify="start")
    return bin_class, bin_var, n_bins


@app.cell(hide_code=True)
def _(
    MODELS,
    alt,
    bin_class,
    bin_var,
    clean,
    mo,
    n_bins,
    np,
    pl,
    wp_expr,
    wp_from_cols,
):
    _df = clean if bin_class.value < 0 else clean.filter(pl.col("cls") == bin_class.value)
    _v = bin_var.value
    if _v == "cls":
        _df = _df.with_columns(bin=pl.col("cls").cast(pl.Int32))
    else:
        _vals = _df[_v].to_numpy()
        _edges = np.unique(np.quantile(_vals, np.linspace(0, 1, n_bins.value + 1)[1:-1]))
        _df = _df.with_columns(bin=pl.Series(np.searchsorted(_edges, _vals, side="right").astype(np.int32)))
    # global means for covariance contribution shares
    _g = {}
    for _m in MODELS:
        for _k in (0, 1):
            _y = _df[f"y{_k}"].clip(-2, 2).to_numpy(); _p = _df[f"{_m}{_k}"].clip(-2, 2).to_numpy(); _w = np.abs(_y)
            _g[(_m, _k)] = ((_w * _y).sum() / _w.sum(), (_w * _p).sum() / _w.sum())
    _agg = _df.group_by("bin").agg([pl.len().alias("rows"), pl.col(_v).min().alias("lo"), pl.col(_v).max().alias("hi"),
                                    ((pl.col("ay0") + pl.col("ay1")) / 2).sum().alias("weight")] + sum((wp_expr(m) for m in MODELS), [])).sort("bin")
    _cov = {}
    for _m in MODELS:
        _parts = []
        for _k in (0, 1):
            _my, _mp = _g[(_m, _k)]
            _W = _agg[f"{_m}_W{_k}"].to_numpy()
            _c = _agg[f"{_m}_WYP{_k}"].to_numpy() - _mp * _agg[f"{_m}_WY{_k}"].to_numpy() - _my * _agg[f"{_m}_WP{_k}"].to_numpy() + _my * _mp * _W
            _parts.append(_c / _c.sum())
        _cov[_m] = 0.5 * (_parts[0] + _parts[1])
    bins_table = _agg.select("bin", "rows", "lo", "hi", (pl.col("weight") / pl.col("weight").sum()).alias("weight share")).with_columns(
        **{f"WP {m}": wp_from_cols(_agg, m) for m in MODELS}, **{f"cov share {m}": _cov[m] for m in ("base", "ab18")}
    ).with_columns(**{"ab18 - base": pl.col("WP ab18") - pl.col("WP base")})
    _long = bins_table.unpivot(index=["bin"], on=[f"WP {m}" for m in MODELS], variable_name="model", value_name="WP")
    _line = alt.Chart(_long).mark_line(point=True, strokeWidth=2).encode(
        x=alt.X("bin:O", title=f"{_v} quantile bin"), y=alt.Y("WP:Q", scale=alt.Scale(zero=False)),
        color=alt.Color("model:N", scale=alt.Scale(range=["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"])),
        tooltip=["bin", "model", alt.Tooltip("WP:Q", format=".4f")]).properties(width=520, height=260, title="within-bin WP")
    _bar = alt.Chart(bins_table).mark_bar(color="#8a8984", cornerRadiusEnd=3).encode(
        x=alt.X("bin:O"), y=alt.Y("weight share:Q"), tooltip=["bin", alt.Tooltip("weight share:Q", format=".3f"), "lo", "hi"]
    ).properties(width=260, height=260, title="metric weight share")
    mo.vstack([mo.hstack([_line, _bar]), bins_table])
    return


@app.cell(hide_code=True)
def _(alt, clean, mo, pl, wp_expr, wp_from_cols):
    _agg = clean.group_by("seq").agg([pl.first("cls"), pl.len().alias("scored_rows"),
                                        ((pl.col("ay0") + pl.col("ay1")) / 2).sum().alias("weight"),
                                        pl.col("since_book").median().alias("median_since_book")] + sum((wp_expr(m) for m in ("base", "ab18")), []))
    seq_table = _agg.select("seq", "cls", "scored_rows", "weight", "median_since_book").with_columns(
        wp_base=wp_from_cols(_agg, "base"), wp_ab18=wp_from_cols(_agg, "ab18")).with_columns(
        delta=pl.col("wp_ab18") - pl.col("wp_base"), cls=pl.col("cls").cast(pl.Utf8)).filter(pl.col("scored_rows") > 50)
    _brush = alt.selection_interval(encodings=["x", "y"])
    _sc = alt.Chart(seq_table).mark_circle(size=28, opacity=0.6).encode(
        x=alt.X("wp_base:Q", title="per-sequence WP, baseline x0.25"), y=alt.Y("wp_ab18:Q", title="per-sequence WP, ab18"),
        color=alt.Color("cls:N", scale=alt.Scale(range=["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]), title="class"),
        tooltip=["seq", "cls", "scored_rows", alt.Tooltip("wp_base:Q", format=".3f"), alt.Tooltip("wp_ab18:Q", format=".3f"), "median_since_book"]
    ).add_params(_brush).properties(width=520, height=420, title="sequences (brush to select)")
    seq_chart = mo.ui.altair_chart(_sc, chart_selection=False, legend_selection=False)
    mo.vstack([mo.md("## Per-sequence explorer\nWithin-sequence WP per model on clean sequences. Brush a region to list sequences."), seq_chart])
    return seq_chart, seq_table


@app.cell(hide_code=True)
def _(mo, pl, seq_chart, seq_table):
    _sel = seq_chart.value
    _sel = seq_table if _sel is None or len(_sel) == 0 else pl.DataFrame(_sel) if not isinstance(_sel, pl.DataFrame) else _sel
    mo.vstack([mo.md(f"selected **{len(_sel)}** sequences; mean delta ab18 - base {float(_sel['delta'].mean()):+.4f}"),
               mo.ui.table(_sel.sort("delta"), page_size=10)])
    return


if __name__ == "__main__":
    app.run()
