import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():

    import json, re
    from hashlib import sha256
    from pathlib import Path
    import numpy as np
    import polars as pl
    import matplotlib.pyplot as plt
    import marimo as mo
    ROOT = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
    PLOT_DIR = ROOT / "artifacts" / "plots"
    PLOT_DIR.mkdir(parents=True, exist_ok=True)


    def refresh_plot_manifest() -> None:
        rows: list[str] = []
        for image in sorted(PLOT_DIR.iterdir()):
            if not image.is_file() or image.suffix.lower() not in {".png", ".svg", ".pdf"}:
                continue
            name = image.name
            if name.startswith("viz__"):
                source = "artifacts/viz/" + name.removeprefix("viz__")
            elif name.startswith("research_artifacts__"):
                source = "research/artifacts/" + name.removeprefix("research_artifacts__")
            elif name.startswith("research_visual__"):
                source = "research/visual/" + name.removeprefix("research_visual__").replace("__", "/")
            else:
                source = "notebooks/learning_dynamics.py (marimo export cell)"
            digest = sha256(image.read_bytes()).hexdigest()[:16]
            link = "../../artifacts/plots/" + name
            rows.append(f"| [{name}]({link}) | {source} | {digest} | {image.stat().st_size:,} |")
        manifest = ROOT / "research" / "artifacts" / "plots_manifest.md"
        header = chr(10).join((
            "# Research plot archive",
            "",
            "All research figures are archived in artifacts/plots/. Notebook plots refresh this index when saved.",
            "",
            "| Plot | Source | SHA-256 prefix | Bytes |",
            "|---|---|---:|---:|",
            "",
        ))
        manifest.write_text(header + chr(10).join(rows) + chr(10))


    return PLOT_DIR, ROOT, json, mo, np, pl, plt, refresh_plot_manifest


@app.cell(hide_code=True)
def _(ROOT, json, mo, pl):

    def _args(run):
        log = ROOT / "artifacts" / "logs" / f"{run}.log"
        try:
            return json.loads(log.open().readline())
        except Exception:
            return {}

    _rows = []
    for _h in sorted((ROOT / "artifacts" / "runs").glob("*/history.json")):
        _run = _h.parent.name
        try:
            _hist = json.loads(_h.read_text())
        except Exception:
            continue
        _a = _args(_run)
        for _e in _hist:
            _bc = _e.get("by_class") or {}
            _row = dict(run=_run, epoch=_e.get("epoch"), wp=_e.get("weighted_pearson"),
                fold=int(_a.get("valid_fold", -1)), ema=float(_a.get("ema_teacher", 0.0)),
                epochs=_a.get("epochs"), lr=_a.get("lr"), wd=_a.get("wd"), hidden=_a.get("hidden"),
                chunk=_a.get("chunk"), since=float(_a.get("since_weight", 0.0)),
                train_source=str(_a.get("train_source")), init=str(_a.get("init", "")))
            for _k in ("0", "1", "2", "3"):
                _v = _bc.get(_k)
                _row[f"c{_k}"] = (_v.get("wp") if isinstance(_v, dict) else _v)
            _rows.append(_row)
    hist = pl.DataFrame(_rows, infer_schema_length=None)
    mo.md(f"**{hist['run'].n_unique()} runs, {hist.height} epoch evaluations** from `artifacts/runs/*/history.json`")
    return (hist,)


@app.cell(hide_code=True)
def _(PLOT_DIR, hist, pl, plt, refresh_plot_manifest):

    _fam = {
        "ab47_long_wd0p01_f0": ("12 ep, no EMA, wd .01", "tab:red"),
        "ab47_long_wd0p1_f0": ("12 ep, no EMA, wd .1", "tab:orange"),
        "ab62_since_f0": ("EMA-4 since", "tab:blue"),
        "ab68_ema6_since_f0": ("EMA-6 since", "tab:green"),
        "ab81_wd01_f0": ("EMA-6 since wd .1", "tab:olive"),
        "ab81_h192_f0": ("EMA-6 since h192", "tab:cyan"),
        "ab66_ch500_f0": ("EMA-4 chunk 500", "tab:purple"),
        "ab66_onorm_f0": ("EMA-4 online norm", "tab:brown"),
        "ab70_validrepeat_f0": ("EMA-4 valid x2", "tab:gray"),
    }
    _fig, _axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for _run, (_lab, _col) in _fam.items():
        _d = hist.filter(pl.col("run") == _run).sort("epoch")
        if _d.height == 0:
            continue
        _axes[0].plot(_d["epoch"] + 1, _d["wp"], marker="o", ms=3, lw=1.6, color=_col, label=_lab)
        _axes[1].plot(_d["epoch"] + 1, _d["c3"], marker="o", ms=3, lw=1.6, color=_col, label=_lab)
    _axes[0].set_title("fold-0 validation WP by epoch"); _axes[1].set_title("class 3 only")
    for _ax in _axes:
        _ax.set_xlabel("epoch"); _ax.grid(alpha=.3)
    _axes[0].set_ylim(0.645, 0.68)
    _axes[0].legend(fontsize=7, loc="lower right")
    _fig.tight_layout()
    _fig.savefig(PLOT_DIR / "learning_curves.png", dpi=160, bbox_inches="tight")
    refresh_plot_manifest()
    _fig
    return


@app.cell(hide_code=True)
def _(PLOT_DIR, hist, mo, np, pl, plt, refresh_plot_manifest):

    _pairs = []
    for _run in hist["run"].unique().to_list():
        if _run.startswith("ab82_ft_") and not _run.endswith("h"):
            _h = hist.filter(pl.col("run") == _run + "h")
            _f = hist.filter(pl.col("run") == _run)
            if _h.height and _f.height:
                _pairs.append(dict(member=_run.replace("ab82_ft_", ""), head_only=_h["wp"][-1], full=_f["wp"][-1],
                                   head_c3=_h["c3"][-1], full_c3=_f["c3"][-1]))
    ft = pl.DataFrame(_pairs).with_columns((pl.col("full") - pl.col("head_only")).alias("full_minus_head"),
                                            (pl.col("full_c3") - pl.col("head_c3")).alias("c3_full_minus_head")).sort("member")
    _fig, _ax = plt.subplots(figsize=(8, 3.6))
    _x = np.arange(ft.height)
    _ax.bar(_x, ft["full_minus_head"] * 1e4, color=["tab:red" if v < 0 else "tab:green" for v in ft["full_minus_head"]])
    _ax.set_xticks(_x, ft["member"].to_list(), rotation=45, ha="right", fontsize=8)
    _ax.axhline(0, color="k", lw=.8); _ax.set_ylabel("full FT − head-only FT (×1e-4 WP)")
    _ax.set_title("Second (all-parameter) fine-tune stage vs head-only stage, fold models")
    _fig.tight_layout()
    _fig.savefig(PLOT_DIR / "finetune_stage_comparison.png", dpi=160, bbox_inches="tight")
    refresh_plot_manifest()
    mo.vstack([ft, _fig])
    return


@app.cell(hide_code=True)
def _(PLOT_DIR, ROOT, mo, np, pl, plt, refresh_plot_manifest):

    import sys as _sys
    _sys.path.insert(0, str(ROOT / "scripts"))
    import decide as _decide
    import lb_calibration as _L
    from ensemble_scan import parse_blend as _parse
    from holdout_compare import blend_sums as _bs
    from wnn.splits import valid_row_groups as _vrg

    _v = np.load(ROOT / "cache" / "valid_targets.npz")
    _y, _m, _rg, _cls = _v["y"], _v["mask"], _v["row_group"], _v["market_class"]
    _, _held = _vrg(ROOT / "datasets" / "valid.parquet")
    _idx = np.flatnonzero(np.isin(_rg, _held))
    _adv = np.load(ROOT / "cache" / "adv_valid_score.npy")[_idx]
    _c = _cls[_idx]
    _unlike = np.zeros(len(_idx), bool)
    for _k in range(4):
        _s = _c == _k
        _unlike[_s] = _adv[_s] >= np.quantile(_adv[_s], 1 / 3)
    def _score(s, lam, w):
        return _decide.wp(s * w[:, None, None, None], lam)
    cal_rows = []
    for _name, (_spec, _pub) in _L.SCORED.items():
        _s = _bs(_parse(_spec), _idx, _y, _m)
        cal_rows.append(dict(package=_name, public=_pub, raw_holdout=_score(_s, 0.0, np.ones(len(_idx))),
                             calibrated=_score(_s, 0.05, _unlike.astype(float))))
    cal_df = pl.DataFrame(cal_rows)
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4.3))
    for _ax, _col, _t in ((_axes[0], "raw_holdout", "raw holdout, exact mask"), (_axes[1], "calibrated", "train-unlike 2/3, mask softened 0.05")):
        _x, _yy = cal_df[_col].to_numpy(), cal_df["public"].to_numpy()
        _ax.scatter(_x, _yy, s=28, color="tab:blue")
        for _n, _a, _b in zip(cal_df["package"], _x, _yy):
            _ax.annotate(_n.replace("final_", ""), (_a, _b), fontsize=6.5, xytext=(3, 2), textcoords="offset points")
        _b1, _a1 = np.polyfit(_x, _yy, 1)
        _xx = np.linspace(_x.min(), _x.max(), 10)
        _ax.plot(_xx, _a1 + _b1 * _xx, "k--", lw=.8)
        _r = np.corrcoef(_x, _yy)[0, 1]
        _ax.set_title(f"{_t}  (pearson {_r:.3f})", fontsize=9); _ax.set_xlabel("local WP"); _ax.set_ylabel("public LB"); _ax.grid(alpha=.3)
    _fig.tight_layout()
    _fig.savefig(PLOT_DIR / "leaderboard_proxy_calibration.png", dpi=160, bbox_inches="tight")
    refresh_plot_manifest()
    mo.vstack([mo.md("### Public leaderboard vs local proxies (10 scored packages, clean 368-sequence holdout)"), _fig])
    return


if __name__ == "__main__":
    app.run()
