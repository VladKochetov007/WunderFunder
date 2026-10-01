import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    mo.md("# Connectome: data reconstruction and model research\nGlobal Weighted Pearson, clip ±2, scored required rows pooled across sequences. Target expected public score: **0.6660**. The expectation gate remains unmet.")
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    from pathlib import Path
    import json
    campaign_dir = Path("research/campaign_2026-09-29")
    job_rows = [json.loads(p.read_text()) for p in sorted((campaign_dir / "jobs").glob("*.json"))]
    gpu_seconds_used = sum(float(j.get("elapsed_seconds", 0)) for j in job_rows if j["kind"] == "gpu")
    lines = [f"- {j['name']}: {j['status']}, {j['elapsed_seconds']:.1f}s, exit {j.get('exit_code')}" for j in job_rows]
    mo.md(f"**GPU budget:** {gpu_seconds_used / 3600:.3f} / 8.000 hours\n\n" + "\n".join(lines))
    return


@app.cell(hide_code=True)
def _(mo):
    from pathlib import Path as _Path
    _control_rows = []
    for _fold in range(4):
        for _seed in (3, 4):
            _parent = f"c20260929_since_s{_seed}_f{_fold}"
            _ft = f"c20260929_ft_since_s{_seed}_f{_fold}"
            _control_rows.append({"fold": _fold, "seed": _seed,
                "parent fixed epoch 3": (_Path("artifacts/runs") / _parent / "model_ep3.pt").exists(),
                "two-stage FT seed 3": (_Path("artifacts/runs") / _ft / "model_ep0.pt").exists(),
                "validation predictions": (_Path("cache") / f"pred_{_ft}_valid.npy").exists()})
    mo.vstack([mo.md("## Clean matched controls\nTraining excludes linked evaluation components. These are recipe analogs of the public pair; fold selection has been inspected previously."), mo.ui.table(_control_rows, selection=None)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Data questions under investigation
    - Can current price codes identify spread across independent training chains?
    - Do transition moves remain consistent across target branches and quiet rows?
    - Which causal feature statistics explain changes in target distributions?
    - Do external physical-book publication experiments distinguish valid mechanisms for the supplied anonymised data?

    **Existing counterevidence:** class 3 quote-clock reconstruction still fails on 36.5% of eligible target mass; a content-memory feature forecasting gain did not improve target WP. Continuous whole-book price shifts failed the codebook support test. Each new representation needs its own falsifier.

    Important results will be saved under `research/campaign_2026-09-30/`, with source identities and fixed sample membership.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    from pathlib import Path as _DataPath
    import json as _data_json
    _pooled = _data_json.loads((_DataPath("research/campaign_2026-09-30/data_reconstruction") / "pooled_results.json").read_text())
    _spec = _data_json.loads((_DataPath("research/campaign_2026-09-30/audit") / "data_spec_audit.json").read_text())
    _relation_rows = []
    for _class, _counts in _pooled["by_diagnostic_class"].items():
        _relation_rows.append({"diagnostic class": int(_class), "held-out TRAIN groups": _counts["groups"], "all rows": _counts["rows"], "mapped rows %": round(100 * _counts["mapped_rows"] / _counts["rows"], 2), "class exact on mapped %": round(100 * _counts["class_exact_rows"] / _counts["mapped_rows"], 2), "interior spread exact on mapped %": round(100 * _counts["interior_spread_exact_rows"] / _counts["interior_mapped_rows"], 2), "outside offset2 on mapped %": round(100 * _spec["summary"][_class]["outside_offset2_fraction"], 2)})
    mo.vstack([mo.md("## Measured current-code structure\nIndependent raw TRAIN replay passed. Fit64 / test32 distinct components; 640,000 test rows. Class strata below use targets for grading only. The deployable lookup reads only current prices and the TRAIN-fitted map. Unknown rows stay in the denominators.\n\n**Target-law limits:** fixed losing offset2 fails on many classes0/1 tails; first-passage alternatives failed. Nonzero code cycles rule out an absolute-price lookup. These are data identities, with no measured model gain."), mo.ui.table(_relation_rows, selection=None), mo.md("[Full report](../research/campaign_2026-09-30/data_reconstruction/REPORT.md). Original auxiliary remains a control: its future-tail supervision may be useful.")])
    return


@app.cell(hide_code=True)
def _(mo):
    from pathlib import Path as _EventPath
    import json as _event_json
    _event_result = _event_json.loads((_EventPath("research/campaign_2026-09-30/data_reconstruction/full112_relations") / "quote_endpoint_results.json").read_text())
    _event_rows = []
    for _class in ("0", "1", "2", "3"):
        _c = _event_result["counts"][_class]
        _known = _c["known_nonzero_events"] + _c["known_zero_events"]
        _event_rows.append({"diagnostic class": int(_class), "code events": _c["code_events"], "known decoded move coverage %": round(100 * _known / _c["code_events"], 2), "nonzero moves": _c["known_nonzero_events"], "nonzero without a6 endpoint": _c["not_endpoint_on_known_nonzero_events"], "zero with a6 endpoint": _c["endpoint_on_known_zero_events"], "unknown move events": _c["unknown_move_events"]})
    mo.vstack([mo.md("## Direct causal event coordinate\n**a6 == float32(-5.199337482452393)** matches all67,839 known nonzero TRAIN-supervised moves and none of45,649 known zero moves. An independent raw-parquet replay reproduced every count. Unknown movement remains16,894/130,382 events. This flag reads only the current input and needs no lookup at inference.\n\nPhysical quote identity and elapsed time units remain unproven. E-091's earlier age correlations are reproduced with stronger monotonicity limits; they do not establish seconds. The normal-quantile endpoint fingerprint is compatible with H-003, not a source match. Three nonorigin endpoint rows have unchanged price codes. No model score gain has been measured."), mo.ui.table(_event_rows, selection=None)])
    return


if __name__ == "__main__":
    app.run()
