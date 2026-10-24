import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import json
    from pathlib import Path
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    pa.set_cpu_count(4)
    pa.set_io_thread_count(2)
    repo = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
    eda_dir = repo / "research/campaign_2026-10-24/resume4h_r1/eda"
    return eda_dir, json, mo, np, pa, pq, repo


@app.cell
def _(mo):
    mo.md("""
    # Research EDA: clocks, censoring, and useful disagreement

    This notebook supports the October 24 four-hour research cycle. It uses deterministic samples and streaming reads under the shared memory limit. Existing validation is **exposed**; descriptive associations are not new held-out evidence.

    **Questions:** How often is field age still censored by sequence start? Does age mainly track sequence position or activity? Where does the fixed history correction help? Which updated PRs change our next experiment?

    The official metric and training recipes remain unchanged. A label-derived slice is an explanatory view, never a serving feature.
    """)
    return


@app.cell
def sample_plan(eda_dir, json, mo, np, repo):
    data_dir = repo / "datasets"
    feature_names = [f"i{i}_{g}{j}" for i in range(2) for g, n in [("p", 22), ("v", 22), ("dp", 4), ("dv", 4)] for j in range(n)] + [f"a{i}" for i in range(8)]
    _eda_rng = np.random.default_rng(20261024)
    eda_sample = {name: np.sort(_eda_rng.choice(n, 96, replace=False)).tolist() for name, n in [("train", 10607), ("valid", 1873)]}
    eda_plan = {"id": "EDA-CENSORING-20261024-R1", "seed": 20261024, "sample": eda_sample, "selection": "96 uniform row groups per source without target/class stratification", "feature_columns": feature_names, "position_edges": [0, 99, 500, 2000, 10000, 20000], "age_thresholds": [1, 16, 64, 512, 4096, 10000], "questions": ["field-specific censored age coverage", "exact versus capped ages beyond position", "age and update activity by source and position"], "status": "frozen_before_new_values", "label_use": "descriptive clipped-label magnitude mass only; no predictor or coefficient fitting", "exposure": "previously inspected train/valid; not fresh holdout", "comparison_scope": "descriptive source samples; no official model score or causal conclusion", "input_identity": {name: {"path": str(data_dir/f"{name}.parquet"), "bytes": (data_dir/f"{name}.parquet").stat().st_size, "mtime_ns": (data_dir/f"{name}.parquet").stat().st_mtime_ns} for name in eda_sample}}
    _plan_path = eda_dir / "EDA_PLAN_R1.json"
    if _plan_path.exists():
        assert json.loads(_plan_path.read_text()) == eda_plan
    else:
        _plan_path.write_text(json.dumps(eda_plan, indent=2)+"\n")
    mo.md(f"Frozen sample: **{sum(map(len, eda_sample.values()))} whole sequences**, two source files, seed 20261024. One row group in memory at a time.")
    return data_dir, eda_sample, feature_names


@app.cell
def clock_statistics(np, pa):
    def describe_censoring(table: pa.Table, names: list[str]) -> dict[str, object]:
        """Summarize causal field clocks within one complete observed sequence."""
        x = np.column_stack([table[c].to_numpy() for c in names]).astype(np.float32)
        y = np.column_stack([table[c].to_numpy() for c in ["t0", "t1"]]).astype(np.float64)
        need = table["need_prediction"].to_numpy().astype(bool)
        scored = table["is_scored"].to_numpy().astype(bool) if "is_scored" in table.column_names else need
        selected = need & scored
        seq = table["seq_ix"].to_numpy()
        assert len(x) == 20000 and len(np.unique(seq)) == 1
        assert np.isfinite(x).all() and np.isfinite(y).all()
        idx = np.arange(len(x), dtype=np.int32)
        weight = np.abs(np.clip(y, -2, 2)) * selected[:, None]
        changed = np.zeros(x.shape, dtype=bool)
        changed[1:] = x[1:] != x[:-1]
        thresholds = np.asarray([1, 16, 64, 512, 4096, 10000])
        position_edges = [0, 99, 500, 2000, 10000, 20000]
        result = {
            "seq_ix": int(seq[0]), "rows": len(x), "selected_rows": int(selected.sum()),
            "selected_mass": weight.sum(0), "changes": changed.sum(0),
            "censored_rows": np.zeros(len(names), dtype=np.int64),
            "censored_mass": np.zeros((len(names), 2)),
            "long_age_rows": np.zeros((len(names), len(thresholds)), dtype=np.int64),
            "long_age_mass": np.zeros((len(names), len(thresholds), 2)),
            "censored_long_age_mass": np.zeros((len(names), len(thresholds), 2)),
            "position_selected_rows": np.zeros(5, dtype=np.int64),
            "position_censored_rows": np.zeros((5, len(names)), dtype=np.int64),
            "position_selected_mass": np.zeros((5, 2)),
            "position_censored_mass": np.zeros((5, len(names), 2)),
        }
        for k, (lo, hi) in enumerate(zip(position_edges[:-1], position_edges[1:])):
            result["position_selected_rows"][k] = selected[lo:hi].sum()
            result["position_selected_mass"][k] = weight[lo:hi].sum(0)
        for j in range(len(names)):
            seen = np.maximum.accumulate(changed[:, j])
            age = idx - np.maximum.accumulate(np.where(changed[:, j], idx, 0))
            censored = ~seen
            result["censored_rows"][j] = (censored & selected).sum()
            result["censored_mass"][j] = weight[censored].sum(0)
            for k, threshold in enumerate(thresholds):
                old = age >= threshold
                result["long_age_rows"][j, k] = (old & selected).sum()
                result["long_age_mass"][j, k] = weight[old].sum(0)
                result["censored_long_age_mass"][j, k] = weight[old & censored].sum(0)
            for k, (lo, hi) in enumerate(zip(position_edges[:-1], position_edges[1:])):
                result["position_censored_rows"][k, j] = (censored[lo:hi] & selected[lo:hi]).sum()
                result["position_censored_mass"][k, j] = weight[lo:hi][censored[lo:hi]].sum(0)
        return result

    # Exact synthetic invariant: a never-changing field stays censored; an observed
    # change ends censoring at that row, with zero age there. No labels are fitted.
    _test_change = np.array([False, False, True, False])
    _test_idx = np.arange(4)
    assert (~np.maximum.accumulate(_test_change)).tolist() == [True, True, False, False]
    assert (_test_idx - np.maximum.accumulate(np.where(_test_change, _test_idx, 0))).tolist() == [0, 1, 0, 1]

    return (describe_censoring,)


@app.cell
def stream_sample(
    data_dir,
    describe_censoring,
    eda_dir,
    eda_sample,
    feature_names,
    json,
    mo,
    np,
    pq,
):
    import hashlib as _hashlib
    import time as _time
    import resource as _resource
    _plan_sha = _hashlib.sha256((eda_dir / "EDA_PLAN_R1.json").read_bytes()).hexdigest()
    _result_path = eda_dir / "CENSORING_SAMPLE_R1.json"
    if _result_path.exists():
        eda_results = json.loads(_result_path.read_text())
        assert eda_results["plan_sha256"] == _plan_sha
    else:
        _start = _time.monotonic()
        eda_results = {"plan_sha256": _plan_sha, "new_fits": 0, "model_evaluations": 0, "sources": {}, "limitations": ["Uniform sequence sample, not a full census", "Train denominator is need_prediction; valid denominator additionally requires is_scored", "All data were previously exposed", "Censoring association does not establish a predictive benefit", "Raw source size/mtime and footer checked; entire raw Parquet payloads not rehashed"]}
        for _source, _groups in eda_sample.items():
            _pf = pq.ParquetFile(data_dir / f"{_source}.parquet", memory_map=True)
            assert _pf.num_row_groups == {"train": 10607, "valid": 1873}[_source]
            _columns = feature_names + ["seq_ix", "need_prediction", "t0", "t1"]
            if _source == "valid":
                _columns.append("is_scored")
            _aggregate = None
            _members = []
            for _step, _group in enumerate(_groups):
                _table = _pf.read_row_group(_group, columns=_columns, use_threads=True)
                _row = describe_censoring(_table, feature_names)
                _members.append({"row_group": _group, "seq_ix": _row["seq_ix"], "selected_rows": _row["selected_rows"]})
                if _aggregate is None:
                    _aggregate = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in _row.items() if k != "seq_ix"}
                else:
                    for _key in _aggregate:
                        _aggregate[_key] += _row[_key]
                del _table, _row
                if (_step + 1) % 16 == 0:
                    print(f"{_source}: {_step + 1}/{len(_groups)} complete sequences")
            eda_results["sources"][_source] = {"members": _members, "statistics": {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in _aggregate.items()}}
            del _pf, _aggregate
        eda_results["wall_seconds"] = _time.monotonic() - _start
        eda_results["process_peak_rss_kib"] = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss
        _result_path.write_text(json.dumps(eda_results, indent=2) + "\n")
    mo.md(f"Completed **{sum(len(v['members']) for v in eda_results['sources'].values())} sequences** in {eda_results['wall_seconds']:.1f}s. Peak kernel RSS: {eda_results['process_peak_rss_kib']/1024:.0f} MiB. Results saved with the frozen sample-plan hash.")

    return


if __name__ == "__main__":
    app.run()
