import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import polars as pl
    import matplotlib.pyplot as plt
    from pathlib import Path

    ROOT = Path(mo.notebook_dir()).parent
    DATA_DIR = ROOT / "datasets"
    SEQUENCE_LENGTH = 20_000
    SEED = 20261017

    FEATURE_COLUMNS = [
        f"i{i}_{kind}{j}"
        for i in range(2)
        for kind, n in [("p", 22), ("v", 22), ("dp", 4), ("dv", 4)]
        for j in range(n)
    ] + [f"a{i}" for i in range(8)]

    GROUPS = {
        "i0 bid price-like": [f"i0_p{i}" for i in range(11)],
        "i0 ask price-like": [f"i0_p{i}" for i in range(11, 22)],
        "i0 bid volume-like": [f"i0_v{i}" for i in range(11)],
        "i0 ask volume-like": [f"i0_v{i}" for i in range(11, 22)],
        "i0 trade price": [f"i0_dp{i}" for i in range(4)],
        "i0 trade volume": [f"i0_dv{i}" for i in range(4)],
        "i1 bid price-like": [f"i1_p{i}" for i in range(11)],
        "i1 ask price-like": [f"i1_p{i}" for i in range(11, 22)],
        "i1 bid volume-like": [f"i1_v{i}" for i in range(11)],
        "i1 ask volume-like": [f"i1_v{i}" for i in range(11, 22)],
        "i1 trade price": [f"i1_dp{i}" for i in range(4)],
        "i1 trade volume": [f"i1_dv{i}" for i in range(4)],
        "additional a0-a7": [f"a{i}" for i in range(8)],
    }
    plt.rcParams.update({
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e6e5e0",
        "grid.linewidth": 0.5,
    })
    return (
        DATA_DIR,
        FEATURE_COLUMNS,
        GROUPS,
        SEED,
        SEQUENCE_LENGTH,
        mo,
        np,
        pa,
        pl,
        plt,
        pq,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Raw train and validation data EDA

    A lightweight look at the supplied Parquet files: raw sample values, per-column
    quantiles, feature-group distributions, and short raw sequence traces.

    The plots and summaries use **four complete 20,000-row sequences per file**,
    selected deterministically with seed 20261017. All rows in those sampled
    sequences are included, including warmup and unscored rows. The row-group IDs
    are shown below so the sample can be reproduced. These are descriptive sample
    summaries, not full-file estimates or validation scores.

    Feature groups follow the supplied state layout (112 columns, positions 0–111).
    The two target columns t0 and t1 are shown separately.
    """)
    return


@app.cell(hide_code=True)
def _(DATA_DIR, FEATURE_COLUMNS, SEED, SEQUENCE_LENGTH, mo, np, pa, pl, pq):
    def load_sample(split):
        path = DATA_DIR / f"{split}.parquet"
        parquet = pq.ParquetFile(path)
        names = parquet.schema_arrow.names
        count = parquet.metadata.num_row_groups
        candidates = np.arange(count, dtype=np.int32)

        # This local train row group is documented as corrupt in src/wnn/data.py.
        if split == "train":
            candidates = candidates[candidates != 528]

        rng = np.random.default_rng(SEED + (1 if split == "valid" else 0))
        selected = np.sort(rng.choice(candidates, size=4, replace=False)).tolist()

        columns = ["seq_ix", "step_in_seq"]
        if "need_prediction" in names:
            columns.append("need_prediction")
        columns += FEATURE_COLUMNS
        columns += [name for name in ("is_scored", "t0", "t1") if name in names]

        tables = []
        rows = []
        for row_group in selected:
            table = parquet.read_row_group(row_group, columns=columns, use_threads=False)
            if table.num_rows != SEQUENCE_LENGTH:
                raise ValueError(f"{split} row group {row_group} has {table.num_rows} rows")
            table = table.append_column(
                "row_group", pa.array(np.full(table.num_rows, row_group, dtype=np.int32))
            )
            tables.append(table)
            rows.append({
                "split": split,
                "row_group": row_group,
                "seq_ix": table.column("seq_ix")[0].as_py(),
                "rows": table.num_rows,
            })

        frame = pl.from_arrow(pa.concat_tables(tables))
        return {"frame": frame, "row_groups": selected, "rows": rows}

    TRAIN = load_sample("train")
    VALID = load_sample("valid")
    SAMPLE_INFO = pl.DataFrame(TRAIN["rows"] + VALID["rows"]).sort(["split", "row_group"])
    mo.vstack([
        mo.md(f"Loaded {TRAIN['frame'].height:,} train rows and {VALID['frame'].height:,} validation rows."),
        SAMPLE_INFO,
    ])
    return TRAIN, VALID


@app.cell(hide_code=True)
def _(FEATURE_COLUMNS, TRAIN, VALID, mo, np, pl):
    SUMMARY_COLUMNS = FEATURE_COLUMNS + ["t0", "t1"]

    def summarize_columns(split, frame):
        summary_rows = []
        for column in SUMMARY_COLUMNS:
            if column not in frame.columns:
                continue
            values = frame.get_column(column).to_numpy()
            finite = np.isfinite(values)
            valid = values[finite]
            q = np.quantile(valid, [0, 0.01, 0.25, 0.5, 0.75, 0.99, 1]) if valid.size else [np.nan] * 7
            summary_rows.append({
                "split": split,
                "column": column,
                "rows": int(values.size),
                "finite": int(finite.sum()),
                "nonfinite": int((~finite).sum()),
                "min": float(q[0]),
                "p01": float(q[1]),
                "p25": float(q[2]),
                "median": float(q[3]),
                "p75": float(q[4]),
                "p99": float(q[5]),
                "max": float(q[6]),
                "mean": float(np.mean(valid)) if valid.size else np.nan,
                "std": float(np.std(valid)) if valid.size else np.nan,
            })
        return summary_rows

    COLUMN_SUMMARY = pl.DataFrame(
        summarize_columns("train", TRAIN["frame"]) + summarize_columns("valid", VALID["frame"])
    )
    mo.vstack([
        mo.md("## Per-column raw values\n\nQuantiles are computed over the sampled rows for each feature column. Use the table search and pagination to inspect individual columns."),
        mo.ui.table(COLUMN_SUMMARY, page_size=20),
    ])
    return


@app.cell(hide_code=True)
def _(GROUPS, TRAIN, VALID, mo, np, plt):
    PLOT_GROUPS = dict(GROUPS)
    PLOT_GROUPS["target t0"] = ["t0"]
    PLOT_GROUPS["target t1"] = ["t1"]

    hist_fig, hist_axes = plt.subplots(4, 4, figsize=(16, 12))
    hist_axes = hist_axes.ravel()
    for hist_ax, (hist_label, hist_columns) in zip(hist_axes, PLOT_GROUPS.items()):
        hist_train = TRAIN["frame"].select(hist_columns).to_numpy().reshape(-1)
        hist_valid = VALID["frame"].select(hist_columns).to_numpy().reshape(-1)
        hist_train = hist_train[np.isfinite(hist_train)]
        hist_valid = hist_valid[np.isfinite(hist_valid)]
        hist_pooled = np.concatenate([hist_train, hist_valid])
        if hist_pooled.size == 0:
            hist_ax.set_title(f"{hist_label}: no finite values")
            continue
        hist_low, hist_high = np.quantile(hist_pooled, [0.01, 0.99])
        if hist_high <= hist_low:
            hist_low, hist_high = hist_low - 0.5, hist_high + 0.5
        hist_edges = np.linspace(hist_low, hist_high, 51)
        hist_ax.hist(hist_train, bins=hist_edges, density=True, alpha=0.45, label="train")
        hist_ax.hist(hist_valid, bins=hist_edges, density=True, alpha=0.45, label="validation")
        hist_ax.set_title(hist_label)
        hist_ax.set_xlabel("raw value (1st–99th percentile window)")
        hist_ax.set_ylabel("density")
        if hist_ax is hist_axes[0]:
            hist_ax.legend(frameon=False, fontsize=8)

    for hist_ax in hist_axes[len(PLOT_GROUPS):]:
        hist_ax.set_visible(False)
    hist_fig.suptitle("Raw feature-group and target distributions", y=1.01, fontsize=14)
    hist_fig.tight_layout()
    mo.vstack([
        mo.md("## Raw distributions\n\nEach panel pools columns in one supplied feature group. Train and validation use the same 1st–99th percentile x-range per panel; quantile tables above retain tails and extrema."),
        hist_fig,
    ])
    return


@app.cell(hide_code=True)
def _(TRAIN, VALID, mo, np, plt):
    TRACE_COLUMNS = [
        "i0_p0", "i0_p10", "i0_p11",
        "i0_v0", "i0_v11", "i0_dp0",
        "i0_dv0", "i1_p0", "a0",
    ]
    trace_length = 1_000
    trace_x = np.arange(trace_length)
    trace_fig, trace_axes = plt.subplots(3, 3, figsize=(15, 9))
    trace_axes_flat = trace_axes.ravel()
    for trace_ax, trace_column in zip(trace_axes_flat, TRACE_COLUMNS):
        trace_train = TRAIN["frame"].get_column(trace_column).head(trace_length).to_numpy()
        trace_valid = VALID["frame"].get_column(trace_column).head(trace_length).to_numpy()
        trace_ax.plot(trace_x, trace_train, lw=0.8, alpha=0.8, label="train")
        trace_ax.plot(trace_x, trace_valid, lw=0.8, alpha=0.8, label="validation")
        trace_ax.set_title(trace_column)
        trace_ax.set_xlabel("row in sampled sequence")
        trace_ax.set_ylabel("raw value")
        if trace_ax is trace_axes_flat[0]:
            trace_ax.legend(frameon=False, fontsize=8)
    trace_fig.suptitle("Raw values through the first sampled sequence", y=1.01, fontsize=14)
    trace_fig.tight_layout()
    mo.vstack([
        mo.md("## Short raw sequence traces\n\nFirst 1,000 rows of the first sampled sequence in each file. Axes retain raw values; no normalization or smoothing."),
        trace_fig,
    ])
    return


@app.cell(hide_code=True)
def _(FEATURE_COLUMNS, TRAIN, VALID, mo, pl):
    RAW_META_COLUMNS = [
        name for name in ["row_group", "seq_ix", "step_in_seq", "need_prediction", "is_scored", "t0", "t1"]
        if name in TRAIN["frame"].columns and name in VALID["frame"].columns
    ]
    RAW_COLUMNS = RAW_META_COLUMNS + FEATURE_COLUMNS
    train_raw_rows = TRAIN["frame"].select(RAW_COLUMNS).head(5)
    valid_raw_rows = VALID["frame"].select(RAW_COLUMNS).head(5)

    feature_rows = []
    for feature_position, feature_name in enumerate(FEATURE_COLUMNS):
        train_values = TRAIN["frame"].get_column(feature_name).head(5).to_list()
        valid_values = VALID["frame"].get_column(feature_name).head(5).to_list()
        feature_rows.append({
            "state_position": feature_position,
            "feature": feature_name,
            **{f"train_row_{i}": value for i, value in enumerate(train_values)},
            **{f"valid_row_{i}": value for i, value in enumerate(valid_values)},
        })
    RAW_FEATURE_VALUES = pl.DataFrame(feature_rows)

    mo.vstack([
        mo.md("## Raw state values for all columns\n\nOne row per state feature, in the supplied 0–111 order. The five values per split are rows 0–4 of each sampled sequence."),
        mo.ui.table(RAW_FEATURE_VALUES, page_size=20),
        mo.md("## Raw rows\n\nFive unmodified rows from the first sampled sequence in each file, including metadata, targets, and the full 112-feature state. Use the table search and Columns control to inspect the wide row format."),
        mo.md("### Train sample"),
        train_raw_rows,
        mo.md("### Validation sample"),
        valid_raw_rows,
    ])
    return


if __name__ == "__main__":
    app.run()
