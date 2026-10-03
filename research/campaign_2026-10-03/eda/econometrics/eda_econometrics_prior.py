import marimo

__generated_with = "0.16.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path
    ROOT = Path(__file__).resolve().parents[4]
    OUT = Path(__file__).resolve().parent
    FIG = OUT / "fig"
    CACHE = OUT / "cache"
    return CACHE, FIG, OUT, Path, ROOT, matplotlib, mo, np, pd, plt, pq


@app.cell
def _(mo):
    mo.md("""# Econometric EDA: stochastic structure of inputs and targets
    Time-series econometrician view: lattice targets, horizon, heteroskedasticity, metric-optimal estimand.""")
    return


if __name__ == "__main__":
    app.run()
