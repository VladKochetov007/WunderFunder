import marimo

__generated_with = "0.0.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import sys
    sys.path.insert(0, '/home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-03/eda/fractal')
    import numpy as np
    import fx
    mo.md("# Fractal / long-memory EDA\nHeavy computations run in standalone scripts (numba, streaming row groups); this notebook loads their .npz outputs and shows the PNGs.")
    return fx, mo, np


if __name__ == "__main__":
    app.run()
