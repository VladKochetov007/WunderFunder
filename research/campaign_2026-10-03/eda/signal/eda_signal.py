from __future__ import annotations

import marimo

__generated_with = "0.25.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import os, sys, json
    import numpy as np

    HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, HERE)
    os.chdir(HERE)
    import sio, feat

    return HERE, feat, json, mo, np, os, sio


@app.cell
def _(mo):
    mo.md(
        """
        # Signal-processing EDA (clean room)
        Scripts in this folder do the heavy lifting (each writes .npz/.json/.png); this notebook only
        displays them. Order: `catalogue.py` (per-sequence clocks + target quantum), `acf_clock.py`,
        `spectra.py`, `flicker.py`/`flicker2.py` (limit cycles), `jumpcause.py`, `horizon.py`,
        `ridge_bands.py` (linear SNR map by band/group/clock).
        """
    )
    return


@app.cell
def _(mo, np):
    z = np.load("catalogue.npz")
    names = list(z["names"])
    rows = []
    for split in ("train", "valid"):
        A = z[split]
        q = np.round(A[:, 2], 4)
        for uu in np.unique(q):
            m = q == uu
            c = {n: A[m, i] for i, n in enumerate(names)}
            rows.append(
                dict(
                    split=split,
                    quantum=float(uu),
                    ticks_per_unit=round(1 / uu, 2),
                    n_seq=int(m.sum()),
                    frac=round(m.mean(), 3),
                    t0_std=round(np.nanmean(c["t0std"]), 3),
                    t0_mean=round(np.nanmean(c["t0mean"]), 3),
                    rho_t0_t1=round(np.nanmean(c["rho"]), 3),
                    t0_changes=int(np.mean(c["n_t0chg"])),
                    i0_price_events=int(np.mean(c["n_i0p0"])),
                )
            )
    mo.vstack(
        [
            mo.md("## Target quantum classes (targets = integer ticks x quantum)"),
            mo.ui.table(rows, selection=None),
        ]
    )
    return


@app.cell
def _(mo):
    figs = [
        "fig_overview_rg0_0_20000.png",
        "fig_overview_rg0_5000_5600.png",
        "fig_pevents_rg0.png",
        "fig_pscatter_rg0.png",
        "fig_acf_clock.png",
        "fig_spectra_event.png",
        "fig_horizon_xcorr.png",
        "fig_ridge_bands.png",
        "fig_cycle_example.png",
        "fig_class_id.png",
    ]
    import os as _os

    mo.vstack([mo.vstack([mo.md(f"### {f}"), mo.image(src=f)]) for f in figs if _os.path.exists(f)])
    return


@app.cell
def _(json, mo, os):
    out = []
    for clk in ("ev", "row"):
        fn = f"ridge_bands_{clk}.json"
        if os.path.exists(fn):
            r = json.load(open(fn))
            out += [
                dict(
                    clock=clk,
                    config=k,
                    t0=round(v[0], 4),
                    t1=round(v[1], 4),
                    mean=round((v[0] + v[1]) / 2, 4),
                )
                for k, v in r.items()
            ]
    mo.vstack(
        [
            mo.md(
                "## Ridge filter-bank SNR map (pooled |y|-weighted corr on valid, is_scored rows)"
            ),
            mo.ui.table(out, selection=None),
        ]
    )
    return


@app.cell
def _(mo):
    import os as _os

    txt = (
        open("REPORT.txt").read()
        if _os.path.exists("REPORT.txt")
        else "(REPORT.txt not written yet)"
    )
    mo.md("## REPORT.txt\n```\n" + txt + "\n```")
    return


@app.cell
def _(HERE, json, mo):
    from pathlib import Path as _Path

    _root = _Path(HERE)
    _state = json.loads((_root / "STATE.json").read_text())
    _hypothesis = json.loads((_root / "HYPOTHESES.json").read_text())
    _result_path = _root / "clock_probe_results.json"
    _status = (
        "Measured clock probe results are available below."
        if _result_path.exists()
        else "UNSCORED: inspect STATE.json and the preserved probe logs for status."
    )
    _items = [
        mo.md("## Bounded continuation: filter-clock falsifier"),
        mo.md(_status),
        mo.md(_hypothesis["hypothesis"]),
        mo.md("Matched control: " + _hypothesis["matched_control"]),
        mo.md("Frozen decision: " + _hypothesis["decision"]),
        mo.md("Sample exposure: " + _state["independence"]),
    ]
    if _result_path.exists():
        _results = json.loads(_result_path.read_text())
        _items.append(
            mo.ui.table(
                [dict(clock=_name, **_values) for _name, _values in _results["configs"].items()],
                selection=None,
            )
        )
        _items.append(
            mo.md(
                f"Own minus shared: {_results['own_minus_shared']:.5f}; "
                f"frozen acceptance passed: {_results['supports_own_clock']}. "
                "Uncertainty is descriptive; connected-component dependence is unresolved."
            )
        )
        for _figure in ("fig_clock_probe.png", "fig_clock_activity.png"):
            if (_root / _figure).exists():
                _items.append(mo.image(src=str(_root / _figure)))
    _items.append(
        mo.md("Recovery summary:\n```\n" + (_root / "RECOVERY_SUMMARY.txt").read_text() + "\n```")
    )
    mo.vstack(_items)
    return


if __name__ == "__main__":
    app.run()
