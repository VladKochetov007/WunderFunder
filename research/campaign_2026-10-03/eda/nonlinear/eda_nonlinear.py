from __future__ import annotations

import marimo

__generated_with = "0.25.0"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    from pathlib import Path

    import marimo as mo

    own_dir = Path(__file__).resolve().parent
    results = json.loads((own_dir / "results_offsets_v1.json").read_text())
    return mo, own_dir, results


@app.cell(hide_code=True)
def _(mo, own_dir):
    mo.md(
        "## Nonlinear continuation: causal residual offsets\n\n"
        "The original notebook is preserved as eda_nonlinear_precontinuation.py. "
        "This notebook reads the completed frozen CPU probe; it performs no fits. "
        "RECOVERY_SUMMARY.txt preserves the predecessor's evidence and caveats.\n\n"
        + (own_dir / "RECOVERY_SUMMARY.txt").read_text()
    )
    return


@app.cell(hide_code=True)
def _(mo, results):
    _lines = [
        "|Calibration source|Model|WP t0|WP t1|Mean WP|Delta mean vs affine|",
        "|---|---|---:|---:|---:|---:|",
    ]
    for _record in results["scores"]:
        _lines.append(
            f"|{_record['source']}|{_record['model']}|"
            f"{_record['wp_t0']:.6f}|{_record['wp_t1']:.6f}|"
            f"{_record['wp_average']:.6f}|"
            f"{_record['delta_average_vs_affine']:+.6f}|"
        )
    mo.md("\n".join(_lines))
    return


@app.cell(hide_code=True)
def _(mo, own_dir):
    mo.vstack(
        [
            mo.image(own_dir / "fig09_offsets_controls_v1.png"),
            mo.image(own_dir / "fig10_offset_coefficients_v1.png"),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo, own_dir, results):
    _sampling = (own_dir / "sampling_v1.json").read_text()
    _checks = (own_dir / "causality_checks_v1.json").read_text()
    mo.md(
        "## Frozen verdicts and provenance\n\n"
        f"Evaluation scored rows: {results['evaluation_scored_rows']}. "
        "Disjoint from known own samples and reconstructed default stream600; "
        "the actual old stream invocation and campaign exposure remain unknown. "
        + results["uncertainty_caveat"]
        + "\n\nVerdicts:\n\n```\n"
        + str(results["verdicts"])
        + "\n```\n\nCausality checks:\n\n```json\n"
        + _checks
        + "\n```\n\nSampling manifest:\n\n```json\n"
        + _sampling
        + "\n```"
    )
    return


if __name__ == "__main__":
    app.run()
