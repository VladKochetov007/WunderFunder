from __future__ import annotations

import marimo

app = marimo.App(width="full")


@app.cell
def _():
    import json
    import sys
    from pathlib import Path

    import marimo as mo
    import pandas as pd

    own = Path(__file__).resolve().parent
    sys.path.insert(0, str(own))
    from render_continuation import render_all

    return json, mo, own, pd, render_all


@app.cell
def _(mo):
    mo.md("""
    # HFT continuation: quote identities, causal sign and scale

    These are frozen small probes of the analyst's existing decoder. Physical
    quote reconstruction is an assumption under test. Samples are disjoint from
    this analyst's documented prior validation exposure; campaign-wide exposure
    and cross-sequence dependence are unknown. No campaign baseline was read.

    This notebook reads saved results and regenerates PNGs only. Raw data probes
    are separate scripts and must run through the continuation CPU wrapper.
    """)


@app.cell
def _(json, own, pd):
    identity = json.loads((own / "quote_identity_summary_v1.json").read_text())
    sign_scale = json.loads((own / "causal_sign_scale_summary_v1.json").read_text())
    ablation = json.loads((own / "spread_ablation_summary_v1.json").read_text())
    sequence_table = pd.read_csv(own / "quote_identity_v1.csv")
    return ablation, identity, sequence_table, sign_scale


@app.cell
def _(identity, mo, sequence_table, sign_scale):
    mo.vstack(
        [
            mo.md(f"""
        H1 rejects exact shared-reference spread identities in
        **{identity["shared_reference_spread_rejected_sequences"]}/20 sequences**.
        Negative implied spread occupies **{identity["negative_spread_fraction"]:.1%}**
        of rows. Transition modal agreement is **{identity["transition_purity"]:.1%}**,
        versus **{identity["transition_control"]:.1%}** after a matched shuffle.

        H2 uses input-only predictions and a classifier fitted only to 300 saved
        TRAIN warm-up prefixes. Quantum classification accuracy is
        **{sign_scale["warmup_class_accuracy"]:.1%}** on 20 own-unseen sequences.
        Full/prefix decoder and EMA discrepancy:
        **{sign_scale["causal_prefix_max_error"]}**. Oracle target scales never
        enter these predictions.
        """),
            mo.ui.table(sequence_table),
        ]
    )


@app.cell
def _(mo, render_all):
    figure_paths = render_all()
    mo.vstack([mo.image(str(path), caption=path.name) for path in figure_paths])
    return (figure_paths,)


@app.cell
def _(ablation, mo, sign_scale):
    mo.vstack(
        [
            mo.md("""
        The largest-quantum regime reverses the pooled momentum sign. Correct
        quantum identification does not establish optimal prediction amplitude.
        H3 compares current-book predictions with constants available at row99
        on a second, separately frozen own-unseen cohort. All scores are pooled
        clipped WP, not averages of per-sequence correlations. No row-level
        significance or baseline improvement is claimed.
        """),
            mo.ui.table(sign_scale["pooled_wp"]),
            mo.ui.table(ablation["pooled_wp"]),
        ]
    )


if __name__ == "__main__":
    app.run()
