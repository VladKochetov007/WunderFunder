"""Linear continuation: read saved results without refitting or scanning data.

Original notebook preserved as eda_linear_prior.py. Export headlessly through
continuation/run_cpu.sh eda-linear .venv/bin/marimo export html <file> -o <own html>.
"""
import marimo

__generated_with = "0.25.0"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    from pathlib import Path
    import marimo as mo
    OUT = Path("/home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-03/eda/linear")
    results = json.loads((OUT / "cadence_results.json").read_text())
    manifest = json.loads((OUT / "cadence_manifest.json").read_text())
    hypotheses = json.loads((OUT / "HYPOTHESES.json").read_text())
    parity = json.loads((OUT / "cadence_parity.json").read_text())
    return OUT, hypotheses, manifest, mo, parity, results


@app.cell
def _(mo):
    mo.md("""
    # Linear analyst: temporal memory versus cadence

    Prior levels + causal EMA32/256 scored 0.6484 on 300 repeatedly inspected validation
    sequences. This continuation freezes a smaller matched fitting experiment and a new
    validation sample disjoint from recoverable own prior samples. It tests whether
    same-phase filtering preserves the gain and whether period8 beats a period7 placebo.

    Every input transform is causal, updates on all rows and resets per sequence.
    Targets, scoring mask, quantum classes and future sequence statistics are never inputs.
    """)
    return


@app.cell
def _(manifest, mo, results):
    mo.md(f"""
    Fitting: **{results['training_sequences']} already-exposed training sequences**,
    {results['training_rows']:,} selected rows, stride31, lambda0.001 without tuning.
    Evaluation: **{results['evaluation_sequences']} sequences**, {results['evaluation_scored_rows']:,}
    scored rows; excluded {manifest['prior_valid_union_count']} recoverable prior validation
    sequences. Model selection never uses this evaluation sample.

    The sample is own-disjoint, with unrecorded predecessor/campaign exposure unknown.
    Sequence contributions and leave-one-out ranges are descriptive, not confidence
    intervals based on campaign connected components. No strong baseline was inspected.
    """)
    return


@app.cell
def _(mo, results):
    _rows = [{"model": _name, **_score} for _name, _score in results["scores"].items()]
    mo.ui.table(_rows, selection=None)
    return


@app.cell
def _(OUT, mo):
    mo.image(str(OUT / "fig05_cadence_comparison.png"))
    return


@app.cell
def _(OUT, mo):
    mo.image(str(OUT / "fig06_cadence_prediction_example.png"))
    return


@app.cell
def _(OUT, mo):
    import json as _context_json
    _context = _context_json.loads((OUT / "cadence_context_results.json").read_text())
    mo.md(f"""
    The archived 400-sequence EMA model scores **{_context['scores']['L+E__lam0.001__w']['wp']:.5f}**
    on these same64 sequences, versus **0.64774** on its old300 development sample.
    The gap is not explained solely by fitting128 rather than400 sequences.
    This follow-up reused the now-exposed64-sequence sample and was frozen separately
    in CONTEXT_HYPOTHESIS.json; it is not independent confirmation.
    """)
    return


@app.cell
def _(OUT, mo):
    mo.image(str(OUT / "fig07_archived_model_stability.png"))
    return


@app.cell
def _(mo, parity, results):
    mo.md(f"""
    Frozen-rule outcomes:

    - Phase8 retains **{results['phase8_ema_gain_retention']:.1%}** of the EMA gain over levels.
    - Reject cadence-only account by frozen rule: **{results['H1_cadence_only_rejected_by_frozen_rule']}**.
    - Reject phase8 as a practical replacement: **{results['H1_phase8_replacement_rejected_by_frozen_rule']}**.
    - Cadence-averaging dependence suggested: **{results['H1_cadence_averaging_dependence_suggested_by_frozen_rule']}**.
    - Phase8 minus phase7 pooled WP: **{results['H2_phase8_minus_phase7']:+.5f}**.

    Online/batch maximum feature discrepancy:
    **{max(v['online_max_abs_error'] for v in parity.values()):.3g}**;
    future-extension discrepancy:
    **{max(v['future_extension_max_abs_error'] for v in parity.values()):.3g}**.

    Equal temporal decay and coefficient count do not equalize sampling support or
    response lag. These tests can reject a useful premise; they cannot uniquely establish
    an economic mechanism. Inference requires only input rows, EMA states and a short ring buffer.
    """)
    return


@app.cell
def _(hypotheses, manifest, mo):
    mo.accordion({
        "Frozen hypotheses": mo.json(hypotheses),
        "Exact sampled row groups and seq_ix identifiers": mo.json(manifest),
    })
    return


@app.cell
def _(OUT, mo):
    mo.md((OUT / "RECOVERY_SUMMARY.txt").read_text())
    return


if __name__ == "__main__":
    app.run()
