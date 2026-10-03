from __future__ import annotations

import marimo

app = marimo.App(width="medium")


@app.cell
def _() -> tuple[object, ...]:
    import json
    from pathlib import Path

    import marimo as mo

    owned_dir = Path(__file__).resolve().parent
    results = json.loads((owned_dir / "s30_results.json").read_text())
    blocks = json.loads((owned_dir / "s31_results.json").read_text())
    prices = json.loads((owned_dir / "s32_results.json").read_text())
    provenance = json.loads((owned_dir / "s30_provenance.json").read_text())
    return blocks, mo, owned_dir, prices, provenance, results


@app.cell
def _(mo: object, results: dict) -> None:
    mo.md(
        f"""
        # Olympiad continuation: invariants versus prediction rules

        The predecessor found four target lattices and an input clock whose
        resets cover all target changes. Those exact observations do not imply
        that a prediction should remain constant inside the same interval.

        The frozen controlled probe scored **{results["valid_scored_rows"]:,}**
        eligible rows in **20 valid sequences**, with 32 separate train sequences.
        Every sampled sequence was already exposed by predecessor exhaustive
        checks. These are exploratory diagnostic scores, with no strong-baseline
        comparison and no public-score forecast.

        Current inputs scored **{results["scores"]["current"]["mean_wp"]:.4f}**;
        carrying the same predictor from `a6` resets scored
        **{results["scores"]["carry_a6"]["mean_wp"]:.4f}**.
        Current revision won on **{results["sequence_win_fraction"]:.0%}** of
        sampled sequences. Hard persistence is rejected for this ridge probe.
        """
    )


@app.cell
def _(blocks: dict, mo: object, prices: dict, results: dict) -> None:
    _rows = []
    for _probe, _payload in [
        ("frozen anchor probe", results),
        ("adaptive block ablation", blocks),
        ("adaptive p/dp ablation", prices),
    ]:
        for _name, _scores in _payload["scores"].items():
            _rows.append({"probe": _probe, "model": _name, **_scores})
    mo.ui.table(_rows, selection=None, pagination=False)


@app.cell
def _(blocks: dict, mo: object, prices: dict, results: dict) -> None:
    mo.md(
        f"""
        The fixed `< -5.19` reset detector missed
        **{results["h2_threshold_misses"]}** target transitions and produced
        **{results["h2_threshold_extras"]}** extra resets in this bounded sample.
        Price-change anchors missed **{results["price_anchor_misses"]}** transitions.
        This audit uses targets offline; they never enter predictions.

        Updating clocks recovered **{blocks["clock_recovery_fraction"]:.1%}** of
        the anchor-to-current score gap. Updating the six `v/dv` representatives
        changed mean WP by **{blocks["flow_gain_over_anchor"]:+.4f}**. Both frozen
        follow-up hypotheses failed for this linear feature set.

        Updating `p/dp` recovered **{prices["anchor_pdp_recovery_fraction"]:.1%}**
        of that gap; holding them fixed cost
        **{prices["holding_pdp_loss"]:.4f}** WP.
        This concerns encoded input representations; column economics and exact
        invertibility remain unproven.

        Prefix recomputation passed for true, price and five delayed reset streams
        at three cutoffs on train 104 and valid 6. Normalizers use training rows
        only. No oracle regime, future minimum, completed duration, or target
        feature is used. Sequence bootstrap summaries are descriptive because
        campaign connected components are unavailable.
        """
    )


@app.cell
def _(mo: object, owned_dir: object) -> None:
    mo.vstack(
        [
            mo.image((owned_dir / _name).read_bytes(), width="100%")
            for _name in [
                "f06_anchor_probe.png",
                "f07_anchor_predictions.png",
                "f08_revision_blocks.png",
                "f09_price_revision.png",
            ]
        ]
    )


@app.cell
def _(mo: object, provenance: dict) -> None:
    mo.md("All sampled row groups and sequence IDs; every row is previously exposed.")
    mo.ui.table(provenance["sequence_ids"], selection=None, pagination=False)


@app.cell
def _(mo: object, owned_dir: object) -> None:
    mo.md(
        f"""
        Reproduce under the continuation resource contract, from repository root:

        ```bash
        research/campaign_2026-10-03/continuation/run_cpu.sh eda-olympiad .venv/bin/python -B {owned_dir}/s30_anchor_probe.py
        research/campaign_2026-10-03/continuation/run_cpu.sh eda-olympiad .venv/bin/python -B {owned_dir}/s31_revision_blocks.py
        research/campaign_2026-10-03/continuation/run_cpu.sh eda-olympiad .venv/bin/python -B {owned_dir}/s32_price_revision.py
        ```

        Test specifications were frozen before their scores in `HYPOTHESES.json`,
        `HYPOTHESES_s31.json`, and `HYPOTHESES_s32.json`. Source/specification hashes,
        dataset metadata, sequence IDs, scores and fitted coefficients are saved
        alongside this notebook. The notebook reads saved artifacts and does not
        refit models or rescan datasets.
        """
    )


if __name__ == "__main__":
    app.run()
