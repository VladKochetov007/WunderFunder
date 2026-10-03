from __future__ import annotations

import marimo

app = marimo.App(width="full")


@app.cell
def _():
    import json
    from pathlib import Path

    import marimo as mo

    directory = Path(__file__).resolve().parent
    result = json.loads((directory / "continuation_results.json").read_text())
    events = json.loads((directory / "continuation_events_results.json").read_text())
    return directory, events, mo, result


@app.cell
def _(mo, result):
    mo.md(
        f"""# Puzzle A: current-state falsifier

        Fixed train groups 1600–1611; valid groups 900–911. All previously
        exposed to target-summary scans. Current input only; labels never
        enter features. Frozen hypotheses and sampled hashes accompany results.

        **Measured negative:** sum ridge scored-row R²
        {result["metrics"]["aligned"]["sum_r2"]:.3f}, shifted-label control
        {result["metrics"]["shifted_control"]["sum_r2"]:.3f}.
        This rejects this particular transfer model and does not establish
        the exact target construction or any strong-baseline gain.

        Prefix consistency: {result["prefix_checks_pass"]}.
        Training rows: {result["train_fit_rows"]}; scored validation rows:
        {result["valid_scored_rows"]}. No row-independent uncertainty claims.
        """
    )
    return


@app.cell
def _(mo, result):
    mo.ui.table(
        [
            {
                "split": row["split"],
                "row_group": row["row_group"],
                "seq_ix": row["seq_ix"],
                "sum_state_disagreement": row["sum_state"]["modal_disagreement"],
                "difference_state_disagreement": row["difference_state"]["modal_disagreement"],
                "shifted_sum_disagreement": row["shifted_sum_state"]["modal_disagreement"],
                "quantum_diagnostic_only": row["diagnostic_quantum"],
            }
            for row in result["records"]
        ]
    )
    return


@app.cell
def _(directory, mo):
    mo.vstack(
        [
            mo.image(str(directory / "continuation_falsifier.png")),
            mo.image(str(directory / "continuation_events.png")),
        ]
    )
    return


@app.cell
def _(events, mo):
    mo.ui.table(
        [
            {
                "split": row["split"],
                "seq_ix": row["seq_ix"],
                "unchanged_all_input_rows": row["aligned"]["unchanged_all_inputs"]["rows"],
                "sum_changes_without_input_changes": row["aligned"]["unchanged_all_inputs"][
                    "sum_changed"
                ],
                "sum_changes_without_price_changes": row["aligned"]["unchanged_i0_prices"][
                    "sum_changed"
                ],
                "cancelling_changes_without_price_changes": row["aligned"]["unchanged_i0_prices"][
                    "cancelling_target_changes"
                ],
            }
            for row in events["records"]
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        """Reproduce from the repository root, with CUDA disabled by the wrapper:

        ```bash
        research/campaign_2026-10-03/continuation/run_cpu.sh eda-puzzle-a .venv/bin/python research/campaign_2026-10-03/eda/puzzle_a/continuation_probe.py
        research/campaign_2026-10-03/continuation/run_cpu.sh eda-puzzle-a .venv/bin/python research/campaign_2026-10-03/eda/puzzle_a/continuation_events.py
        research/campaign_2026-10-03/continuation/run_cpu.sh eda-puzzle-a .venv/bin/marimo export html research/campaign_2026-10-03/eda/puzzle_a/eda_puzzle_a.py -o research/campaign_2026-10-03/eda/puzzle_a/eda_puzzle_a.html
        ```

        Circular shift is a deliberately noncausal diagnostic control, never
        a predictor input. Exact-window equality screens only selected prices;
        it cannot establish independent source sequences. Group uncertainty
        awaits controller connected-component checks. Frozen files retain
        pre-score definitions; the event test was an explicitly adaptive test.
        """
    )
    return


if __name__ == "__main__":
    app.run()
