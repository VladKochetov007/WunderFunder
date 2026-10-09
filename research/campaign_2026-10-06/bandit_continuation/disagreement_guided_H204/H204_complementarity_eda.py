import marimo

__generated_with = "0.25.0"
app = marimo.App(width="full")


@app.cell
def _():
    import csv
    import json
    import sys
    from pathlib import Path

    import marimo as mo

    project_root = Path.cwd()
    current_dir = Path(__file__).resolve().parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    import matplotlib.pyplot as plt
    import numpy as np
    from disagreement_screen import (
        E515_DIR,
        association_screen,
        chain_diagnostics,
        load_e514_inputs,
    )
    from simplex_stack import E516_RELATIVE_DIR, SOURCE_ORDER, fit_and_crossfit

    return (
        E515_DIR,
        E516_RELATIVE_DIR,
        SOURCE_ORDER,
        association_screen,
        chain_diagnostics,
        csv,
        fit_and_crossfit,
        json,
        load_e514_inputs,
        mo,
        np,
        plt,
        project_root,
    )


@app.cell
def _(load_e514_inputs, mo, project_root):
    inputs = load_e514_inputs(project_root)
    source_status = mo.md(
        f"""
        **Frozen source checks passed.** Manifest `{inputs.source_manifest_sha256}`;
        chain partition `{inputs.chain_partition_sha256}`; {len(inputs.row_groups)}
        groups across {len(set(inputs.chain_by_sequence.tolist()))} complete chains.
        All inputs are from previously inspected fold 0.
        """
    )
    return inputs, source_status


@app.cell
def _(
    E515_DIR,
    association_screen,
    chain_diagnostics,
    csv,
    inputs,
    json,
    mo,
    np,
    plt,
    project_root,
):
    _candidate_scales = {"ab12": 1.0, "ab48": 1.0, "baseline": 0.25}
    diagnostics = chain_diagnostics(
        inputs.targets,
        inputs.is_scored,
        inputs.anchor,
        inputs.candidate_predictions,
        _candidate_scales,
        inputs.chain_by_sequence,
        inputs.chain_to_outer_fold,
        np.arange(inputs.targets.shape[0]),
    )
    screen = association_screen(
        diagnostics,
        seed=2_985_631_662,
        bootstrap_replicates=2_000,
        permutation_replicates=5_000,
    )
    e515_result = {
        "experiment_id": "E-515",
        "hypothesis_id": "H-204",
        "metric": "weighted_squared_error_surrogate_only",
        "data_exposure": "previously inspected fold 0; exploratory, not confirmation",
        "source_manifest_sha256": inputs.source_manifest_sha256,
        "chain_partition_sha256": inputs.chain_partition_sha256,
        "sample_groups": int(inputs.targets.shape[0]),
        "chains": int(len(set(inputs.chain_by_sequence.tolist()))),
        "candidate_scales": _candidate_scales,
        "bootstrap_replicates": 2_000,
        "permutation_replicates": 5_000,
        "seed": 2_985_631_662,
        "comparisons": screen,
        "screen_passed": any(row["screen_gate_pass"] for row in screen),
        "interpretation": "Association is not evidence that guided training improves the official endpoint.",
        "new_public_gain_verified": False,
        "official_metric_score": None,
        "expected_public_gain": None,
        "public_forecast": None,
        "numerical_prior_update": None,
    }
    _output_directory = project_root / E515_DIR
    plot_directory = _output_directory / "plots"
    plot_directory.mkdir(parents=True, exist_ok=True)
    (_output_directory / "E515_RESULT.json").write_text(
        json.dumps(e515_result, indent=2, sort_keys=True) + "\n"
    )
    with (_output_directory / "E515_CHAIN_DIAGNOSTICS.csv").open("w", newline="") as _stream:
        _writer = csv.DictWriter(_stream, fieldnames=list(diagnostics[0]))
        _writer.writeheader()
        _writer.writerows(diagnostics)

    _candidates = sorted({str(_row["candidate"]) for _row in screen})
    _targets = sorted({int(_row["target"]) for _row in screen})
    _fig_a, _axes_a = plt.subplots(len(_candidates), len(_targets), figsize=(11, 13), squeeze=False)
    for _row_index, _candidate in enumerate(_candidates):
        for _column_index, _target in enumerate(_targets):
            _ax = _axes_a[_row_index, _column_index]
            _selected = [
                row
                for row in diagnostics
                if row["candidate"] == _candidate and row["target"] == _target
            ]
            for _fold in sorted({int(_row["outer_fold"]) for _row in _selected}):
                _fold_rows = [_row for _row in _selected if int(_row["outer_fold"]) == _fold]
                _ax.scatter(
                    [_row["disagreement"] for _row in _fold_rows],
                    [_row["anchor_weighted_squared_error"] for _row in _fold_rows],
                    s=20,
                    alpha=0.72,
                    label=f"outer fold {_fold}",
                )
            _ax.set_title(f"{_candidate}, target {_target}")
            _ax.set_xlabel("weighted squared disagreement")
            _ax.set_ylabel("anchor weighted squared error")
            if _row_index == 0 and _column_index == 0:
                _ax.legend(fontsize=8)
    _fig_a.suptitle("E-515: disagreement vs anchor error by connected chain")
    _fig_a.tight_layout()
    _fig_a.savefig(plot_directory / "disagreement_vs_anchor_error.png", dpi=170)
    plt.close(_fig_a)

    _excess_rows = [_row for _row in screen]
    _fig_b, _ax_b = plt.subplots(figsize=(12, 5))
    _labels = [f"{_row['candidate']} / T{_row['target']}" for _row in _excess_rows]
    _centers = [float(_row["top_quartile_error_excess"]) for _row in _excess_rows]
    _intervals = [_row["top_quartile_error_excess_ci95"] for _row in _excess_rows]
    _errors = np.asarray(
        [
            [center - float(interval[0]), float(interval[1]) - center]
            for center, interval in zip(_centers, _intervals, strict=True)
        ]
    ).T
    _ax_b.errorbar(np.arange(len(_labels)), _centers, yerr=_errors, fmt="o", capsize=3)
    _ax_b.axhline(0.0, color="black", linewidth=1)
    _ax_b.set_xticks(np.arange(len(_labels)), _labels, rotation=40, ha="right")
    _ax_b.set_ylabel("top-quartile minus remaining anchor error")
    _ax_b.set_title("E-515: chain-bootstrap error excess, exploratory")
    _fig_b.tight_layout()
    _fig_b.savefig(plot_directory / "disagreement_top_quartile_error_excess.png", dpi=170)
    plt.close(_fig_b)
    screen_table = mo.ui.table(screen)
    return (screen_table,)


@app.cell
def _(
    E516_RELATIVE_DIR,
    SOURCE_ORDER,
    fit_and_crossfit,
    inputs,
    json,
    mo,
    np,
    plt,
    project_root,
):
    import sys as _sys
    _sys.path.insert(0, str(project_root / "src"))
    e516_result, oof_predictions = fit_and_crossfit(inputs)
    e516_directory = project_root / E516_RELATIVE_DIR
    e516_directory.mkdir(parents=True, exist_ok=True)
    (e516_directory / "E516_RESULT.json").write_text(
        json.dumps(e516_result, indent=2, sort_keys=True) + "\n"
    )
    np.save(e516_directory / "E516_OOF_PREDICTIONS.npy", oof_predictions, allow_pickle=False)
    e516_plot_directory = e516_directory / "plots"
    e516_plot_directory.mkdir(parents=True, exist_ok=True)
    _fold_records = e516_result["folds"]
    _fold_labels = [f"fold {_row['outer_fold']}" for _row in _fold_records]
    _weights_by_target = []
    for _target in range(2):
        _weights_by_target.append(
            np.asarray(
                [_record["fits"][_target]["weights"] for _record in _fold_records],
                dtype=np.float64,
            )
        )
    _fig_a, _axes_a = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for _target, _ax in enumerate(_axes_a):
        _bottom = np.zeros(len(_fold_records))
        for _source_index, _source in enumerate(SOURCE_ORDER):
            _values = _weights_by_target[_target][:, _source_index]
            _ax.bar(_fold_labels, _values, bottom=_bottom, label=_source)
            _bottom += _values
        _ax.set_title(f"Target {_target}")
        _ax.set_ylim(0, 1)
        _ax.set_ylabel("calibration-only simplex weight")
    _axes_a[0].legend(fontsize=8)
    _fig_a.suptitle("E-516: target-wise weights fit only on other chain folds")
    _fig_a.tight_layout()
    _fig_a.savefig(e516_plot_directory / "targetwise_simplex_weights.png", dpi=170)
    plt.close(_fig_a)

    _interval = e516_result["paired_chain_bootstrap_delta_95pct"]
    _delta = e516_result["delta_vs_anchor"]
    _fig_b, _axes_b = plt.subplots(1, 2, figsize=(10, 4))
    _axes_b[0].bar(
        ["C1 anchor", "cross-fitted simplex"],
        [e516_result["anchor"]["wp"], e516_result["simplex_stack"]["wp"]],
    )
    _axes_b[0].set_ylabel("global_wp_clip2_v1")
    _axes_b[0].set_title("OOF point scores")
    _axes_b[1].errorbar(
        ["simplex minus anchor"],
        [_delta],
        yerr=[[_delta - _interval[0]], [_interval[1] - _delta]],
        fmt="o",
        capsize=4,
    )
    _axes_b[1].axhline(0.0, color="black", linewidth=1)
    _axes_b[1].set_ylabel("paired OOF score difference")
    _axes_b[1].set_title("Chain-bootstrap interval; conditional")
    _fig_b.suptitle("E-516: exposed development chains")
    _fig_b.tight_layout()
    _fig_b.savefig(e516_plot_directory / "simplex_stack_vs_anchor.png", dpi=170)
    plt.close(_fig_b)
    e516_summary = mo.md(
        f"""
        **E-516 OOF endpoint** — anchor `{e516_result["anchor"]["wp"]:.6f}`;
        target-wise simplex `{e516_result["simplex_stack"]["wp"]:.6f}`;
        delta `{_delta:+.6f}`; paired chain-bootstrap interval
        `[{_interval[0]:+.6f}, {_interval[1]:+.6f}]`.

        Positive donor weight is a fitted diagnostic. This remains exposed-fold
        development evidence, not a public-score forecast or confirmation.
        """
    )
    return e516_summary, oof_predictions


@app.cell(hide_code=True)
def _(E516_RELATIVE_DIR, inputs, mo, np, oof_predictions, plt, project_root):
    import csv as _csv
    import sys as _sys
    _sys.path.insert(0, str(project_root / "src"))
    from wnn.gmetric import global_wp as _global_wp

    _e516_chain_summary_rows = []
    for _chain in sorted(int(value) for value in np.unique(inputs.chain_by_sequence)):
        _sequence_indices = np.flatnonzero(inputs.chain_by_sequence == _chain)
        _anchor_metric = _global_wp(
            inputs.targets[_sequence_indices],
            inputs.anchor[_sequence_indices],
            inputs.is_scored[_sequence_indices],
        )
        _stack_metric = _global_wp(
            inputs.targets[_sequence_indices],
            oof_predictions[_sequence_indices],
            inputs.is_scored[_sequence_indices],
        )
        _e516_chain_summary_rows.append({
            "chain_id": _chain,
            "outer_fold": int(inputs.chain_to_outer_fold[_chain]),
            "sequence_count": int(len(_sequence_indices)),
            "selected_rows": int(_anchor_metric["selected_rows"]),
            "anchor_t0_wp": float(_anchor_metric["t0"]),
            "stack_t0_wp": float(_stack_metric["t0"]),
            "delta_t0_wp": float(_stack_metric["t0"] - _anchor_metric["t0"]),
            "anchor_t1_wp": float(_anchor_metric["t1"]),
            "stack_t1_wp": float(_stack_metric["t1"]),
            "delta_t1_wp": float(_stack_metric["t1"] - _anchor_metric["t1"]),
            "anchor_chain_wp": float(_anchor_metric["wp"]),
            "stack_chain_wp": float(_stack_metric["wp"]),
            "delta_chain_wp": float(_stack_metric["wp"] - _anchor_metric["wp"]),
        })
    _e516_summary_path = project_root / E516_RELATIVE_DIR / "plots" / "E516_CHAIN_SUMMARY.csv"
    _e516_summary_path.parent.mkdir(parents=True, exist_ok=True)
    with _e516_summary_path.open("w", newline="") as _stream:
        _writer = _csv.DictWriter(_stream, fieldnames=list(_e516_chain_summary_rows[0]))
        _writer.writeheader()
        _writer.writerows(_e516_chain_summary_rows)

    _fig, _axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    for _target, _ax in enumerate(_axes):
        _delta_key = f"delta_t{_target}_wp"
        _ordered = sorted(_e516_chain_summary_rows, key=lambda _row: _row[_delta_key])
        _x = np.arange(len(_ordered))
        _y = np.asarray([_row[_delta_key] for _row in _ordered])
        _fold = np.asarray([_row["outer_fold"] for _row in _ordered])
        _scatter = _ax.scatter(_x, _y, c=_fold, cmap="viridis", s=18, alpha=0.8)
        _ax.axhline(0, color="black", linewidth=1)
        _ax.set_title(f"Target {_target}")
        _ax.set_xlabel("chains sorted by within-chain WP delta")
        _ax.set_ylabel("stack minus anchor chain WP")
    _fig.colorbar(_scatter, ax=_axes, label="frozen outer fold")
    _fig.suptitle("E-516 per-chain diagnostics (descriptive; WP is non-additive)")
    _fig.subplots_adjust(top=0.82, bottom=0.16, wspace=0.16)
    _fig.savefig(_e516_summary_path.with_name("simplex_per_chain_delta.png"), dpi=170)
    plt.close(_fig)
    e516_chain_summary_note = mo.md(
        "Per-chain WP is a descriptive heterogeneity view; it is non-additive and "
        "does not replace the pooled official endpoint. Saved CSV: "
        f"`{_e516_summary_path.relative_to(project_root)}`; plot: "
        f"`{_e516_summary_path.with_name('simplex_per_chain_delta.png').relative_to(project_root)}`."
    )
    e516_chain_summary_table = mo.ui.table(_e516_chain_summary_rows)

    return e516_chain_summary_note, e516_chain_summary_table


@app.cell
def _(
    e516_chain_summary_note,
    e516_chain_summary_table,
    e516_summary,
    mo,
    screen_table,
    source_status,
):
    mo.vstack([source_status, screen_table, e516_summary, e516_chain_summary_note, e516_chain_summary_table])
    return






if __name__ == "__main__":
    app.run()
