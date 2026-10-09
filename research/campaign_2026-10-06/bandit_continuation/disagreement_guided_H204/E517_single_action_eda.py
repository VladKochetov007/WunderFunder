import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def e517_setup():
    import sys
    from pathlib import Path
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    project_root = Path.cwd()
    e517_directory = project_root / "research/campaign_2026-10-06/bandit_continuation/disagreement_guided_H204"
    for import_root in (project_root / "src", e517_directory):
        if str(import_root) not in sys.path:
            sys.path.insert(0, str(import_root))

    return mo, np, plt, project_root


@app.cell(hide_code=True)
def e517_verified_inputs(mo, project_root):
    from disagreement_screen import load_e514_inputs

    inputs = load_e514_inputs(project_root)
    source_status = mo.md(
        f"**Frozen source checks passed.** Manifest `{inputs.source_manifest_sha256}`; "
        f"chain partition `{inputs.chain_partition_sha256}`; "
        f"{len(inputs.row_groups)} groups across "
        f"{len(set(inputs.chain_by_sequence.tolist()))} complete chains. "
        "These inputs are all previously exposed development chains."
    )
    source_status

    return (inputs,)


@app.cell(hide_code=True)
def e517_explicit_trigger(mo):
    e517_run_button = mo.ui.run_button(label="Run the preregistered E-517 screen")
    e517_run_button

    return (e517_run_button,)


@app.cell(hide_code=True)
def e517_gated_blend_registered_action(
    e517_run_button,
    inputs,
    mo,
    np,
    plt,
    project_root,
):
    if not e517_run_button.value:
        _e517_output = mo.md("E-517 is registered and has not run. Use the single visible button above to execute the frozen screen.")
    else:

        import hashlib as _hashlib
        import json as _json
        import os as _os
        from datetime import datetime as _datetime, timezone as _timezone
        from pathlib import Path as _Path

        import sys as _sys
        _sys.path.insert(0, str(project_root / "src"))
        from disagreement_gate import run_registered_e517 as _run_registered_e517

        _e517_directory = project_root / "research/campaign_2026-10-06/bandit_continuation/disagreement_guided_H204"
        _e517_plot_directory = _e517_directory / "plots"
        _e517_result_path = _e517_directory / "E517_RESULT.json"
        _e517_gated_path = _e517_directory / "E517_GATED_OOF_PREDICTIONS.npy"
        _e517_global_path = _e517_directory / "E517_GLOBAL_OOF_PREDICTIONS.npy"
        _e517_plot_path = _e517_plot_directory / "E517_FOLD_GATE_RESULTS.png"
        _e517_attempt_path = _e517_directory / "E517_ATTEMPT_STARTED.json"

        if _e517_attempt_path.exists():
            if not (_e517_result_path.exists() and _e517_gated_path.exists() and _e517_global_path.exists()):
                raise RuntimeError(
                    "E-517 already started without a complete score artifact; do not rerun this frozen ID."
                )
            _e517_result = _json.loads(_e517_result_path.read_text())
            _e517_gated_oof = np.load(_e517_gated_path, mmap_mode="r", allow_pickle=False)
            _e517_global_oof = np.load(_e517_global_path, mmap_mode="r", allow_pickle=False)
        else:
            _e517_directory.mkdir(parents=True, exist_ok=True)
            _e517_fd = _os.open(
                _e517_attempt_path,
                _os.O_WRONLY | _os.O_CREAT | _os.O_EXCL,
                0o644,
            )
            with _os.fdopen(_e517_fd, "w") as _e517_stream:
                _json.dump(
                    {
                        "experiment_id": "E-517",
                        "started_at_utc": _datetime.now(_timezone.utc).isoformat(),
                        "registered_bootstrap_seed": 3772181504,
                        "input_load_mode": "shared verified E514 Marimo input cell",
                        "scientific_fit": False,
                        "public_submission": False,
                    },
                    _e517_stream,
                    indent=2,
                )
                _e517_stream.write("\n")
            _e517_inputs, _e517_result, _e517_gated_oof, _e517_global_oof = _run_registered_e517(
                project_root, inputs
            )

        if not _e517_plot_path.exists():
            _e517_plot_directory.mkdir(parents=True, exist_ok=True)
            _e517_folds = _e517_result["folds"]
            _e517_labels = [f"fold {row['outer_fold']}" for row in _e517_folds]
            _e517_figure, _e517_axes = plt.subplots(1, 2, figsize=(13, 5))
            _e517_x = np.arange(len(_e517_folds))
            _e517_width = 0.24
            for _e517_index, (_e517_key, _e517_label) in enumerate(
                [
                    ("anchor", "C1 anchor"),
                    ("global_ab12", "global ab12"),
                    ("gated_ab12", "gated ab12"),
                ]
            ):
                _e517_values = [float(row[_e517_key]["wp"]) for row in _e517_folds]
                _e517_axes[0].bar(
                    _e517_x + (_e517_index - 1) * _e517_width,
                    _e517_values,
                    _e517_width,
                    label=_e517_label,
                )
            _e517_axes[0].set_xticks(_e517_x, _e517_labels)
            _e517_axes[0].set_ylabel("global_wp_clip2_v1")
            _e517_axes[0].set_title("Held-out fold endpoints")
            _e517_axes[0].legend(fontsize=8)
            _e517_row_mass = [
                float(row["evaluation_gate"]["gated_scored_row_fraction"])
                for row in _e517_folds
            ]
            _e517_weight_mass = [
                float(row["evaluation_gate"]["gated_official_weight_fraction"])
                for row in _e517_folds
            ]
            _e517_axes[1].bar(_e517_x - 0.18, _e517_row_mass, 0.36, label="scored row fraction")
            _e517_axes[1].bar(_e517_x + 0.18, _e517_weight_mass, 0.36, label="official weight fraction")
            _e517_axes[1].set_xticks(_e517_x, _e517_labels)
            _e517_axes[1].set_ylim(0, 1)
            _e517_axes[1].set_ylabel("routed fraction")
            _e517_axes[1].set_title("Realized gate mass")
            _e517_axes[1].legend(fontsize=8)
            _e517_figure.suptitle("E-517: exposed whole-chain cross-fit")
            _e517_figure.tight_layout()
            _e517_figure.savefig(_e517_plot_path, dpi=170)
            plt.close(_e517_figure)

        _e517_plot_digest = _hashlib.sha256(_e517_plot_path.read_bytes()).hexdigest()
        if _e517_result.get("plot_sha256") is not None:
            if _e517_result["plot_sha256"] != _e517_plot_digest:
                raise RuntimeError("Saved E-517 plot differs from the frozen result digest.")
        else:
            _e517_result["plot_path"] = str(_e517_plot_path.relative_to(project_root))
            _e517_result["plot_sha256"] = _e517_plot_digest
            _e517_result["notebook_sha256"] = _hashlib.sha256(_Path(__file__).read_bytes()).hexdigest()
            _e517_result["analysis_completed_at_utc"] = _datetime.now(_timezone.utc).isoformat()
            _e517_result_path.write_text(_json.dumps(_e517_result, indent=2, sort_keys=True) + "\n")

        _e517_fold_table = [
            {
                "outer_fold": int(row["outer_fold"]),
                "anchor_wp": float(row["anchor"]["wp"]),
                "global_ab12_wp": float(row["global_ab12"]["wp"]),
                "gated_ab12_wp": float(row["gated_ab12"]["wp"]),
                "beta_gate": float(row["fit"]["gated_beta"]),
                "beta_global": float(row["fit"]["global_beta"]),
                "threshold": float(row["fit"]["threshold"]),
                "eval_gated_rows": int(row["evaluation_gate"]["gated_scored_row_count"]),
                "eval_scored_rows": int(row["evaluation_gate"]["scored_row_count"]),
                "eval_row_fraction": float(row["evaluation_gate"]["gated_scored_row_fraction"]),
                "eval_official_weight_fraction": float(row["evaluation_gate"]["gated_official_weight_fraction"]),
            }
            for row in _e517_result["folds"]
        ]
        _e517_gate_summary = mo.md(
            f"""
            **E-517 completed on previously exposed development chains.**
            Anchor WP: {_e517_result["anchor"]["wp"]:.6f};
            gated ab12: {_e517_result["gated_ab12"]["wp"]:.6f};
            global ab12 control: {_e517_result["global_ab12"]["wp"]:.6f}.
            Gated minus anchor: {_e517_result["delta_gated_vs_anchor"]:+.6f};
            95% chain-bootstrap CI
            {_e517_result["paired_chain_bootstrap_delta_gated_vs_anchor_ci95"]}.
            Gated minus global: {_e517_result["delta_gated_vs_global"]:+.6f};
            95% CI {_e517_result["paired_chain_bootstrap_delta_gated_vs_global_ci95"]}.
            Registered screen: {"PASS" if _e517_result["development_gate_passed"] else "FAIL"}.
            This is an exposed local screen, not a public score claim.
            Saved plot: {_e517_result["plot_path"]}.
            """
        )
        _e517_output = mo.vstack([_e517_gate_summary, mo.ui.table(_e517_fold_table)])
    _e517_output
    return


if __name__ == "__main__":
    app.run()
