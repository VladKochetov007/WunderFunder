import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def e518_setup():
    import sys
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    project_root = Path.cwd().resolve()
    lineage = project_root / "research/campaign_2026-10-06/bandit_continuation/disagreement_guided_H204"
    for import_root in (project_root / "src", lineage):
        if str(import_root) not in sys.path:
            sys.path.insert(0, str(import_root))
    return mo, np, plt, project_root, lineage


@app.cell(hide_code=True)
def e518_verified_inputs(mo, project_root):
    from disagreement_screen import load_e514_inputs

    inputs = load_e514_inputs(project_root)
    source_status = mo.md(
        f"**Frozen E-514 source checks passed.** Manifest `{inputs.source_manifest_sha256}`; "
        f"partition `{inputs.chain_partition_sha256}`; {len(inputs.row_groups)} groups, "
        f"{len(set(inputs.chain_by_sequence.tolist()))} complete chains. "
        "All groups are previously exposed development data."
    )
    source_status  # noqa: B018 - Marimo's final expression renders the audit status.
    return (inputs,)


@app.cell(hide_code=True)
def e518_explicit_trigger(mo):
    e518_run_button = mo.ui.run_button(label="Run the preregistered E-518 metric-aligned gate screen")
    e518_run_button  # noqa: B018 - Marimo's final expression renders the run control.
    return (e518_run_button,)


@app.cell(hide_code=True)
def e518_registered_action(e518_run_button, inputs, lineage, mo, np, plt, project_root):
    if not e518_run_button.value:
        e518_output = mo.md(
            "E-518 is frozen and paused. Use the visible button once after the independent pre-score review."
        )
    else:
        import hashlib
        import json
        import os
        from datetime import UTC, datetime
        from pathlib import Path

        result_path = lineage / "E518_RESULT.json"
        gated_path = lineage / "E518_GATED_OOF_PREDICTIONS.npy"
        global_path = lineage / "E518_GLOBAL_OOF_PREDICTIONS.npy"
        plot_path = lineage / "plots/E518_METRIC_ALIGNED_GATE.png"
        attempt_path = lineage / "E518_ATTEMPT_STARTED.json"

        if attempt_path.exists():
            if not (result_path.is_file() and gated_path.is_file() and global_path.is_file()):
                raise RuntimeError("E-518 has a prior attempt marker without complete results; do not rerun this ID.")
            result = json.loads(result_path.read_text(encoding="utf-8"))
            gated_oof = np.load(gated_path, mmap_mode="r", allow_pickle=False)
            global_oof = np.load(global_path, mmap_mode="r", allow_pickle=False)
        else:
            marker = {
                "experiment_id": "E-518",
                "started_at_utc": datetime.now(UTC).isoformat(),
                "preregistration": "METRIC_ALIGNED_GATE_PREREGISTRATION_E518.md",
                "bootstrap_seed": 998465967,
                "scientific_fit": False,
                "neural_model_fit": False,
                "public_submission": False,
            }
            fd = os.open(attempt_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(marker, stream, indent=2)
                stream.write("\n")
            from metric_aligned_gate import run_registered_e518

            result, gated_oof, global_oof = run_registered_e518(project_root, inputs)

        if not plot_path.is_file():
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            folds = result["folds"]
            x = np.arange(len(folds))
            width = 0.24
            figure, axes = plt.subplots(1, 2, figsize=(13, 5))
            for offset, (key, label) in enumerate(
                (("anchor", "C1 anchor"), ("global", "global ab12"), ("gated", "gated ab12"))
            ):
                axes[0].bar(
                    x + (offset - 1) * width,
                    [fold[key]["wp"] for fold in folds],
                    width,
                    label=label,
                )
            axes[0].set_xticks(x, [f"fold {fold['outer_fold']}" for fold in folds])
            axes[0].set_ylabel("global_wp_clip2_v1")
            axes[0].set_title("Held-out complete-chain endpoints")
            axes[0].legend(fontsize=8)
            axes[1].bar(x - 0.18, [fold["beta_global"] for fold in folds], 0.36, label="global beta")
            axes[1].bar(x + 0.18, [fold["beta_gate"] for fold in folds], 0.36, label="gated beta")
            axes[1].set_xticks(x, [f"fold {fold['outer_fold']}" for fold in folds])
            axes[1].set_ylim(0, 1)
            axes[1].set_ylabel("selected beta")
            axes[1].set_title("Exact-metric calibration")
            axes[1].legend(fontsize=8)
            figure.suptitle("E-518: exposed-pool score-aligned coefficient screen")
            figure.tight_layout()
            figure.savefig(plot_path, dpi=170)
            plt.close(figure)

        result["plot_path"] = str(plot_path.relative_to(project_root))
        result["plot_sha256"] = hashlib.sha256(plot_path.read_bytes()).hexdigest()
        result["notebook_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        fold_table = [
            {
                "fold": fold["outer_fold"],
                "anchor_wp": fold["anchor"]["wp"],
                "global_wp": fold["global"]["wp"],
                "gated_wp": fold["gated"]["wp"],
                "beta_global": fold["beta_global"],
                "beta_gate": fold["beta_gate"],
                "routed_rows": fold["evaluation_gate_rows"],
                "routed_row_fraction": fold["evaluation_gate_row_fraction"],
                "routed_weight_fraction": fold["evaluation_gate_official_weight_fraction"],
            }
            for fold in result["folds"]
        ]
        summary = mo.md(
            f"**E-518 completed on exposed chains.** Exact official WP: anchor "
            f"`{result['anchor']['wp']:.6f}`, gated `{result['gated']['wp']:.6f}`, "
            f"global control `{result['global']['wp']:.6f}`. Registered screen: "
            f"**{'PASS' if result['development_gate_passed'] else 'FAIL'}**. "
            "This is a development screen, not a public score claim. "
            f"Plot: `{result['plot_path']}`."
        )
        e518_output = mo.vstack([summary, mo.ui.table(fold_table)])
    e518_output  # noqa: B018 - Marimo renders the action result.
    return


if __name__ == "__main__":
    app.run()
