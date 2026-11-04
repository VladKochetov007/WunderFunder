from __future__ import annotations

import json
from pathlib import Path

from wnn.multistage_search import atomic_write_json, file_sha256, read_json


def export_stage4_ensemble_plots(result_path: Path) -> dict[str, object]:
    """Plot qualified component WP and learned CPU simplex weights without epoch axes."""
    result = read_json(result_path)
    if result.get("schema") != "wnn.multistage-stage4-ensemble.v1":
        raise ValueError("stage 4 plotting requires the durable ensemble result")
    ensemble = result.get("ensemble")
    if not isinstance(ensemble, dict) or not isinstance(ensemble.get("members"), list):
        raise ValueError("ensemble result has no member table")
    members = ensemble["members"]
    if not members:
        raise ValueError("ensemble result contains no components")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = result_path.parent / "plots"
    output.mkdir(parents=True, exist_ok=True)
    names = [str(item["name"]) for item in members]
    weights = [float(item["weight"]) for item in members]
    figure, axis = plt.subplots(figsize=(max(8, len(names) * 0.55), 4.5))
    axis.bar(names, weights, color="steelblue")
    axis.set(xlabel="Qualified inner-control predictor", ylabel="Simplex weight")
    axis.set_title("Stage 4 CPU convex ensemble weights")
    axis.tick_params(axis="x", rotation=35)
    axis.grid(axis="y", alpha=0.25)
    _save_pair(figure, output, "stage4_component_weights")

    figure, axis = plt.subplots(figsize=(8, 4.5))
    objective = [float(item["objective"]["value"]) for item in members]
    labels = names + ["Learned convex ensemble"]
    values = objective + [float(ensemble["control_wp"]["wp"])]
    colors = ["gray"] * len(names) + ["darkgreen"]
    axis.barh(labels, values, color=colors)
    axis.set(
        xlabel="Inner train proxy WP", title="Stage 4 control comparison (same held-out members)"
    )
    axis.grid(axis="x", alpha=0.25)
    _save_pair(figure, output, "stage4_control_wp")

    exports = [
        {"path": str(path.resolve()), "sha256": file_sha256(path)}
        for path in sorted(output.glob("stage4_*.png")) + sorted(output.glob("stage4_*.svg"))
    ]
    sidecar = {
        "schema": "wnn.multistage-plot-source.v1",
        "stage": 4,
        "plot_type": "ensemble_trials_and_weights",
        "result_path": str(result_path.resolve()),
        "result_sha256": file_sha256(result_path),
        "scope": result.get("scope"),
        "auxiliary_split_sha256": result.get("auxiliary_split_sha256"),
        "primary_split_sha256": result.get("primary_split_sha256"),
        "optimization": result.get("optimization"),
        "components": members,
        "fixed_primary_tl3_reference": result.get("fixed_primary_tl3_reference"),
        "plots": exports,
    }
    atomic_write_json(output / "ensemble_plot_source.json", sidecar)
    return {
        "status": "exported",
        "plot_count": len(exports),
        "source_path": str(output / "ensemble_plot_source.json"),
    }


def export_stage4_trial_plots(run_dir: Path) -> dict[str, object]:
    """Refresh trial-level WP and component-weight graphs from actual Optuna outcomes."""
    event_path = run_dir / "events.jsonl"
    if not event_path.is_file():
        return {"status": "no_trials", "plot_count": 0}
    events = [
        json.loads(line)
        for line in event_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    trials = [
        event
        for event in events
        if isinstance(event, dict)
        and event.get("event") == "stage4_optuna_trial"
        and event.get("status") in {"COMPLETE", "PRUNED", "FAIL"}
    ]
    if not trials:
        return {"status": "no_terminal_trials", "plot_count": 0}
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = run_dir / "stage4" / "plots"
    output.mkdir(parents=True, exist_ok=True)
    trials.sort(key=lambda item: int(item["trial_number"]))
    x = [int(item["trial_number"]) for item in trials]
    values = [
        float(item["objective_value"])
        if isinstance(item.get("objective_value"), (int, float))
        else float("nan")
        for item in trials
    ]
    best: float | None = None
    best_so_far: list[float] = []
    for value in values:
        if value == value:
            best = value if best is None else max(best, value)
        best_so_far.append(float("nan") if best is None else best)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.scatter(x, values, color="steelblue", label="Actual pooled clipped WP trial")
    axis.plot(x, best_so_far, color="black", label="Best so far")
    axis.set(xlabel="Optuna trial number", ylabel="Inner train proxy WP")
    axis.set_title("Stage 4 CPU TPE blend search")
    axis.legend()
    axis.grid(True, alpha=0.25)
    _save_pair(figure, output, "stage4_wp_by_trial")
    completed = [item for item in trials if isinstance(item.get("candidate_weights"), list)]
    exported = 2
    if completed:
        matrix = [item["candidate_weights"] for item in completed]
        figure, axis = plt.subplots(figsize=(9, 4.5))
        bottoms = [0.0] * len(completed)
        for index in range(len(matrix[0])):
            widths = [float(row[index]) for row in matrix]
            axis.bar(
                [int(item["trial_number"]) for item in completed],
                widths,
                bottom=bottoms,
                label=f"candidate_{index}",
            )
            bottoms = [left + width for left, width in zip(bottoms, widths, strict=True)]
        axis.set(xlabel="Optuna trial number", ylabel="Simplex component weight")
        axis.set_title("Stage 4 candidate weights by trial")
        axis.legend(loc="upper right", ncols=2, fontsize=8)
        _save_pair(figure, output, "stage4_weights_by_trial")
        exported += 2
    paths = sorted(output.glob("stage4_*_by_trial.png")) + sorted(
        output.glob("stage4_*_by_trial.svg")
    )
    sidecar = {
        "schema": "wnn.multistage-plot-source.v1",
        "stage": 4,
        "plot_type": "optuna_trials_without_epoch_axis",
        "events_path": str(event_path.resolve()),
        "events_sha256": file_sha256(event_path),
        "trial_count": len(trials),
        "trials": trials,
        "plots": [{"path": str(path.resolve()), "sha256": file_sha256(path)} for path in paths],
    }
    atomic_write_json(output / "trial_plot_source.json", sidecar)
    return {
        "status": "exported",
        "plot_count": exported,
        "source_path": str(output / "trial_plot_source.json"),
    }


def _save_pair(figure: object, output: Path, stem: str) -> None:
    for suffix in ("png", "svg"):
        figure.savefig(
            output / f"{stem}.{suffix}", dpi=160 if suffix == "png" else None, bbox_inches="tight"
        )
    import matplotlib.pyplot as plt

    plt.close(figure)
