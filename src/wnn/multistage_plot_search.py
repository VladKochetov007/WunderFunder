from __future__ import annotations

import json
import math
from pathlib import Path

from wnn.multistage_plot_utils import save_pair


def export_stage1_search_plots(run_dir: Path) -> dict[str, object]:
    """Plot stage1 Optuna outcomes and measured worker costs."""
    return export_stage_search_plots(run_dir, 1)


def export_stage_search_plots(run_dir: Path, stage: int) -> dict[str, object]:
    """Plot one fitted stage's actual Optuna objective and measured trial costs."""
    if stage not in {1, 2, 3}:
        raise ValueError("Optuna fit-stage plots support stages 1 through 3")
    event_path = run_dir / "events.jsonl"
    if not event_path.exists():
        return {"status": "no_trials", "plot_count": 0}
    events = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines()]
    outcomes = [
        event
        for event in events
        if isinstance(event, dict)
        and event.get("event") in ("trial_finished", "trial_failed")
        and event.get("stage") == stage
    ]
    if not outcomes:
        return {"status": "no_terminal_trials", "plot_count": 0}
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = run_dir / f"stage{stage}" / "plots"
    destination.mkdir(parents=True, exist_ok=True)
    ordered = sorted(outcomes, key=lambda item: int(item.get("trial_number", -1)))
    x = [int(item["trial_number"]) for item in ordered]
    values = [
        float(item["objective_value"])
        if isinstance(item.get("objective_value"), (int, float))
        and math.isfinite(float(item["objective_value"]))
        else math.nan
        for item in ordered
    ]
    best_so_far: list[float] = []
    best: float | None = None
    for value in values:
        if math.isfinite(value):
            best = value if best is None else max(best, value)
        best_so_far.append(math.nan if best is None else best)
    figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.scatter(x, values, c=[_status_color(str(item.get("status"))) for item in ordered])
    axis.plot(x, best_so_far, color="black", linewidth=1.4, label="Best observed so far")
    axis.set(
        xlabel="Optuna trial number",
        ylabel="Inner train proxy WP",
        title=f"Stage {stage} TPE search",
    )
    axis.legend()
    save_pair(figure, destination, f"stage{stage}_objective_by_trial")
    statuses = [str(item.get("status", "unknown")) for item in ordered]
    costs = [float(item.get("duration_seconds", 0.0) or 0.0) for item in ordered]
    cumulative: list[float] = []
    total = 0.0
    for cost in costs:
        total += max(0.0, cost)
        cumulative.append(total)
    figure, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].bar(x, costs, color=[_status_color(status) for status in statuses])
    axes[0].set_ylabel("Worker seconds")
    axes[0].set_title("Measured per-trial wall cost")
    axes[1].plot(x, cumulative, marker=".")
    axes[1].set(xlabel="Optuna trial number", ylabel="Cumulative worker seconds")
    save_pair(figure, destination, f"stage{stage}_cost_by_trial")
    return {"status": "exported", "plot_count": 4, "path": str(destination)}


def _status_color(status: str) -> str:
    return {"COMPLETE": "tab:green", "PRUNED": "tab:orange", "FAIL": "tab:red"}.get(
        status.upper(), "tab:gray"
    )
