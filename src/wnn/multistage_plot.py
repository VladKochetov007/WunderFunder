from __future__ import annotations

from pathlib import Path

from wnn.multistage_plot_data import (
    metric,
    read_evaluation_events,
    read_training_events,
)
from wnn.multistage_plot_fit_figures import (
    _control_context,
    _plot_online,
    _plot_optimizer,
    _plot_pair,
    _plot_resources,
    _plot_wp,
)
from wnn.multistage_search import atomic_write_json, file_sha256, read_json


def export_stage1_trial_plots(
    trial_request_path: Path, fit_dir: Path, *, force: bool = False
) -> dict[str, object]:
    """Export actual stage1 loss, WP, objective, optimizer, and resource PNG/SVG graphs."""
    return export_stage_trial_plots(trial_request_path, fit_dir, 1, force=force)


def export_stage_trial_plots(
    trial_request_path: Path,
    fit_dir: Path,
    stage: int,
    *,
    force: bool = False,
    plots_dir: Path | None = None,
) -> dict[str, object]:
    """Export actual native training and control plots for one fitted stage trial."""
    if stage not in {1, 2, 3}:
        raise ValueError("fitted trial plots are supported for stages 1 through 3")
    evaluations_path, events_path = (
        fit_dir / "monitor_evaluations.jsonl",
        fit_dir / "monitor_events.jsonl",
    )
    if not evaluations_path.is_file() or not events_path.is_file():
        raise FileNotFoundError("native monitoring journals are not ready for plot export")
    request = read_json(trial_request_path)
    evaluations, eval_policy = read_evaluation_events(evaluations_path)
    training, train_policy = read_training_events(events_path)
    if not evaluations:
        return {"status": "awaiting_first_evaluation", "plot_count": 0}
    output = (plots_dir or fit_dir / "plots").resolve()
    output.mkdir(parents=True, exist_ok=True)
    marker = output / "last_exported_evaluation.json"
    latest_signature = {
        "event_sha256": file_sha256(evaluations_path),
        "event_count": len(evaluations),
        "trial_request_sha256": file_sha256(trial_request_path),
    }
    if not force and marker.exists() and read_json(marker) == latest_signature:
        return {"status": "unchanged", "plot_count": 5}
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wall_times = [
        str(event["wall_time"]) for event in evaluations if isinstance(event.get("wall_time"), str)
    ]
    first_wall = wall_times[0] if wall_times else None
    best_wp = max(
        (metric(event, "control/wp") for event in evaluations),
        default=None,
    )
    context = _control_context(evaluations[-1])
    control_label = "Inner train-only control (selection proxy)"
    reference_wp = _parent_reference_wp(request)
    figure_specs = [
        (
            "train_eval_vs_control_loss",
            _plot_pair,
            (evaluations, "train_eval/loss", "control/loss", "Weighted MSE", control_label),
        ),
        (
            "train_eval_vs_control_wp",
            _plot_wp,
            (evaluations, control_label, reference_wp),
        ),
        (
            "online_training_objective",
            _plot_online,
            (training, first_wall),
        ),
        (
            "optimizer_telemetry",
            _plot_optimizer,
            (training, first_wall),
        ),
        (
            "resource_telemetry",
            _plot_resources,
            (training, first_wall),
        ),
    ]
    exported: list[dict[str, str]] = []
    for name, plotter, args in figure_specs:
        figure = plotter(plt, *args)
        if figure is None:
            continue
        figure.suptitle(
            f"Stage {stage} trial {request.get('trial_number')} — {name.replace('_', ' ')}",
            fontsize=12,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.95))
        for suffix in ("png", "svg"):
            path = output / f"{name}.{suffix}"
            figure.savefig(path, dpi=160 if suffix == "png" else None, bbox_inches="tight")
            exported.append({"path": str(path), "sha256": file_sha256(path)})
        plt.close(figure)
    source = {
        "schema": "wnn.multistage-plot-source.v1",
        "stage": stage,
        "trial_number": request.get("trial_number"),
        "scope": "inner_train_proxy",
        "control": context,
        "best_observed_control_wp": best_wp,
        "raw_monitor_events": str(events_path),
        "raw_monitor_events_sha256": train_policy["raw_events_sha256"],
        "raw_evaluations": str(evaluations_path),
        "raw_evaluations_sha256": eval_policy["raw_evaluations_sha256"],
        "sampling_policy": train_policy,
        "evaluation_sampling_policy": eval_policy,
        "plots": exported,
        "checkpoint_and_config_identity": {
            "config_sha256": request.get("config_sha256"),
            "auxiliary_split_sha256": request.get("auxiliary_split_sha256"),
            "profile_identity_sha256": request.get("profile_identity_sha256"),
            "initializer_sha256": request.get("initializer_sha256"),
            "teacher_cache_manifest_sha256": request.get("teacher_cache_manifest_sha256"),
            "teacher_pool_id": request.get("teacher_pool_id"),
            "pair_id": request.get("pair_id"),
            "pair_role": request.get("pair_role"),
            "reused_skip_control": request.get("reused_skip_control"),
        },
        "stage_boundary": _stage_boundary(stage, request),
    }
    atomic_write_json(output / "plot_source.json", source)
    atomic_write_json(marker, latest_signature)
    return {
        "status": "exported",
        "plot_count": len(exported),
        "source_path": str(output / "plot_source.json"),
    }


def _parent_reference_wp(request: dict[str, object]) -> float | None:
    """Return an explicitly recorded fixed parent WP for the stage2 reference line."""
    reused = request.get("reused_skip_control")
    objective = reused.get("objective") if isinstance(reused, dict) else None
    nested = objective.get("objective") if isinstance(objective, dict) else None
    value = nested.get("value") if isinstance(nested, dict) else None
    return float(value) if isinstance(value, (int, float)) else None


def _stage_boundary(stage: int, request: dict[str, object]) -> dict[str, object]:
    """Describe optimizer resets or changed objectives without joining stage curves."""
    if stage == 2:
        return {
            "kind": "fresh_optimizer_lr_reset",
            "at_optimizer_epoch": 0,
            "initializer_sha256": request.get("initializer_sha256"),
        }
    if stage == 3:
        params = request.get("params")
        return {
            "kind": "new_distillation_objective",
            "at_optimizer_epoch": 0,
            "kd_weight": params.get("kd_weight") if isinstance(params, dict) else None,
        }
    return {"kind": "fresh_base_fit", "at_optimizer_epoch": 0}
