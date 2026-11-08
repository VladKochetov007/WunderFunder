from __future__ import annotations

import argparse
import logging
from pathlib import Path

from scripts.sequence_monitor_plot_distributions import component_distribution_figures
from wnn.multistage_plot_data import read_evaluation_events, read_training_events
from wnn.multistage_plot_fit_figures import (
    _plot_online,
    _plot_optimizer,
    _plot_pair,
    _training_series,
)
from wnn.multistage_search import atomic_write_json, file_sha256

LOGGER = logging.getLogger(__name__)


def export_plots(run_dir: Path, plots_dir: Path) -> dict[str, object]:
    """Render unchanged collected native journals locally without loading any model.

    Preserve every epoch evaluation; reuse the declared native telemetry strides.
    The source receipt binds actual bytes and masks, leaving missing metrics absent.
    """
    evaluation_path = run_dir / "monitor_evaluations.jsonl"
    events_path = run_dir / "monitor_events.jsonl"
    evaluations, evaluation_policy = read_evaluation_events(evaluation_path)
    training, training_policy = read_training_events(events_path)
    if not evaluations:
        return {"status": "awaiting_first_evaluation", "plot_count": 0}
    observer_path = run_dir / "monitor_observer.jsonl"
    observer_policy: dict[str, object] = {}
    if observer_path.is_file():
        observer, observer_policy = read_training_events(observer_path)
        training.extend(observer)
    rank_path = run_dir / "monitor_rank0.jsonl"
    rank_zero, rank_policy = read_training_events(rank_path) if rank_path.is_file() else ([], {})
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir.mkdir(parents=True, exist_ok=True)
    roles = [
        role
        for role in ("valid", "control")
        if any(f"{role}/wp" in event.get("metrics", {}) for event in evaluations)
    ]
    if len(roles) != 1:
        raise ValueError("plot export requires exactly one explicitly recorded evaluation role")
    role = roles[0]
    latest_context = evaluations[-1].get("context", {})
    train_context = latest_context.get("train_eval", {}) if isinstance(latest_context, dict) else {}
    role_context = latest_context.get(role, {}) if isinstance(latest_context, dict) else {}
    train_label = _cohort_label(
        "Train-eval in-sample subset", train_context, include_selection=True
    )
    role_label = _cohort_label(
        "Official valid" if role == "valid" else "Fitting-only inner control",
        role_context,
    )
    first_wall = min(str(event["wall_time"]) for event in training + evaluations)
    figures = [
        (
            "train_eval_vs_evaluation_wmse",
            _plot_pair(
                plt,
                evaluations,
                "train_eval/loss",
                f"{role}/loss",
                "Weighted MSE",
                role_label,
                train_label=train_label,
            ),
        ),
        (
            "train_eval_vs_evaluation_wp",
            _plot_pair(
                plt,
                evaluations,
                "train_eval/wp",
                f"{role}/wp",
                "Pooled WP",
                role_label,
                train_label=train_label,
            ),
        ),
        (
            "target_t0_wp",
            _plot_pair(
                plt,
                evaluations,
                "train_eval/wp/t0",
                f"{role}/wp/t0",
                "Target t0 WP",
                role_label,
                train_label=train_label,
            ),
        ),
        (
            "target_t1_wp",
            _plot_pair(
                plt,
                evaluations,
                "train_eval/wp/t1",
                f"{role}/wp/t1",
                "Target t1 WP",
                role_label,
                train_label=train_label,
            ),
        ),
        ("online_objectives", _plot_online(plt, training, first_wall)),
        ("optimizer_telemetry", _plot_optimizer(plt, training, first_wall)),
    ]
    figures.extend(component_distribution_figures(plt, evaluations))
    if rank_zero:
        # Native B300 rank rows lack update timestamps. Use their actual update
        # clock rather than interpreting sampled source-file mtime as update time.
        for name, plotter in (
            ("native_rank0_objectives_by_update", _plot_online),
            ("native_rank0_optimizer_by_update", _plot_optimizer),
        ):
            figure = plotter(plt, rank_zero, None)
            if figure is not None:
                figure.axes[-1].set_xlabel("Optimizer updates")
                figures.append((name, figure))
    keys = {key for event in training for key in event.get("metrics", {})}
    for name, selected in (
        (
            "gpu_utilization",
            sorted(
                key for key in keys if key.startswith("gpu/") and key.endswith("utilization_pct")
            ),
        ),
        (
            "gpu_memory",
            sorted(key for key in keys if key.startswith("gpu/") and "memory_used_bytes" in key),
        ),
        (
            "progress_ages",
            sorted(
                key for key in keys if key.startswith("observer/") and key.endswith("_age_seconds")
            ),
        ),
        (
            "sampled_parameter_norms",
            sorted(
                key
                for key in keys
                if key.startswith("optimizer/sampled_") and key.endswith("_norm")
            ),
        ),
        ("throughput", sorted(key for key in keys if key.startswith("throughput/"))),
    ):
        rows = _training_series(training, selected, first_wall)
        if not any(rows.values()):
            continue
        figure, axis = plt.subplots(figsize=(9, 4.5))
        for key, points in rows.items():
            if points:
                axis.plot([p[0] for p in points], [p[1] for p in points], label=key)
        axis.set(xlabel="Elapsed seconds", ylabel=name.replace("_", " "))
        axis.legend(fontsize=7)
        axis.grid(alpha=0.25)
        figures.append((name, figure))
    exports: dict[str, str] = {}
    for name, figure in figures:
        if figure is None:
            continue
        figure.suptitle(f"{run_dir.name}: {name.replace('_', ' ')}", fontsize=11)
        figure.text(
            0.01,
            0.01,
            _comparison_note(role),
            fontsize=7,
        )
        figure.tight_layout(rect=(0, 0.07, 1, 0.95))
        for suffix in ("png", "svg"):
            path = plots_dir / f"{name}.{suffix}"
            figure.savefig(path, dpi=160 if suffix == "png" else None)
            exports[path.name] = file_sha256(path)
        plt.close(figure)
    source_files = {
        name: file_sha256(run_dir / name)
        for name in (
            "run_metadata.json",
            "monitor_events.jsonl",
            "monitor_evaluations.jsonl",
            "monitor_observer.jsonl",
            "monitor_rank0.jsonl",
            "rank0.jsonl",
            "epoch_index.json",
            "terminal_status.json",
        )
        if (run_dir / name).is_file()
    }
    receipt = {
        "schema": "wnn.sequence-monitor-plots.v1",
        "status": "exported",
        "run_directory": str(run_dir.resolve()),
        "source_sha256": source_files,
        "evaluation_role": role,
        "evaluation_context": evaluations[-1].get("context", {}),
        "train_eval_comparability": train_context.get("comparison_to_evaluation", {}),
        "training_sampling_policy": training_policy,
        "evaluation_sampling_policy": evaluation_policy,
        "observer_sampling_policy": observer_policy,
        "rank_zero_sampling_policy": rank_policy,
        "plots_sha256": exports,
        "plot_count": len(exports),
        "model_evaluations_triggered": 0,
    }
    atomic_write_json(plots_dir / "plot_source.json", receipt)
    return receipt


def _cohort_label(base: str, context: object, *, include_selection: bool = False) -> str:
    """Show recorded cohort size and mask without inferring missing metadata."""
    if not isinstance(context, dict):
        return base
    details: list[str] = []
    count = context.get("member_count")
    if isinstance(count, int):
        details.append(f"n={count}")
    selection = context.get("selection")
    if include_selection and isinstance(selection, str):
        details.append(selection)
    mask = context.get("row_mask")
    if isinstance(mask, str):
        details.append(mask)
    return f"{base} ({', '.join(details)})" if details else base


def _comparison_note(role: str) -> str:
    """State the interpretation limit of each reported cohort pair."""
    if role == "valid":
        return (
            "Train-eval is an in-sample fitting subset; its difference from full official valid\n"
            "is not a calibrated generalization gap. Valid is development-exposed."
        )
    return (
        "Train-eval and inner control are fitting diagnostics;\n"
        "neither difference is independent validation."
    )


def main() -> None:
    """Export local PNG/SVG artifacts from a completed immutable collection snapshot."""
    parser = argparse.ArgumentParser(
        description="Plot native sequence journals without model evaluation"
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--plots-dir", required=True, type=Path)
    args = parser.parse_args()
    result = export_plots(args.run_dir, args.plots_dir)
    LOGGER.info("Plot export %s: %s files", result["status"], result["plot_count"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
