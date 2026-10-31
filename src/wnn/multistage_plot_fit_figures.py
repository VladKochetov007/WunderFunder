from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from wnn.multistage_plot_data import elapsed_seconds


def _plot_pair(
    plt: object,
    evaluations: Sequence[dict[str, object]],
    train_key: str,
    control_key: str,
    ylabel: str,
    control_label: str,
) -> object | None:
    values = _series(evaluations, (train_key, control_key))
    if not any(rows for rows in values.values()):
        return None
    fig, axis = plt.subplots(figsize=(8, 4.5))
    for key, label in ((train_key, "Train-eval diagnostic"), (control_key, control_label)):
        rows = values[key]
        if rows:
            axis.plot([row[0] for row in rows], [row[1] for row in rows], marker="o", label=label)
    axis.set(xlabel="Optimizer epoch", ylabel=ylabel)
    axis.legend()
    axis.grid(True, alpha=0.25)
    return fig


def _plot_wp(
    plt: object,
    evaluations: Sequence[dict[str, object]],
    label: str,
    reference_wp: float | None = None,
) -> object | None:
    keys = ("train_eval/wp", "control/wp", "control/wp/t0", "control/wp/t1")
    values = _series(evaluations, keys)
    if not any(rows for rows in values.values()):
        return None
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for key, name in (
        ("train_eval/wp", "Train-eval pooled WP"),
        ("control/wp", f"{label} pooled WP"),
    ):
        rows = values[key]
        if rows:
            axes[0].plot([row[0] for row in rows], [row[1] for row in rows], marker="o", label=name)
    axes[0].set_ylabel("Pooled WP")
    if reference_wp is not None and math.isfinite(reference_wp):
        axes[0].axhline(
            reference_wp,
            color="black",
            linestyle="--",
            label="Fixed parent endpoint / reused skip control",
        )
        axes[0].legend()
    axes[0].legend()
    for key, name in (("control/wp/t0", "Control t0"), ("control/wp/t1", "Control t1")):
        rows = values[key]
        if rows:
            axes[1].plot([row[0] for row in rows], [row[1] for row in rows], marker="o", label=name)
    axes[1].set(xlabel="Optimizer epoch", ylabel="Per-target WP")
    axes[1].legend()
    for axis in axes:
        axis.grid(True, alpha=0.25)
    return fig


def _plot_online(
    plt: object, events: Sequence[dict[str, object]], first_wall: str | None
) -> object | None:
    keys = ("train/loss", "train/label_weighted_mse", "train/kd_loss", "train/hint_loss")
    rows = _training_series(events, keys, first_wall)
    if not any(points for points in rows.values()):
        return None
    fig, axis = plt.subplots(figsize=(8, 4.5))
    for key, points in rows.items():
        if points:
            axis.plot([point[0] for point in points], [point[1] for point in points], label=key)
    axis.set(
        xlabel="Elapsed seconds",
        ylabel="Online loss",
        title="Online training objective (not validation WMSE)",
    )
    axis.legend()
    axis.grid(True, alpha=0.25)
    return fig


def _plot_optimizer(
    plt: object, events: Sequence[dict[str, object]], first_wall: str | None
) -> object | None:
    rows = _training_series(
        events,
        ("optimizer/gradient_norm_preclip", "optimizer/was_clipped"),
        first_wall,
        starts_with="optimizer/lr/",
    )
    if not any(points for points in rows.values()):
        return None
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    clip_counts: list[int] = []
    for key, points in rows.items():
        if not points:
            continue
        x, y = [point[0] for point in points], [point[1] for point in points]
        if key.startswith("optimizer/lr/"):
            axes[0].plot(x, y, label=key.rsplit("/", 1)[-1])
        elif key == "optimizer/gradient_norm_preclip":
            axes[1].plot(x, y, label="Preclip gradient norm")
        elif key == "optimizer/was_clipped":
            for value in y:
                clip_counts.append(int(value > 0))
            axes[1].plot(x, y, alpha=0.2, label="Clipped update indicator")
    axes[0].set_ylabel("Learning rate")
    axes[0].legend()
    axes[1].set(xlabel="Elapsed seconds", ylabel="Gradient / clip event")
    axes[1].legend()
    axes[1].text(
        0.02,
        0.96,
        f"Clipped sampled updates: {sum(clip_counts)}; nonfinite update counters: source runtime fails on first nonfinite update",
        transform=axes[1].transAxes,
        va="top",
        fontsize=8,
    )
    return fig


def _plot_resources(
    plt: object, events: Sequence[dict[str, object]], first_wall: str | None
) -> object | None:
    keys = (
        "gpu/utilization_pct",
        "gpu/device_memory_used_bytes",
        "cpu/process_utilization_pct",
        "cpu/rss_bytes",
        "throughput/objective_rows_per_second",
    )
    rows = _training_series(events, keys, first_wall)
    if not any(points for points in rows.values()):
        return None
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for key, points in rows.items():
        if points and key in ("gpu/utilization_pct", "cpu/process_utilization_pct"):
            axes[0].plot([p[0] for p in points], [p[1] for p in points], label=key)
        if points and key in ("gpu/device_memory_used_bytes", "cpu/rss_bytes"):
            axes[1].plot(
                [p[0] for p in points], [p[1] / (1024**3) for p in points], label=f"{key} GiB"
            )
        if points and key == "throughput/objective_rows_per_second":
            axes[1].plot([p[0] for p in points], [p[1] for p in points], label="Objective rows/sec")
    axes[0].set_ylabel("Utilization %")
    axes[0].legend()
    axes[1].set(xlabel="Elapsed seconds", ylabel="GiB / rows per second")
    axes[1].legend()
    return fig


def _series(
    events: Sequence[dict[str, object]], keys: Sequence[str]
) -> dict[str, list[tuple[float, float]]]:
    rows: dict[str, list[tuple[float, float]]] = {key: [] for key in keys}
    for event in events:
        epoch, metrics = event.get("epoch"), event.get("metrics")
        if not isinstance(epoch, (int, float)) or not isinstance(metrics, dict):
            continue
        for key in keys:
            value = metrics.get(key)
            if _finite_number(value):
                rows[key].append((float(epoch), float(value)))
    return rows


def _training_series(
    events: Sequence[dict[str, object]],
    keys: Sequence[str],
    first_wall: str | None,
    *,
    starts_with: str | None = None,
) -> dict[str, list[tuple[float, float]]]:
    result: dict[str, list[tuple[float, float]]] = {key: [] for key in keys}
    for event in events:
        step = event.get("step")
        if not isinstance(step, int):
            continue
        x = elapsed_seconds(event, first_wall)
        if x is None:
            x = float(step)
        metrics = event.get("metrics")
        if not isinstance(metrics, dict):
            continue
        names = set(keys) | {key for key in metrics if starts_with and key.startswith(starts_with)}
        for key in names:
            value = metrics.get(key)
            if _finite_number(value):
                result.setdefault(key, []).append((x, float(value)))
    return result


def _control_context(event: Mapping[str, object]) -> dict[str, object]:
    context = event.get("context")
    if not isinstance(context, Mapping):
        return {}
    control = context.get("control")
    return dict(control) if isinstance(control, Mapping) else {}


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )
