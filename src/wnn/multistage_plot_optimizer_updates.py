from __future__ import annotations

from collections.abc import Sequence

from wnn.multistage_plot_fit_figures import _training_series

_FAMILIES = (
    (
        "relative",
        ("delta_weight_over_pre_weight_norm",),
        "Actual delta-weight / pre-weight norm (non-floored samples)",
    ),
    ("delta_rms", ("delta_weight_rms",), "Actual parameter-update RMS"),
    ("weight_rms", ("pre_weight_rms",), "Pre-update parameter RMS"),
    ("applied_lr", ("applied_lr",), "Learning rate actually applied before scheduler"),
    ("gradients", ("preclip_gradient_norm", "postclip_gradient_norm"), "Group gradient norm"),
    ("state", ("momentum_norm", "exp_avg_norm"), "Optimizer-state norm"),
    ("denominator_flags", ("zero_weight_denominator_count",), "Groups using relative-norm floor"),
)


def actual_optimizer_update_figures(
    plt: object,
    events: Sequence[dict[str, object]],
    first_wall: str | None,
    *,
    source: str,
) -> list[tuple[str, object]]:
    """Plot actual producer-sampled changes with separate units and honest time axes."""
    all_keys = {key for event in events for key in event.get("metrics", {})}
    figures: list[tuple[str, object]] = []
    for name, suffixes, ylabel in _FAMILIES:
        selected = sorted(
            key
            for key in all_keys
            if key.startswith("optimizer_update/group/") and key.rsplit("/", 1)[-1] in suffixes
        )
        series = _training_series(events, selected, first_wall)
        if not any(series.values()):
            continue
        figure, axis = plt.subplots(figsize=(9, 4.5))
        for key, points in series.items():
            if points:
                axis.plot(
                    [point[0] for point in points],
                    [point[1] for point in points],
                    marker=".",
                    label=key.removeprefix("optimizer_update/group/"),
                )
        axis.set(
            xlabel="Optimizer updates" if first_wall is None else "Elapsed seconds",
            ylabel=ylabel,
        )
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7, ncol=2)
        figures.append((f"actual_optimizer_{source}_{name}", figure))
    return figures
