from __future__ import annotations

from collections.abc import Iterable

COMPONENTS = ("z", "reference", "mixture")
COHORTS = ("train_eval", "valid", "control")
TARGETS = ("t0", "t1")


def component_distribution_figures(
    plt: object, events: Iterable[dict[str, object]]
) -> list[tuple[str, object]]:
    """Plot same-forward component moments by cohort and target when recorded."""
    rows = list(events)
    figures: list[tuple[str, object]] = []
    for component in COMPONENTS:
        for family, statistics in (
            ("means", ("raw_mean", "weighted_raw_mean")),
            ("spread", ("raw_std", "weighted_raw_std")),
        ):
            figure, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
            plotted = False
            for axis, target in zip(axes, TARGETS, strict=True):
                for cohort in COHORTS:
                    for statistic in statistics:
                        key = f"{cohort}/{component}/{statistic}/{target}"
                        points = [
                            (event.get("epoch"), event.get("metrics", {}).get(key))
                            for event in rows
                        ]
                        points = [
                            (epoch, value)
                            for epoch, value in points
                            if isinstance(epoch, (int, float)) and isinstance(value, (int, float))
                        ]
                        if not points:
                            continue
                        axis.plot(
                            [point[0] for point in points],
                            [point[1] for point in points],
                            label=f"{cohort} {statistic.replace('_', ' ')}",
                        )
                        plotted = True
                axis.set_title(target)
                axis.set_xlabel("Evaluation epoch")
                axis.grid(alpha=0.25)
            if not plotted:
                plt.close(figure)
                continue
            axes[0].set_ylabel("Prediction moment")
            axes[-1].legend(fontsize=7)
            figure.suptitle(f"{component}: per-target {family} (same forward)")
            figures.append((f"prediction_{component}_{family}_by_target", figure))
    return figures
