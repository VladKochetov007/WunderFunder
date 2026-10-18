"""Fit-component-only fixed-horizon target-operator screen."""

from __future__ import annotations

import numpy as np

from .constants import GROUPS, HORIZONS


def _row_mean(block: np.ndarray) -> np.ndarray:
    result = np.full(len(block), np.nan, dtype=np.float64)
    finite = np.isfinite(block).all(axis=1)
    result[finite] = block[finite].mean(axis=1, dtype=np.float64)
    return result


def _moments(x: np.ndarray, y: np.ndarray, weight: np.ndarray) -> list[float]:
    x64, y64, w64 = (item.astype(np.float64, copy=False) for item in (x, y, weight))
    return [
        float(v)
        for v in (
            w64.sum(),
            np.dot(w64, x64),
            np.dot(w64, y64),
            np.dot(w64, x64 * x64),
            np.dot(w64, y64 * y64),
            np.dot(w64, x64 * y64),
        )
    ]


def _add(stats: dict[str, object], x: np.ndarray, y: np.ndarray) -> None:
    stats["n"] = int(stats.get("n", 0)) + len(y)
    for key, sums in (
        ("plain", _moments(x, y, np.ones(len(y)))),
        ("magnitude", _moments(x, y, np.abs(y))),
    ):
        previous = stats.get(key, [0.0] * 6)
        stats[key] = [float(a + b) for a, b in zip(previous, sums, strict=True)]


def _correlation(sums: list[float]) -> float | None:
    sw, sx, sy, sxx, syy, sxy = sums
    if sw <= 0:
        return None
    covariance = sxy - sx * sy / sw
    variance_x, variance_y = sxx - sx * sx / sw, syy - sy * sy / sw
    if variance_x * variance_y <= 0:
        return None
    return float(np.clip(covariance / np.sqrt(variance_x * variance_y), -1.0, 1.0))


def _record(stats: dict[str, object] | None) -> dict[str, object]:
    stats = stats or {"n": 0, "plain": [0.0] * 6, "magnitude": [0.0] * 6}
    return {
        "n_rows": int(stats["n"]),
        "pearson_clipped_target": _correlation(stats["plain"]),
        "magnitude_weighted_pearson": _correlation(stats["magnitude"]),
    }


def _merge(
    by_component: dict[str, dict[str, dict[str, dict[str, object]]]],
    components: list[str],
    candidate: str,
    target: int,
) -> dict[str, object]:
    total: dict[str, object] = {"n": 0, "plain": [0.0] * 6, "magnitude": [0.0] * 6}
    for component in components:
        source = by_component.get(component, {}).get(candidate, {}).get(str(target))
        if source is None:
            continue
        total["n"] = int(total["n"]) + int(source["n"])
        for key in ("plain", "magnitude"):
            total[key] = [a + b for a, b in zip(total[key], source[key], strict=True)]
    return total


def target_screens(fit_sequences: list[dict[str, object]]) -> dict[str, object]:
    """Compare two fixed row-mean deltas across prespecified horizons.

    Each held-out fitting component selects a candidate using only the other
    fitting components. Evaluation-role arrays are never passed to this function.
    """
    candidates = {
        f"{family}_mean_delta_h{horizon}": (family, horizon)
        for family in ("i0_p", "i0_dp")
        for horizon in HORIZONS
    }
    by_component: dict[str, dict[str, dict[str, dict[str, object]]]] = {}
    for sequence in fit_sequences:
        x, y, scored = (
            sequence["x"],
            np.clip(sequence["y"].astype(np.float64), -2, 2),
            sequence["scored"],
        )
        component = str(sequence["component"])
        by_component.setdefault(component, {})
        levels = {name: _row_mean(x[:, GROUPS[name]]) for name in ("i0_p", "i0_dp")}
        for candidate, (family, horizon) in candidates.items():
            if len(x) <= horizon:
                continue
            prediction = levels[family][horizon:] - levels[family][:-horizon]
            for target in range(2):
                labels = y[:-horizon, target]
                valid = scored[:-horizon] & np.isfinite(labels) & np.isfinite(prediction)
                if valid.any():
                    stats = (
                        by_component[component]
                        .setdefault(candidate, {})
                        .setdefault(str(target), {})
                    )
                    _add(stats, prediction[valid], labels[valid])

    components = sorted(by_component)
    pooled = {
        candidate: {
            f"target_{target}": _record(_merge(by_component, components, candidate, target))
            for target in range(2)
        }
        for candidate in candidates
    }
    crossfit = []
    for heldout in components if len(components) >= 2 else []:
        training = [component for component in components if component != heldout]
        for target in range(2):
            options = [
                (candidate, _record(_merge(by_component, training, candidate, target)))
                for candidate in candidates
            ]
            eligible = [
                item for item in options if item[1]["magnitude_weighted_pearson"] is not None
            ]
            selected = (
                max(eligible, key=lambda item: abs(item[1]["magnitude_weighted_pearson"]))
                if eligible
                else None
            )
            held_stats = (
                by_component.get(heldout, {}).get(selected[0], {}).get(str(target))
                if selected
                else None
            )
            crossfit.append(
                {
                    "heldout_component": heldout,
                    "target": target,
                    "training_components": len(training),
                    "selected_candidate": selected[0] if selected else None,
                    "training_fit": selected[1] if selected else None,
                    "heldout_fit_component": _record(held_stats),
                }
            )
    return {
        "fit_only": True,
        "target_clipping": [-2.0, 2.0],
        "selection_weight": "absolute clipped target; reported beside unweighted Pearson",
        "candidate_count": len(candidates),
        "candidate_definition": "mean_j(x[t+h,j]-x[t,j]) for i0_p j=0..21 or i0_dp j=0..3",
        "horizons_rows": list(HORIZONS),
        "components": len(components),
        "pooled_fit_components": pooled,
        "leave_one_component_out": crossfit,
        "split_status": "ok" if len(components) >= 2 else "insufficient_fit_components",
        "limits": "fixed retrospective label-operator screen; future x is diagnostic only, never an inference feature",
    }
