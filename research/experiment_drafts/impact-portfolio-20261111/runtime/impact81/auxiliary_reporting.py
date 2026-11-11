"""Explicit auxiliary-loss sufficient sums and receiver telemetry for cards04–06."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from wnn.training.sequence_fit_artifacts import epoch_summary as canonical_epoch_summary

# The canonical recipe and JSON telemetry carry heterogeneous validated values.
WIDTH = 15


def window_totals(raw: Tensor, label_mass: Tensor, variant: str) -> Tensor:
    """Convert globally summed raw terms into one update's correctly normalized metrics."""
    result = torch.zeros(WIDTH, dtype=torch.float64, device=raw.device)
    if variant == "deep_supervision":
        mass = label_mass.double()
        result[1:4] = raw[:3] / mass if mass > 0 else 0
        result[0] = 0.1 * result[1:4].mean()
    elif variant == "pairwise_ranking":
        mass = raw[5:7]
        safe_mass = torch.where(mass > 0, mass, torch.ones_like(mass))
        result[4:6] = torch.where(mass > 0, raw[3:5] / safe_mass, 0)
        result[0] = 0.05 * result[4:6].mean()
        result[6:8] = raw[7:9]
        result[8:10] = raw[9:11]
        result[10:12] = mass
        result[12:14] = raw[11:13]
    result[14] = 1
    return result


def metrics(totals: Tensor) -> dict[str, Any]:
    """Report window means and pooled counts without relabeling ranking as official WP."""
    values = totals.detach().cpu().tolist()
    windows = values[14]
    if windows <= 0:
        raise ValueError("auxiliary summary requires at least one completed update")
    return {
        "weighted_auxiliary_loss": values[0] / windows,
        "deep_wmse_by_depth": [value / windows for value in values[1:4]],
        "pair_loss_by_target": [value / windows for value in values[4:6]],
        "eligible_pair_counts": values[6:8],
        "pair_weight_mass": values[10:12],
        "pair_concordance_by_target": [
            values[8 + target] / values[10 + target] if values[10 + target] > 0 else None
            for target in range(2)
        ],
        "pair_tie_counts": values[12:14],
        "completed_windows": int(windows),
    }


def summarize_epoch(result: Any, config: dict[str, Any]) -> dict[str, Any]:
    """Retain canonical primary metrics and add the actual optimized auxiliary objective."""
    summary = canonical_epoch_summary(result, config)
    auxiliary = metrics(result.auxiliary_totals)
    primary = float(summary["total_loss"])
    return summary | {
        "primary_loss": primary,
        "total_loss": primary + auxiliary["weighted_auxiliary_loss"],
        "impact81_auxiliary": auxiliary,
    }


def profile_fields(results: list[Any], config: dict[str, Any]) -> dict[str, Any]:
    """Aggregate bounded profile updates with the same window normalization as fitting."""
    if not results or sum(item.updates for item in results) == 0:
        return {
            "total_loss": 0.0,
            "primary_loss": 0.0,
            "completed_windows": 0,
            "status": "no_updates",
        }
    from impact81.auxiliary_loop import ExperimentEpochResult

    combined = ExperimentEpochResult(
        sum(item.updates for item in results),
        False,
        sum((item.totals for item in results), torch.zeros_like(results[0].totals)),
        sum(item.elapsed_seconds for item in results),
        sum(
            (item.objective_totals for item in results),
            torch.zeros_like(results[0].objective_totals),
        ),
        sum(
            (item.auxiliary_totals for item in results),
            torch.zeros_like(results[0].auxiliary_totals),
        ),
    )
    summary = summarize_epoch(combined, config)
    return {
        "total_loss": summary["total_loss"],
        "primary_loss": summary["primary_loss"],
        **summary["impact81_auxiliary"],
    }
