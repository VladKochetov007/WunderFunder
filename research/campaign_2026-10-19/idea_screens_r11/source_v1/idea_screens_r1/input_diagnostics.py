"""Input-only diagnostics for the bounded mechanism screens."""

from __future__ import annotations

import numpy as np

from .constants import GROUPS


def _median(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else None


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return float(numerator / denominator)


def _block_summary(block: np.ndarray) -> dict[str, object]:
    """Summarize magnitudes and strict-positive composition eligibility."""
    finite = np.isfinite(block)
    entries = block[finite]
    channel_medians = [
        float(np.median(np.abs(block[finite[:, j], j])))
        for j in range(block.shape[1])
        if finite[:, j].any()
    ]
    complete = finite.all(axis=1)
    ranges = np.ptp(block[complete], axis=1) if complete.any() else np.array([])
    pairs = finite[1:] & finite[:-1]
    steps = np.abs(block[1:] - block[:-1])[pairs]
    positive = complete & (block > 0).all(axis=1)
    low = min(channel_medians) if channel_medians else None
    high = max(channel_medians) if channel_medians else None
    return {
        "finite_entries": int(entries.size),
        "finite_entry_fraction": float(finite.mean()),
        "strictly_positive_complete_rows": int(positive.sum()),
        "strictly_positive_complete_row_fraction": float(positive.mean()),
        "median_abs_value": _median(np.abs(entries)),
        "median_abs_step": _median(steps),
        "median_complete_row_range": _median(ranges),
        "channel_median_abs_min": low,
        "channel_median_abs_max": high,
        "channel_magnitude_ratio_max_to_min": _ratio(high, low),
    }


def _net_variation(block: np.ndarray) -> dict[str, object]:
    """Describe endpoint displacement relative to observed total variation."""
    net, variation, fractions = [], [], []
    for column in range(block.shape[1]):
        series = block[:, column]
        values = series[np.isfinite(series)]
        if not values.size:
            continue
        pairs = np.isfinite(series[1:]) & np.isfinite(series[:-1])
        total = float(np.abs(np.diff(series)[pairs]).sum(dtype=np.float64))
        displacement = float(abs(values[-1] - values[0]))
        net.append(displacement)
        variation.append(total)
        if total > 0:
            fractions.append(displacement / total)
    return {
        "channels_with_finite_values": len(net),
        "channels_with_positive_variation": len(fractions),
        "median_abs_endpoint_displacement": _median(np.asarray(net)),
        "median_total_variation": _median(np.asarray(variation)),
        "median_net_to_variation": _median(np.asarray(fractions)),
    }


def _window_summary(block: np.ndarray, start: int, stop: int) -> dict[str, object]:
    """Summarize one fixed row window without using labels or score masks."""
    view = block[start:stop]
    if not len(view):
        return {"rows": 0, "median_abs_value": None, "median_abs_step": None}
    finite = view[np.isfinite(view)]
    pairs = np.isfinite(view[1:]) & np.isfinite(view[:-1])
    steps = np.abs(view[1:] - view[:-1])[pairs]
    return {
        "rows": len(view),
        "median_abs_value": _median(np.abs(finite)),
        "median_abs_step": _median(steps),
    }


def _innovation_summary(block: np.ndarray) -> dict[str, object]:
    """Report input-difference sizes, not recurrent hidden-state behavior."""
    pairs = np.isfinite(block[1:]).all(axis=1) & np.isfinite(block[:-1]).all(axis=1)
    delta = np.linalg.norm(block[1:][pairs] - block[:-1][pairs], axis=1)
    current = np.linalg.norm(block[1:][pairs], axis=1)
    relative = delta / np.maximum(current, np.finfo(np.float64).tiny)
    return {
        "finite_adjacent_pairs": int(pairs.sum()),
        "delta_l2_median": _median(delta),
        "delta_l2_p95": float(np.quantile(delta, 0.95)) if delta.size else None,
        "delta_to_current_norm_median": _median(relative),
        "interpretation": "input innovations only; recurrent hidden state was not observed",
    }


def input_summary(x: np.ndarray) -> dict[str, object]:
    """Compute fit/eval-safe feature, volume, path, warmup, and innovation summaries."""
    blocks = {name: x[:, columns] for name, columns in GROUPS.items()}
    summaries = {name: _block_summary(block) for name, block in blocks.items()}
    warm_stop = min(99, len(x))
    early_stop = min(198, len(x))
    late_start = max(warm_stop, len(x) - 99)
    warmup = {}
    for name, block in blocks.items():
        warm = _window_summary(block, 0, warm_stop)
        early = _window_summary(block, warm_stop, early_stop)
        late = _window_summary(block, late_start, len(x))
        warmup[name] = {
            "warmup": warm,
            "early_after_warmup": early,
            "late": late,
            "late_to_warmup_abs_level_ratio": _ratio(
                late["median_abs_value"], warm["median_abs_value"]
            ),
            "late_to_warmup_step_ratio": _ratio(late["median_abs_step"], warm["median_abs_step"]),
        }
    return {
        "rows": len(x),
        "fully_finite_feature_rows": int(np.isfinite(x).all(axis=1).sum()),
        "fully_finite_feature_row_fraction": float(np.isfinite(x).all(axis=1).mean()),
        "volume_blocks": {name: summaries[name] for name in ("i0_v", "i1_v")},
        "group_scale_descriptives": summaries,
        "trade_to_quote_magnitude_ratios": {
            "i0_dp_to_i0_p_level": _ratio(
                summaries["i0_dp"]["median_abs_value"], summaries["i0_p"]["median_abs_value"]
            ),
            "i0_dp_to_i0_p_step": _ratio(
                summaries["i0_dp"]["median_abs_step"], summaries["i0_p"]["median_abs_step"]
            ),
            "status": "descriptive only; shared physical units are not established",
        },
        "net_vs_variation": {name: _net_variation(blocks[name]) for name in ("i0_p", "i0_dp")},
        "warmup_vs_later": warmup,
        "innovation_descriptives": _innovation_summary(blocks["i0_p"]),
    }
