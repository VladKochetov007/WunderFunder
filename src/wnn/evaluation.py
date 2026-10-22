"""Paired component bootstrap for official Global WP and legacy diagnostics."""

from __future__ import annotations

import numpy as np

from wnn.gmetric import _centered_moments, _merge_moments, wp_from_sums


def paired_component_moments(
    targets: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build stable weighted Welford moments for every sequence and target.

    The six fields are weight, target mean, prediction mean, target M2,
    prediction M2 and cross moment. They must be merged as moments, never
    summed as raw weighted statistics.
    """
    y = np.asarray(targets, dtype=np.float32)
    a = np.asarray(candidate, dtype=np.float32)
    b = np.asarray(reference, dtype=np.float32)
    m = np.asarray(mask, dtype=bool)
    if y.ndim != 3 or y.shape[-1] != 2 or a.shape != y.shape or b.shape != y.shape:
        raise ValueError("expected aligned (N, T, 2) targets and predictions")
    if m.shape != y.shape[:2]:
        raise ValueError("expected aligned (N, T) selection mask")
    if not m.any():
        raise ValueError("no selected rows")

    out_a = np.zeros((len(y), 2, 6), dtype=np.float64)
    out_b = np.zeros_like(out_a)
    for i in range(len(y)):
        if not m[i].any():
            continue
        raw_y, raw_a, raw_b = y[i, m[i]], a[i, m[i]], b[i, m[i]]
        if not (np.isfinite(raw_y).all() and np.isfinite(raw_a).all() and np.isfinite(raw_b).all()):
            raise ValueError("nonfinite selected target or prediction")
        yi, ai, bi = (np.clip(z, -2, 2).astype(np.float64) for z in (raw_y, raw_a, raw_b))
        for k in range(2):
            out_a[i, k] = _centered_moments(yi[:, k], ai[:, k])
            out_b[i, k] = _centered_moments(yi[:, k], bi[:, k])
    return out_a, out_b


def _score_component_moments(moments: np.ndarray, counts: np.ndarray) -> float:
    """Pool a resample of component moments without subtracting raw squares."""
    weight = counts[:, None] * moments[:, :, 0]
    total = weight.sum(axis=0)
    result = np.zeros(2, dtype=np.float64)
    for k in range(2):
        if total[k] <= 0:
            continue
        active = np.flatnonzero(weight[:, k] > 0)
        anchor_y, anchor_p = moments[active[0], k, 1:3]
        dy = moments[:, k, 1] - anchor_y
        dp = moments[:, k, 2] - anchor_p
        mean_y = np.sum(weight[:, k] * dy) / total[k]
        mean_p = np.sum(weight[:, k] * dp) / total[k]
        variance_y = np.sum(counts * moments[:, k, 3] + weight[:, k] * (dy - mean_y) ** 2)
        variance_p = np.sum(counts * moments[:, k, 4] + weight[:, k] * (dp - mean_p) ** 2)
        covariance = np.sum(
            counts * moments[:, k, 5] + weight[:, k] * (dy - mean_y) * (dp - mean_p)
        )
        if np.sqrt(variance_y / total[k]) > 1e-8 and np.sqrt(variance_p / total[k]) > 1e-8:
            result[k] = np.clip(covariance / np.sqrt(variance_y * variance_p), -1, 1)
    return float(result.mean())


def _guard_raw_precision(sums: np.ndarray) -> None:
    """Refuse raw sums when cancellation can move a Pearson score materially."""
    weight = sums[:, 0]
    safe = np.where(weight > 0, weight, 1.0)
    mean = sums[:, 1:3] / safe[:, None]
    second = sums[:, 3:5] / safe[:, None]
    variance = second - mean * mean
    scale = np.maximum(np.abs(second), mean * mean)
    if np.any((weight[:, None] > 0) & (variance <= 1e-8 * scale) & (scale > 0)):
        raise ValueError("raw moments are precision-ambiguous; use paired_component_moments")


def paired_component_bootstrap(
    candidate: np.ndarray,
    reference: np.ndarray,
    components: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
    centered: bool = False,
    return_draws: bool = False,
) -> dict:
    """Resample complete components and recompute both pooled correlations.

    Args:
        candidate: Per-sequence sufficient statistics shaped (N, 2, 6).
        reference: Aligned baseline sufficient statistics with the same shape.
        components: Component identifiers aligned with the N sequences.
        n_boot: Number of paired component resamples.
        seed: Fixed bootstrap random seed.
        centered: True when inputs are Welford moments from
            ``paired_component_moments``. Raw statistics are refused when
            near-constant variance is ambiguous.
        return_draws: Include paired replicate deltas for independent-fold aggregation.
    """
    a = np.asarray(candidate, dtype=np.float64)
    b = np.asarray(reference, dtype=np.float64)
    groups = np.asarray(components)
    if a.ndim != 3 or a.shape[1:] != (2, 6) or a.shape != b.shape:
        raise ValueError("expected aligned (N, 2, 6) sufficient statistics")
    if groups.shape != (len(a),) or len(a) == 0 or n_boot < 2:
        raise ValueError("need nonempty aligned components and at least two resamples")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("nonfinite sufficient statistics")
    identifiers, inverse = np.unique(groups, return_inverse=True)
    ca = np.zeros((len(identifiers), 2, 6), dtype=np.float64)
    cb = np.zeros_like(ca)
    if centered:
        for i, g in enumerate(inverse):
            for k in range(2):
                ca[g, k] = _merge_moments(ca[g, k], a[i, k])
                cb[g, k] = _merge_moments(cb[g, k], b[i, k])
    else:
        np.add.at(ca, inverse, a)
        np.add.at(cb, inverse, b)
        for sums in (ca.sum(axis=0), cb.sum(axis=0)):
            _guard_raw_precision(sums)
    if centered:
        counts = np.ones(len(ca), dtype=np.int64)
        score_a = _score_component_moments(ca, counts)
        score_b = _score_component_moments(cb, counts)
    else:
        score_a = float(wp_from_sums(ca.sum(axis=0)).mean())
        score_b = float(wp_from_sums(cb.sum(axis=0)).mean())
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        selected = rng.integers(0, len(ca), len(ca))
        if centered:
            counts = np.bincount(selected, minlength=len(ca))
            deltas[i] = _score_component_moments(ca, counts) - _score_component_moments(cb, counts)
        else:
            _guard_raw_precision(ca[selected].sum(axis=0))
            _guard_raw_precision(cb[selected].sum(axis=0))
            deltas[i] = float(
                wp_from_sums(ca[selected].sum(axis=0)).mean()
                - wp_from_sums(cb[selected].sum(axis=0)).mean()
            )
    result = {
        "candidate_wp": score_a,
        "reference_wp": score_b,
        "mean_diff": score_a - score_b,
        "bootstrap_sd": float(deltas.std(ddof=1)),
        "ci95": np.percentile(deltas, [2.5, 97.5]).tolist(),
        "p_le_0": float(np.mean(deltas <= 0)),
        "n_sequences": len(a),
        "n_components": len(ca),
        "n_boot": n_boot,
        "seed": seed,
        "metric_id": "global_wp_clip2_v1",
        "centered_moments": centered,
    }
    if return_draws:
        result["bootstrap_deltas"] = deltas.tolist()
    return result


def paired_bootstrap(a, b, strata=None, n_boot: int = 2000, seed: int = 0) -> dict:
    """Legacy block diagnostic: mean paired difference with percentile 95% CI.

    ``a``/``b``: per-sequence scores aligned by sequence. ``strata``: optional
    labels (e.g. market class); resampling is done within each stratum so the
    class mix stays fixed.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("score arrays must align by sequence")
    ok = np.isfinite(a) & np.isfinite(b)
    d = (a - b)[ok]
    s = np.zeros(len(d), dtype=np.int64) if strata is None else np.asarray(strata)[ok]
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(s == g) for g in np.unique(s)]
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups])
        boots[i] = d[idx].mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "mean_diff": float(d.mean()),
        "ci95": [float(lo), float(hi)],
        "n": int(len(d)),
        "p_le_0": float((boots <= 0).mean()),
    }
