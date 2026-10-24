"""Paired component bootstrap using the runtime's canonical implementation."""

from __future__ import annotations

from typing import Any

from campaign_io import require


def _paired(
    candidate: dict[str, Any],
    reference: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Compute the canonical paired component bootstrap for aligned fixed scores."""
    import numpy as np

    from wnn.evaluation import paired_component_bootstrap

    left_score, right_score = candidate["score"], reference["score"]
    require(
        left_score["split_sha256"] == right_score["split_sha256"],
        "paired endpoints use different splits",
    )
    require(
        left_score["score"]["selected_rows"] == right_score["score"]["selected_rows"],
        "paired endpoints score different row counts",
    )
    left_moments, right_moments = (
        left_score["component_centered_moments"],
        right_score["component_centered_moments"],
    )
    components = sorted(left_moments, key=int)
    require(components == sorted(right_moments, key=int), "paired endpoint component sets differ")
    candidate_values = np.asarray([left_moments[group] for group in components], dtype=np.float64)
    reference_values = np.asarray([right_moments[group] for group in components], dtype=np.float64)
    require(
        candidate_values.shape == reference_values.shape == (len(components), 2, 6),
        "paired component moments have invalid shape",
    )
    require(
        np.allclose(
            candidate_values[:, :, [0, 1, 3]],
            reference_values[:, :, [0, 1, 3]],
            rtol=1e-10,
            atol=1e-8,
        ),
        "paired endpoints use different labels or weights",
    )
    interval = paired_component_bootstrap(
        candidate_values,
        reference_values,
        np.asarray(components),
        n_boot=2000,
        seed=seed,
        centered=True,
    )
    delta = float(left_score["score"]["wp"] - right_score["score"]["wp"])
    require(
        abs(interval["mean_diff"] - delta) < 2e-7, "centered moments and official grades disagree"
    )
    return {
        "candidate_score_sha256": candidate["evidence"]["score_sha256"],
        "reference_score_sha256": reference["evidence"]["score_sha256"],
        "candidate_wp": float(left_score["score"]["wp"]),
        "reference_wp": float(right_score["score"]["wp"]),
        "candidate_minus_reference": delta,
        "component_bootstrap": interval,
    }
