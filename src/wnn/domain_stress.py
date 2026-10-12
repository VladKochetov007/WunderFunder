from __future__ import annotations

import numpy as np

from wnn.evaluation import _score_component_moments


def component_means(values: np.ndarray, components: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Average observable sequence descriptors within each complete component.

    These descriptors define retrospective evaluation panels, not serving inputs.
    Callers must supply input-only descriptors and source-qualified membership.
    """
    values = np.asarray(values, dtype=np.float64)
    components = np.asarray(components)
    if values.ndim != 1 or components.shape != values.shape or len(values) == 0:
        raise ValueError("need nonempty aligned descriptors and component IDs")
    if not np.isfinite(values).all():
        raise ValueError("nonfinite domain descriptor")
    identifiers, inverse = np.unique(components, return_inverse=True)
    sums = np.bincount(inverse, weights=values)
    return identifiers, sums / np.bincount(inverse)


def freeze_binary_panel(
    fitting_values: np.ndarray,
    fitting_components: np.ndarray,
    evaluation_values: np.ndarray,
    evaluation_components: np.ndarray,
    minimum_components: int = 20,
) -> dict:
    """Fit one median cutoff without evaluation labels or evaluation cutoffs.

    Every sequence in a component receives its component's domain. An undersized
    stratum is recorded as unavailable, never repaired using evaluation outcomes.
    """
    if minimum_components < 1:
        raise ValueError("minimum component count must be positive")
    fit_ids, fit_values = component_means(fitting_values, fitting_components)
    eval_ids, eval_values = component_means(evaluation_values, evaluation_components)
    if np.intersect1d(fit_ids, eval_ids).size:
        raise ValueError("domain fitting and evaluation components overlap")
    cutoff = float(np.median(fit_values))
    domain = (eval_values > cutoff).astype(np.int64)
    count = np.bincount(domain, minlength=2)
    return {
        "cutoff": cutoff,
        "cutoff_measure": "median of fitting component mean descriptors",
        "high_rule": "component descriptor > cutoff",
        "fit_component_count": len(fit_ids),
        "evaluation_component_ids": eval_ids.tolist(),
        "evaluation_component_domains": domain.tolist(),
        "component_counts": count.tolist(),
        "minimum_components": minimum_components,
        "status": "available" if np.all(count >= minimum_components) else "undersized",
    }


def mixture_counts(
    components: np.ndarray, domains: np.ndarray, high_fraction: float, minimum_components: int = 20
) -> np.ndarray:
    """Return sequence multipliers for a specified component-domain mixture.

    Sampling is uniform over complete components within each domain, retaining
    all selected rows and original label-magnitude weights within a component.
    Thus this is a component-frequency shift, not a fixed row-mass mixture.
    """
    components, domains = np.asarray(components), np.asarray(domains)
    if components.ndim != 1 or domains.shape != components.shape or len(components) == 0:
        raise ValueError("need aligned component and domain vectors")
    if not np.isin(domains, [0, 1]).all() or not 0 < high_fraction < 1:
        raise ValueError("need binary domains and a mixture strictly inside (0,1)")
    if minimum_components < 1:
        raise ValueError("minimum component count must be positive")
    identifiers = np.unique(components)
    component_domains = []
    for identifier in identifiers:
        assigned = np.unique(domains[components == identifier])
        if len(assigned) != 1:
            raise ValueError("a complete component cannot straddle domains")
        component_domains.append(int(assigned[0]))
    counts = np.bincount(component_domains, minlength=2)
    if np.any(counts < minimum_components):
        raise ValueError("stress stratum has too few independent components")
    multipliers = len(identifiers) * np.array([1 - high_fraction, high_fraction]) / counts
    return multipliers[domains.astype(np.int64)]


def paired_mixture_scores(
    candidate: np.ndarray,
    reference: np.ndarray,
    components: np.ndarray,
    domains: np.ndarray,
    minimum_components: int = 20,
) -> list[dict[str, float | str]]:
    """Recompute pooled WP from centered moments for the two frozen stress mixes.

    Inputs are aligned per-sequence moments from paired_component_moments using
    the official mask. This reports sensitivity, not a new official metric or a
    confidence interval. Reference predictions must belong to the fixed bridge.
    """
    candidate, reference = np.asarray(candidate), np.asarray(reference)
    if candidate.shape != reference.shape or candidate.shape != (len(components), 2, 6):
        raise ValueError("need aligned per-sequence centered moments shaped (N,2,6)")
    if not (np.isfinite(candidate).all() and np.isfinite(reference).all()):
        raise ValueError("nonfinite centered moments")
    if (candidate[:, :, 0] < 0).any() or (reference[:, :, 0] < 0).any():
        raise ValueError("negative moment mass")
    # Equal label moments establish that both predictions use the same rows and weights.
    if not np.allclose(
        candidate[:, :, [0, 1, 3]], reference[:, :, [0, 1, 3]], rtol=1e-12, atol=1e-12
    ):
        raise ValueError("candidate and reference label moments differ")
    output = []
    for fraction in (0.25, 0.75):
        counts = mixture_counts(components, domains, fraction, minimum_components)
        score = _score_component_moments(candidate, counts)
        baseline = _score_component_moments(reference, counts)
        output.append(
            {
                "measure": "component_frequency_mixture_official_mask",
                "high_fraction": fraction,
                "candidate_wp": score,
                "reference_wp": baseline,
                "paired_delta": score - baseline,
            }
        )
    return output
