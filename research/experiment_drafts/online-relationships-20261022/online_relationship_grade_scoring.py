"""Score fixed endpoints and paired fold-local contrasts using canonical metrics."""

from __future__ import annotations

from typing import Any  # Report records have runtime-defined score and interval fields.

import numpy as np
from online_relationship_grade_contract import ARMS, CandidateArtifacts, ParentCacheArtifacts
from online_relationship_grade_core import (
    ROWS,
    official_scores,
    paired_component_contrast,
    paired_endpoint_moments,
)

PRIMARY_PAIRS = (("C", "A"), ("C", "B"))


def grade_fold(
    fold_id: int,
    parent: ParentCacheArtifacts,
    candidates: dict[tuple[int, int, str], CandidateArtifacts],
    targets: np.ndarray,
    mask: np.ndarray,
    components: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Score seven fixed endpoints and form thirteen fold-local paired contrasts."""
    scored_mask = mask & (np.arange(ROWS)[None, :] >= 99)
    endpoints: dict[str, np.ndarray] = {"parent": parent.prediction}
    endpoint_identity = {"parent": parent.identity}
    for lag in (1, 16):
        for arm in ARMS:
            name = f"{arm}_lag{lag}"
            artifact = candidates[(fold_id, lag, arm)]
            endpoints[name] = artifact.prediction
            endpoint_identity[name] = artifact.identity

    endpoint_scores = {
        name: official_scores(targets, mask, prediction) for name, prediction in endpoints.items()
    }
    moments = {
        name: paired_endpoint_moments(targets, prediction, scored_mask)
        for name, prediction in endpoints.items()
    }
    contrasts: list[dict[str, Any]] = []

    def add_contrast(name: str, candidate: str, reference: str) -> None:
        contrasts.append(
            {
                "contrast": name,
                **paired_component_contrast(
                    targets,
                    mask,
                    endpoints[candidate],
                    endpoints[reference],
                    components,
                    endpoint_scores[candidate],
                    endpoint_scores[reference],
                    moments[candidate],
                    moments[reference],
                ),
            }
        )

    for lag in (1, 16):
        for candidate_arm, reference_arm in PRIMARY_PAIRS:
            add_contrast(
                f"{candidate_arm}-{reference_arm}_lag{lag}",
                f"{candidate_arm}_lag{lag}",
                f"{reference_arm}_lag{lag}",
            )
        for arm in ARMS:
            add_contrast(f"{arm}-parent_lag{lag}", f"{arm}_lag{lag}", "parent")
    for arm in ARMS:
        add_contrast(f"lag16-lag1_{arm}", f"{arm}_lag16", f"{arm}_lag1")
    if len(contrasts) != 13:
        raise AssertionError("fold contrast construction did not produce thirteen pairs")
    return (
        {
            "fold": fold_id,
            "evaluation_sequences": len(targets),
            "evaluation_components": int(len(np.unique(components))),
            "parent": endpoint_identity["parent"],
            "endpoints": {
                name: {"identity": endpoint_identity[name], "scores": endpoint_scores[name]}
                for name in endpoints
            },
        },
        contrasts,
    )
