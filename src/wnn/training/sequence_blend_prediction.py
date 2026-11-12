from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from wnn.splits import CampaignSplit
from wnn.training.sequence_cache_ensemble_lineage import validate_producer
from wnn.training.sequence_provenance import canonical_sha256
from wnn.training.sequence_reference_cache import ReferenceCache, validate_reference_identity

# Endpoint receipts are heterogeneous JSON, validated before their fields are consumed.
BLEND_KIND = "sequence-reference-blend-v1"


def prediction_reference(
    path: Path | None,
    requested_alpha: float | None,
    payload: dict[str, Any],
    split: CampaignSplit,
    members: list[tuple[str, int]],
    membership: str,
) -> tuple[ReferenceCache | None, float]:
    """Admit a same-reference fixed mixture, never an unbound treatment-head export."""
    objective_alpha = payload["config"].get("blend_objective_alpha", 0.0)
    if path is None:
        if objective_alpha > 0 or requested_alpha is not None:
            raise ValueError("composite endpoints require --reference-cache")
        return None, 0.0
    if membership != "evaluation":
        raise ValueError("composite fit/KD export is not implemented in this initial experiment")
    if payload["config"]["model"]["hint_dim"] != 0:
        raise ValueError("composite endpoints cannot export hidden hints")
    alpha = objective_alpha if requested_alpha is None else requested_alpha
    if isinstance(alpha, bool) or not isinstance(alpha, (float, int)):
        raise ValueError("explicit finite --blend-alpha is required for the standalone control")
    if not math.isfinite(alpha) or not 0 < alpha <= 1:
        raise ValueError("blend alpha must be finite and in (0, 1]")
    if objective_alpha > 0 and alpha != objective_alpha:
        raise ValueError("evaluation blend differs from the frozen training objective")
    reference = ReferenceCache(path, split, members, membership="evaluation")
    trained_reference = payload["data"].get("reference")
    if objective_alpha > 0:
        fitting = [("train", g) for g in split.train_groups]
        fitting += [("valid", g) for g in split.valid_groups]
        validate_reference_identity(trained_reference, split.sha256, fitting)
        if trained_reference["reference_sha256"] != reference.identity["reference_sha256"]:
            raise ValueError("evaluation and fitting reference models differ")
    elif trained_reference is not None:
        raise ValueError("standalone control unexpectedly contains fitting-reference ancestry")
    return reference, float(alpha)


def blend_producer(
    candidate: dict[str, Any], reference: dict[str, Any], alpha: float
) -> dict[str, Any]:
    """Bind candidate and complete native-reference ancestry into one scored endpoint."""
    composition = {
        "method": "linear_fp32_no_clip",
        "alpha": alpha,
        "candidate_checkpoint_sha256": candidate["checkpoint_sha256"],
        "reference_sha256": reference["reference_sha256"],
    }
    return {
        "kind": BLEND_KIND,
        "split_sha256": candidate["split_sha256"],
        "fit_members": candidate["fit_members"],
        "candidate": candidate,
        "reference": reference,
        "composition": composition,
        "composition_sha256": canonical_sha256(composition),
    }


def validate_blend_producer(
    producer: dict[str, Any], split_hash: str, fit_members: list[tuple[str, int]]
) -> None:
    """Validate a scored mixture without disguising native ancestors as sequence weights."""
    if producer.get("kind") != BLEND_KIND or producer.get("split_sha256") != split_hash:
        raise ValueError("unsupported composite producer or split mismatch")
    if producer.get("fit_members") != [list(member) for member in fit_members]:
        raise ValueError("composite producer fitting membership mismatch")
    candidate, reference = producer.get("candidate"), producer.get("reference")
    if not isinstance(candidate, dict) or not isinstance(reference, dict):
        raise ValueError("composite producer requires candidate and reference identities")
    validate_producer(candidate, split_hash, fit_members)
    validate_reference_identity(reference, split_hash, fit_members)
    composition = producer.get("composition")
    if not isinstance(composition, dict):
        raise ValueError("composite producer is missing its composition")
    alpha = composition.get("alpha")
    if isinstance(alpha, bool) or not isinstance(alpha, (float, int)):
        raise ValueError("invalid composition alpha")
    if not math.isfinite(alpha) or not 0 < alpha <= 1:
        raise ValueError("invalid composition alpha")
    expected = blend_producer(candidate, reference, alpha)
    if producer != expected:
        raise ValueError("composition fields or hashes do not match its two producers")
    objective_alpha = candidate.get("config", {}).get("blend_objective_alpha", 0.0)
    if objective_alpha > 0 and objective_alpha != alpha:
        raise ValueError("scored composition differs from the frozen objective")
    trained_reference = candidate.get("reference")
    if objective_alpha > 0:
        validate_reference_identity(trained_reference, split_hash, fit_members)
        if trained_reference["reference_sha256"] != reference["reference_sha256"]:
            raise ValueError("scored reference differs from the training reference")
    elif trained_reference is not None:
        raise ValueError("standalone candidate cannot have fitting-reference ancestry")
