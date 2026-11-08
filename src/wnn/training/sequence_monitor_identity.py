from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence

import torch

from wnn.training.sequence_fit_utils import SequenceDataset

TRAIN_EVAL_HASH_RANK_NAMESPACE = "wnn.train_eval_selection.hash_rank_v1"


def membership_sha256(members: list[tuple[str, int]]) -> str:
    """Hash exact source-qualified membership without sorting identifiers."""
    encoded = json.dumps([list(member) for member in members], separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def hash_rank_train_eval_members(members: Sequence[tuple[str, int]]) -> list[tuple[str, int]]:
    """Rank source-qualified members deterministically without consulting model seed or order."""
    if len(set(members)) != len(members):
        raise ValueError("hash-ranked train-eval membership contains duplicate members")
    ranked = []
    for source, group in members:
        payload = json.dumps(
            [TRAIN_EVAL_HASH_RANK_NAMESPACE, source, int(group)], separators=(",", ":")
        ).encode()
        ranked.append((hashlib.sha256(payload).digest(), source, int(group)))
    return [(source, group) for _, source, group in sorted(ranked)]


def train_eval_comparability(
    train_eval_mask: str,
    evaluation_mask: str,
    evaluation_role: str,
    selection: str,
    train_eval_count: int,
    evaluation_count: int,
) -> dict[str, object]:
    """Describe diagnostic limits without treating in-sample fit data as validation."""
    if evaluation_role not in ("valid", "control"):
        raise ValueError(f"unsupported evaluation role: {evaluation_role}")
    reason = (
        "train_eval is an in-sample fitting subset; official valid is the full independent evaluation cohort"
        if evaluation_role == "valid"
        else "train_eval and inner control are fitting-only diagnostics; neither is independent validation"
    )
    comparison: dict[str, object] = {
        "evaluation_role": evaluation_role,
        "comparable_to_evaluation": False,
        "gap_is_calibrated_generalization_estimate": False,
        "reason": reason,
        "train_eval_mask_matches_evaluation_mask": train_eval_mask == evaluation_mask,
        "train_eval_selection": selection,
        "train_eval_sequence_count": train_eval_count,
        "evaluation_sequence_count": evaluation_count,
        "selection_claims_representativeness": False,
    }
    if evaluation_role == "valid":
        comparison["comparable_to_primary_valid"] = False
        comparison["official_valid_sequence_count"] = evaluation_count
    return comparison


def train_eval_context(
    members: list[tuple[str, int]],
    evaluation_mask: str,
    evaluation_count: int,
    selection: str,
    train_eval_mask: str,
    input_feature_mode: str,
    evaluation_role: str,
) -> dict[str, object]:
    """Build the immutable event metadata for an in-sample diagnostic cohort."""
    algorithms = {
        "prefix": "legacy_fit_membership_prefix",
        "hash_rank_v1": TRAIN_EVAL_HASH_RANK_NAMESPACE,
    }
    if selection not in algorithms:
        raise ValueError(f"unsupported train-eval selection: {selection}")
    order_basis = (
        "SHA256 rank of opaque source-qualified group keys; no temporal order inferred"
        if selection == "hash_rank_v1"
        else "source-qualified fit-membership order; no temporal order inferred"
    )
    return {
        "evaluation_scope": "fixed source-qualified fitting cohort; diagnostic only",
        "source_scope": "actual fitting membership only",
        "row_mask": train_eval_mask,
        "mask_definition": mask_definition(train_eval_mask),
        "member_count": len(members),
        "member_sha256": membership_sha256(members),
        "source_names": sorted({source for source, _ in members}),
        "selection": selection,
        "selection_algorithm": algorithms[selection],
        "selection_order_basis": order_basis,
        "selection_independent_of_model_seed": True,
        "selection_claims_representativeness": False,
        "comparison_to_evaluation": train_eval_comparability(
            train_eval_mask,
            evaluation_mask,
            evaluation_role,
            selection,
            len(members),
            evaluation_count,
        ),
        "input_feature_mode": input_feature_mode,
    }


def mask_definition(mask: str) -> str:
    """Describe the mask actually applied to each monitoring membership."""
    if mask == "all_after_warmup":
        return "all rows after row 99 warmup; actual fitting objective mask"
    if mask == "is_scored":
        return "official is_scored mask after row 99 warmup"
    raise ValueError(f"unsupported monitoring row mask: {mask}")


def mask_comparison(
    train_eval_mask: str, evaluation_mask: str | None = None, role: str = "valid"
) -> str:
    """Describe how a fitting diagnostic mask compares with this evaluation role."""
    if evaluation_mask is None:
        if train_eval_mask == "is_scored":
            return "train_eval and valid both use the official is_scored mask"
        if train_eval_mask == "all_after_warmup":
            return "train_eval uses all_after_warmup; valid uses official is_scored; masks differ"
        raise ValueError(f"unsupported train-eval mask: {train_eval_mask}")
    train_description = mask_definition(train_eval_mask)
    evaluation_description = mask_definition(evaluation_mask)
    if train_eval_mask == evaluation_mask:
        return f"train_eval and {role} use {train_description}"
    return f"train_eval uses {train_description}; {role} uses {evaluation_description}"


def validate_dataset_members(
    dataset: SequenceDataset,
    expected: list[tuple[str, int]],
    scope: str,
    expected_mask: str,
) -> None:
    """Check canonical membership and the declared row-mask contract."""
    actual = getattr(dataset, "members", None)
    if actual is not None and list(actual) != expected:
        raise ValueError(f"{scope} dataset membership differs from its recorded cohort")
    actual_mask = getattr(dataset, "training_mask", "is_scored")
    if actual_mask != expected_mask:
        label = "official is_scored" if expected_mask == "is_scored" else expected_mask
        raise ValueError(f"{scope} reporting cohort must use the {label} mask")


def gpu_identifier(device: torch.device) -> str | None:
    """Resolve the physical UUID for this CUDA worker without guessing visible ordinals."""
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    try:
        identifier = getattr(torch.cuda.get_device_properties(device), "uuid", None)
    except (AssertionError, RuntimeError, ValueError):
        return None
    if isinstance(identifier, bytes):
        identifier = identifier.decode("ascii", errors="ignore")
    elif identifier is not None and not isinstance(identifier, str):
        identifier = str(identifier)
    if not isinstance(identifier, str) or not identifier:
        return None
    if re.fullmatch(
        r"[0-9A-Fa-f]{32}|[0-9A-Fa-f]{8}(?:-[0-9A-Fa-f]{4}){3}-[0-9A-Fa-f]{12}",
        identifier,
    ):
        return f"GPU-{identifier}"
    return identifier
