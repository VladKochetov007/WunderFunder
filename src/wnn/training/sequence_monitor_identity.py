from __future__ import annotations

import hashlib
import json
import re

import torch

from wnn.training.sequence_fit_utils import SequenceDataset


def membership_sha256(members: list[tuple[str, int]]) -> str:
    """Hash exact source-qualified membership without sorting identifiers."""
    encoded = json.dumps([list(member) for member in members], separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def mask_definition(mask: str) -> str:
    """Describe the mask actually applied to each monitoring membership."""
    if mask == "all_after_warmup":
        return "all rows after row 99 warmup; actual fitting objective mask"
    if mask == "is_scored":
        return "official is_scored mask after row 99 warmup"
    raise ValueError(f"unsupported monitoring row mask: {mask}")


def mask_comparison(train_eval_mask: str) -> str:
    """Describe whether this run's train-eval and valid cohorts share a row mask."""
    if train_eval_mask == "is_scored":
        return "train_eval and valid both use the official is_scored mask"
    if train_eval_mask == "all_after_warmup":
        return "train_eval uses all_after_warmup; valid uses official is_scored; masks differ"
    raise ValueError(f"unsupported train-eval mask: {train_eval_mask}")


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
