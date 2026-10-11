"""Small, explicit operations for cached multi-family distillation."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True, slots=True)
class TeacherSource:
    """One fold-bound teacher prediction and its coefficient in the main target."""

    family: str
    run: str
    coefficient: float
    epoch: int = 3


def combine_teacher_chunk(
    predictions: Mapping[str, np.ndarray],
    sources: Sequence[TeacherSource],
    family_order: Sequence[str],
    homogeneous_family: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build source-scale, normalized and homogeneous targets plus four family targets.

    Args:
        predictions: Float32 arrays with shape ``(sequences, rows, 2)`` keyed by run.
        sources: Teacher members and their original source coefficients.
        family_order: Stable output order for the four family heads.
        homogeneous_family: Family whose mean prediction is scaled for the homogeneous control.

    Returns:
        Source-scale main targets, normalized main targets, homogeneous targets, and
        family means with the last dimension ordered ``family, target``.

    Raises:
        ValueError: If the prediction arrays, family layout, or coefficients differ.
    """
    if len(family_order) != 4 or len(set(family_order)) != 4:
        raise ValueError("family order must name four distinct families")
    if not sources or homogeneous_family not in family_order:
        raise ValueError("a nonempty teacher set and declared homogeneous family are required")
    unknown_families = {source.family for source in sources} - set(family_order)
    if unknown_families or set(source.family for source in sources) != set(family_order):
        raise ValueError("teacher sources must cover exactly the four declared families")
    if len({source.run for source in sources}) != len(sources):
        raise ValueError("each teacher run must occur once in the source mixture")
    if set(source.run for source in sources) - predictions.keys():
        raise ValueError("a teacher prediction is missing")

    shape = predictions[sources[0].run].shape
    if len(shape) != 3 or shape[-1] != 2:
        raise ValueError("teacher predictions must have shape (sequences, rows, 2)")
    total = 0.0
    for source in sources:
        values = predictions[source.run]
        if values.shape != shape or values.dtype != np.float32:
            raise ValueError("teacher prediction shapes and dtypes must match float32")
        if not math.isfinite(source.coefficient) or source.coefficient <= 0:
            raise ValueError("teacher coefficients must be finite and positive")
        if not np.isfinite(values).all():
            raise ValueError(f"teacher prediction contains nonfinite values: {source.run}")
        total += source.coefficient

    main = np.zeros(shape, dtype=np.float32)
    family_values: list[np.ndarray] = []
    for source in sources:
        main += np.float32(source.coefficient) * predictions[source.run]
    for family in family_order:
        members = [predictions[source.run] for source in sources if source.family == family]
        family_values.append(np.add.reduce(members, dtype=np.float32) / np.float32(len(members)))
    families = np.stack(family_values, axis=-2).reshape(*shape[:-1], 8)
    normalized = main / np.float32(total)
    homogeneous = np.float32(total) * family_values[list(family_order).index(homogeneous_family)]
    return main, normalized, homogeneous, families


def extend_family_output_head(
    parent_state: Mapping[str, torch.Tensor], initialized_state: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Copy the warm-start readout while retaining eight new family outputs.

    The parent may already have auxiliary output rows after its two main targets.
    Those rows remain in place; the new family rows are appended after them.
    All trunk and non-output tensors must remain shape compatible.

    Args:
        parent_state: State from a clean two-output parent checkpoint.
        initialized_state: State of the student with its extra eight outputs.

    Returns:
        A complete student state dictionary with parent weights copied in.

    Raises:
        ValueError: If the only permitted shape change is not eight output rows.
    """
    if parent_state.keys() != initialized_state.keys():
        raise ValueError("parent and student checkpoint tensor names differ")
    merged = {name: value.clone() for name, value in initialized_state.items()}
    for name, parent_value in parent_state.items():
        student_value = initialized_state[name]
        if name in ("head.2.weight", "head.2.bias"):
            if (
                parent_value.ndim != student_value.ndim
                or parent_value.shape[0] < 2
                or student_value.shape[0] != parent_value.shape[0] + 8
                or parent_value.shape[1:] != student_value.shape[1:]
            ):
                raise ValueError("family distillation requires exactly eight appended output rows")
            merged[name][: parent_value.shape[0]].copy_(parent_value)
        elif parent_value.shape != student_value.shape:
            raise ValueError(f"warm-start tensor shape differs outside family head: {name}")
        else:
            merged[name].copy_(parent_value)
    return merged


def append_family_output_head(model: torch.nn.Module) -> torch.nn.Module:
    """Append eight readout rows without advancing the legacy CPU RNG stream.

    The family head is initialized only for the opt-in family-distillation arm.
    Construct it on the CPU under a forked RNG and copy the original readout
    rows exactly, so subsequent global-RNG consumers retain the order produced
    by the same model with its original output width.

    Args:
        model: An unplaced model with a ``Sequential`` head ending in ``Linear``.

    Returns:
        The same model with eight initialized output rows appended.

    Raises:
        ValueError: If the model has no supported CPU linear output head.
    """
    head = getattr(model, "head", None)
    if not isinstance(head, torch.nn.Sequential) or not isinstance(head[-1], torch.nn.Linear):
        raise ValueError("family distillation requires a sequential linear output head")
    output = head[-1]
    if output.weight.device.type != "cpu":
        raise ValueError("append the family output head before moving the model to a device")

    old_width = output.out_features
    with torch.random.fork_rng(devices=[]):
        expanded = torch.nn.Linear(
            output.in_features,
            old_width + 8,
            bias=output.bias is not None,
            device=output.weight.device,
            dtype=output.weight.dtype,
        )
    with torch.no_grad():
        expanded.weight[:old_width].copy_(output.weight)
        if output.bias is not None:
            assert expanded.bias is not None
            expanded.bias[:old_width].copy_(output.bias)
    head[-1] = expanded
    return model


def four_family_weighted_mse(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    target_weights: torch.Tensor,
) -> torch.Tensor:
    """Average four family wMSE losses with the main labels' unchanged row weights."""
    if predictions.shape != targets.shape or predictions.shape[-2:] != (4, 2):
        raise ValueError("family predictions and targets must end in four families by two targets")
    if target_weights.shape != predictions.shape[:-2] + (2,):
        raise ValueError("family loss weights must match the two main targets")
    weights = target_weights.unsqueeze(-2)
    denominator = (weights.sum() * 4).clamp_min(1e-8)
    return (weights * (predictions.float() - targets.float()).square()).sum() / denominator


def family_targets_for_loss(targets: torch.Tensor, mode: str = "clip2") -> torch.Tensor:
    """Transform family-head targets for training without changing serving labels.

    Args:
        targets: Cached raw teacher outputs with the family and target dimensions.
        mode: ``clip2`` preserves the historical behavior; ``raw`` keeps teacher values.

    Returns:
        Targets with the original shape and either clipped or unchanged values.

    Raises:
        ValueError: If the requested target mode is unsupported.
    """
    if mode == "clip2":
        return targets.clamp(-2, 2)
    if mode == "raw":
        return targets
    raise ValueError(f"unknown family target mode: {mode}")
