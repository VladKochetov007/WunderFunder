"""Bounded, sequence-local relational knowledge distillation for two outputs."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _validate_rkd_inputs(
    teacher: torch.Tensor,
    student: torch.Tensor,
    valid_mask: torch.Tensor | None,
    *,
    pair_seed: int,
    max_pairs: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if not isinstance(teacher, torch.Tensor) or not isinstance(student, torch.Tensor):
        raise TypeError("teacher and student must be torch tensors")
    if teacher.ndim != 3 or teacher.shape != student.shape or teacher.shape[-1] != 2:
        raise ValueError("teacher and student must have matching shape (batch, rows, 2)")
    if not teacher.is_floating_point() or not student.is_floating_point():
        raise TypeError("teacher and student must have floating-point dtype")
    if teacher.device != student.device:
        raise ValueError("teacher and student must be on the same device")
    if type(pair_seed) is not int:
        raise TypeError("pair_seed must be an integer")
    if type(max_pairs) is not int or max_pairs < 2:
        raise ValueError("max_pairs must be an integer of at least 2")
    if student.device.type not in ("cpu", "cuda"):
        raise ValueError("relational distillation supports CPU and CUDA tensors")

    if valid_mask is not None:
        if not isinstance(valid_mask, torch.Tensor):
            raise TypeError("valid_mask must be a torch tensor or None")
        if valid_mask.shape != teacher.shape[:2] or valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be a boolean (batch, rows) tensor")
        if valid_mask.device != student.device:
            raise ValueError("valid_mask must be on the same device as student")

    work_dtype = torch.float64 if torch.float64 in (teacher.dtype, student.dtype) else torch.float32
    teacher_work = teacher.detach().to(dtype=work_dtype)
    student_work = student.to(dtype=work_dtype)

    if valid_mask is None:
        teacher_valid = teacher_work
        student_valid = student_work
    else:
        teacher_valid = teacher_work[valid_mask]
        student_valid = student_work[valid_mask]
    teacher_finite = torch.isfinite(teacher_valid).all()
    student_finite = torch.isfinite(student_valid).all()
    finite_inputs = teacher_finite & student_finite
    if not finite_inputs.item():
        if not teacher_finite.item():
            raise ValueError("teacher contains non-finite values at valid positions")
        raise ValueError("student contains non-finite values at valid positions")
    return teacher_work, student_work, valid_mask


def _sample_pair_indices(
    positions: torch.Tensor,
    generator: torch.Generator,
    max_pairs: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw disjoint valid endpoints in O(rows + max_pairs) memory."""
    pair_count = min(max_pairs, positions.numel() // 2)
    if pair_count < 2:
        empty = positions[:0]
        return empty, empty
    permutation = torch.randperm(positions.numel(), device=positions.device, generator=generator)
    selected = positions[permutation[: 2 * pair_count]]
    return selected[:pair_count], selected[pair_count:]


def sampled_relation_loss(
    teacher: torch.Tensor,
    student: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    *,
    pair_seed: int,
    max_pairs: int = 256,
) -> torch.Tensor:
    """Return a mean per-sequence normalized pair-distance Smooth-L1 loss.

    Teacher values are detached. The validity mask admits only pairs whose
    endpoints are eligible. Sampling uses one local generator; global Python,
    NumPy, and PyTorch RNG states are not consumed. A sequence contributes
    only when at least two pairs are drawn and its sampled teacher distances
    are not all zero.
    """
    teacher_work, student_work, valid_mask = _validate_rkd_inputs(
        teacher, student, valid_mask, pair_seed=pair_seed, max_pairs=max_pairs
    )
    generator = torch.Generator(device=student.device)
    generator.manual_seed(pair_seed)
    sequence_losses: list[torch.Tensor] = []
    sequence_contributes: list[torch.Tensor] = []
    invalid_distance = torch.zeros((), dtype=torch.bool, device=student.device)
    batch_size, row_count, _ = teacher.shape
    if valid_mask is None:
        positions_by_sequence = [torch.arange(row_count, device=student.device)] * batch_size
    else:
        valid_coordinates = valid_mask.nonzero(as_tuple=False)
        valid_counts = torch.bincount(valid_coordinates[:, 0], minlength=batch_size).tolist()
        positions_by_sequence = list(torch.split(valid_coordinates[:, 1], valid_counts))

    for batch_index in range(batch_size):
        left, right = _sample_pair_indices(positions_by_sequence[batch_index], generator, max_pairs)
        if left.numel() < 2:
            continue

        teacher_distances = torch.linalg.vector_norm(
            teacher_work[batch_index, left] - teacher_work[batch_index, right], dim=-1
        )
        student_distances = torch.linalg.vector_norm(
            student_work[batch_index, left] - student_work[batch_index, right], dim=-1
        )
        invalid_distance = (
            invalid_distance
            | (~torch.isfinite(teacher_distances) | ~torch.isfinite(student_distances)).any()
        )

        tiny = torch.finfo(teacher_distances.dtype).tiny
        teacher_mean = teacher_distances.mean().clamp_min(tiny)
        student_mean = student_distances.mean().clamp_min(tiny)
        teacher_normalized = teacher_distances / teacher_mean
        student_normalized = student_distances / student_mean
        pair_losses = F.smooth_l1_loss(student_normalized, teacher_normalized, reduction="none")
        sequence_losses.append(pair_losses.mean())
        sequence_contributes.append((teacher_distances != 0).any())

    if not sequence_losses:
        # Empty slicing keeps a graph edge without reading masked-out NaNs.
        return student_work.reshape(-1)[:0].sum()

    checks = torch.stack((invalid_distance, torch.stack(sequence_contributes).any()))
    invalid_distance_found, contributes_found = checks.tolist()
    if invalid_distance_found:
        raise ValueError("computed pair distances are non-finite at valid positions")

    losses = torch.stack(sequence_losses)
    contributes = torch.stack(sequence_contributes)
    if not contributes_found:
        # The empty slice keeps a graph edge while avoiding a masked-value sum.
        return student_work.reshape(-1)[:0].sum()
    return losses[contributes].mean()


def add_relation_loss(
    label_loss: torch.Tensor,
    teacher: torch.Tensor,
    student: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    *,
    weight: float,
    pair_seed: int,
    max_pairs: int = 256,
) -> torch.Tensor:
    """Add the relation scalar, preserving exact identity when disabled."""
    if weight == 0:
        return label_loss
    if isinstance(weight, bool) or not isinstance(weight, (int, float)):
        raise TypeError("weight must be a Python number")
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("weight must be finite and nonnegative")
    relation = sampled_relation_loss(
        teacher, student, valid_mask, pair_seed=pair_seed, max_pairs=max_pairs
    )
    return label_loss + weight * relation


def validate_rkd_configuration(
    *,
    weight: float,
    max_pairs: int,
    has_source_cache: bool,
    cache_variant: str,
    architecture: str,
    auxiliary_quote: float,
    calibrating_reference: bool,
    controlled_loss: bool,
    classification_joint: bool,
) -> None:
    """Fail early for enabled configurations that cannot supply raw source rows."""
    if weight == 0:
        return
    if isinstance(weight, bool) or not isinstance(weight, (int, float)):
        raise ValueError("--distill-rkd-weight must be a Python number")
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("--distill-rkd-weight must be finite and nonnegative")
    if max_pairs < 2:
        raise ValueError("--distill-rkd-pairs must be at least 2")
    if not has_source_cache:
        raise ValueError("--distill-rkd-weight requires --distill-cache-root")
    if cache_variant != "source":
        raise ValueError("--distill-rkd-weight requires --distill-cache-variant source")
    if architecture == "specialist":
        raise ValueError("--distill-rkd-weight does not support --arch specialist")
    if auxiliary_quote > 0:
        raise ValueError("--distill-rkd-weight cannot be combined with --aux-quote")
    if calibrating_reference:
        raise ValueError("--distill-rkd-weight is unsupported during --calibrate-distill-reference")
    if controlled_loss:
        raise ValueError(
            "--distill-rkd-weight is unsupported with controlled microbatch loss modes"
        )
    if classification_joint:
        raise ValueError("--distill-rkd-weight is unsupported with --loss ce_joint")
