from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from wnn.rkd_component import (
    _sample_pair_indices,
    add_relation_loss,
    sampled_relation_loss,
    validate_rkd_configuration,
)


def _geometry() -> torch.Tensor:
    return torch.tensor(
        [
            [0.0, 0.0],
            [0.2, 1.1],
            [2.4, -0.3],
            [1.2, 3.8],
            [-2.0, 1.7],
            [4.1, 2.2],
            [3.3, -3.1],
            [-1.5, -2.7],
        ],
        dtype=torch.float32,
    )[None]


def _numpy_rng_equal(first: tuple[object, ...], second: tuple[object, ...]) -> bool:
    return (
        first[0] == second[0]
        and np.array_equal(first[1], second[1])
        and first[2] == second[2]
        and first[3] == second[3]
        and first[4] == second[4]
    )


def test_zero_weight_returns_exact_loss_object_without_touching_inputs_or_rng() -> None:
    label_loss = object()
    python_state = random.getstate()
    numpy_state = np.random.get_state()  # noqa: NPY002 - snapshot the process-global legacy RNG.
    torch_state = torch.random.get_rng_state()

    result = add_relation_loss(
        label_loss,  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        weight=0.0,
        pair_seed=-1,
        max_pairs=0,
    )

    assert result is label_loss
    assert random.getstate() == python_state
    assert _numpy_rng_equal(np.random.get_state(), numpy_state)  # noqa: NPY002 - global-state check.
    assert torch.equal(torch.random.get_rng_state(), torch_state)


def test_relation_is_zero_for_identical_outputs_and_scale_translation_invariant() -> None:
    teacher = _geometry()
    baseline = sampled_relation_loss(teacher, teacher.clone(), pair_seed=42, max_pairs=4)
    assert baseline.item() == pytest.approx(0.0, abs=1e-7)

    for scale in (1e-3, 1.0, 1e3):
        student = scale * teacher + torch.tensor([17.0, -9.0])
        loss = sampled_relation_loss(teacher, student, pair_seed=42, max_pairs=4)
        assert loss.item() == pytest.approx(baseline.item(), abs=2e-5)


def test_time_order_perturbation_changes_noncollinear_geometry() -> None:
    teacher = _geometry()
    student = teacher[:, torch.tensor([2, 5, 0, 7, 1, 4, 6, 3])]
    matched = sampled_relation_loss(teacher, teacher.clone(), pair_seed=7, max_pairs=4)
    perturbed = sampled_relation_loss(teacher, student, pair_seed=7, max_pairs=4)
    assert matched.item() < 1e-7
    assert perturbed.item() > matched.item() + 1e-3


def test_zero_teacher_distance_pairs_remain_in_the_relation_mean() -> None:
    generator = torch.Generator().manual_seed(3)
    left, right = _sample_pair_indices(torch.arange(4), generator, 2)
    teacher = torch.zeros((1, 4, 2), dtype=torch.float32)
    with torch.no_grad():
        teacher[0, right[1], 0] = 1.0
    student = teacher.clone()
    with torch.no_grad():
        student[0, left[0], 0] = 1.0

    loss = sampled_relation_loss(teacher, student, pair_seed=3, max_pairs=2)
    assert loss.item() > 0.0


def test_one_pair_and_collapsed_teacher_are_skipped_with_graph_zero() -> None:
    student = torch.randn((1, 3, 2), requires_grad=True)
    one_pair = sampled_relation_loss(torch.randn_like(student), student, pair_seed=1)
    assert one_pair.item() == 0.0
    assert one_pair.requires_grad

    collapsed_teacher = torch.zeros((1, 8, 2))
    collapsed_student = torch.randn((1, 8, 2), requires_grad=True)
    collapsed_loss = sampled_relation_loss(collapsed_teacher, collapsed_student, pair_seed=2)
    assert collapsed_loss.item() == 0.0
    collapsed_loss.backward()
    assert collapsed_student.grad is not None
    assert torch.count_nonzero(collapsed_student.grad).item() == 0


def test_masked_nonfinite_values_are_ignored_but_valid_nonfinite_values_raise() -> None:
    teacher = _geometry()
    student = teacher.clone()
    teacher[0, 7] = float("nan")
    student[0, 7] = float("inf")
    mask = torch.ones((1, 8), dtype=torch.bool)
    mask[0, 7] = False
    loss = sampled_relation_loss(teacher, student, mask, pair_seed=12, max_pairs=4)
    assert torch.isfinite(loss)

    mask[0, 7] = True
    with pytest.raises(ValueError, match="teacher contains non-finite"):
        sampled_relation_loss(teacher, student, mask, pair_seed=12)
    teacher[0, 7] = 1.0
    with pytest.raises(ValueError, match="student contains non-finite"):
        sampled_relation_loss(teacher, student, mask, pair_seed=12)


def test_empty_mask_returns_finite_graph_zero_even_for_masked_nan() -> None:
    teacher = torch.full((2, 8, 2), float("nan"))
    student = torch.full((2, 8, 2), float("nan"), requires_grad=True)
    mask = torch.zeros((2, 8), dtype=torch.bool)
    loss = sampled_relation_loss(teacher, student, mask, pair_seed=4)
    assert torch.isfinite(loss)
    assert loss.item() == 0.0
    loss.backward()
    assert student.grad is not None
    assert torch.count_nonzero(student.grad).item() == 0


def test_teacher_is_detached_student_gets_gradients_and_rng_is_local() -> None:
    teacher = _geometry().requires_grad_()
    student = (teacher.detach() + 0.1 * torch.randn_like(teacher)).requires_grad_()
    python_state = random.getstate()
    numpy_state = np.random.get_state()  # noqa: NPY002 - snapshot the process-global legacy RNG.
    torch_state = torch.random.get_rng_state()

    loss = sampled_relation_loss(teacher, student, pair_seed=99, max_pairs=4)
    loss.backward()

    assert teacher.grad is None
    assert student.grad is not None
    assert random.getstate() == python_state
    assert _numpy_rng_equal(np.random.get_state(), numpy_state)  # noqa: NPY002 - global-state check.
    assert torch.equal(torch.random.get_rng_state(), torch_state)


def test_same_seed_is_deterministic_and_pair_count_is_bounded_at_full_rows(monkeypatch):
    teacher = _geometry()
    first = sampled_relation_loss(teacher, teacher.roll(1, dims=1), pair_seed=101, max_pairs=4)
    second = sampled_relation_loss(teacher, teacher.roll(1, dims=1), pair_seed=101, max_pairs=4)
    assert torch.equal(first, second)

    calls: list[int] = []
    original = torch.randperm

    def tracked_randperm(n: int, *args: object, **kwargs: object) -> torch.Tensor:
        calls.append(n)
        return original(n, *args, **kwargs)

    monkeypatch.setattr(torch, "randperm", tracked_randperm)
    positions = torch.arange(20_000)
    left, right = _sample_pair_indices(positions, torch.Generator().manual_seed(7), 256)
    assert calls == [20_000]
    assert left.numel() == right.numel() == 256
    assert torch.unique(torch.cat((left, right))).numel() == 512
    assert positions.numel() == 20_000


def test_rkd_configuration_fails_closed_only_when_enabled() -> None:
    validate_rkd_configuration(
        weight=0.0,
        max_pairs=0,
        has_source_cache=False,
        cache_variant="normalized",
        architecture="specialist",
        auxiliary_quote=1.0,
        calibrating_reference=True,
        controlled_loss=True,
        classification_joint=True,
    )
    with pytest.raises(ValueError, match="requires --distill-cache-root"):
        validate_rkd_configuration(
            weight=0.1,
            max_pairs=256,
            has_source_cache=False,
            cache_variant="source",
            architecture="gru",
            auxiliary_quote=0.0,
            calibrating_reference=False,
            controlled_loss=False,
            classification_joint=False,
        )
    with pytest.raises(ValueError, match="requires --distill-cache-variant source"):
        validate_rkd_configuration(
            weight=0.1,
            max_pairs=256,
            has_source_cache=True,
            cache_variant="normalized",
            architecture="gru",
            auxiliary_quote=0.0,
            calibrating_reference=False,
            controlled_loss=False,
            classification_joint=False,
        )
    valid_arguments = {
        "weight": 0.1,
        "max_pairs": 256,
        "has_source_cache": True,
        "cache_variant": "source",
        "architecture": "gru",
        "auxiliary_quote": 0.0,
        "calibrating_reference": False,
        "controlled_loss": False,
        "classification_joint": False,
    }
    validate_rkd_configuration(**valid_arguments)
    with pytest.raises(ValueError, match="controlled microbatch"):
        validate_rkd_configuration(**(valid_arguments | {"controlled_loss": True}))
    with pytest.raises(ValueError, match="--loss ce_joint"):
        validate_rkd_configuration(**(valid_arguments | {"classification_joint": True}))
    valid_arguments = {
        "weight": 0.1,
        "max_pairs": 256,
        "has_source_cache": True,
        "cache_variant": "source",
        "architecture": "gru",
        "auxiliary_quote": 0.0,
        "calibrating_reference": False,
        "controlled_loss": False,
        "classification_joint": False,
    }
    validate_rkd_configuration(**valid_arguments)
    with pytest.raises(ValueError, match="controlled microbatch"):
        validate_rkd_configuration(**(valid_arguments | {"controlled_loss": True}))
    with pytest.raises(ValueError, match="--loss ce_joint"):
        validate_rkd_configuration(**(valid_arguments | {"classification_joint": True}))
