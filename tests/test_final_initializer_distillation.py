from __future__ import annotations

import argparse

import pytest
import torch

from tests.test_trainer_resume_integration import training
from wnn.nn import RNNModel
from wnn.training_qat import configure_qat, invalidate_qat_cache


def _valid_args(**changes: object) -> argparse.Namespace:
    values = {
        "final_distill_initializer": True,
        "final_model_adaptation": True,
        "init": "clean_parent@ep0",
        "inherit_init_preprocessing": True,
        "precision": "fp32",
        "arch": "gru",
        "cell": "gru",
        "enc": "linear",
        "recurrent_rank": 0,
        "distill_cache_root": "/cache/checked",
        "distill": "",
        "loss": "wmse",
        "loss_norm": "chunk",
        "distill_alpha": 0.9,
        "distill_alpha_far": None,
        "distill_boot_h": 0,
        "distill_reanchor": 0.0,
        "distill_cache_scale": 1.0,
        "distill_family_weight": 0.0,
        "update_rows": 0,
        "detach_rows": 0,
        "distill_denominator": "chunk",
        "detach_boundary_jitter": False,
        "origin_view": "off",
        "credit_mode": "deterministic",
        "ema_teacher": 0.0,
        "ema_metric_weight": False,
        "ema_reference_rows": 0,
        "eval_ema": False,
        "swa_frac": 0.0,
        "td_alpha": 0.0,
        "resume_state": None,
        "save_training_state": False,
    }
    values.update(changes)
    return argparse.Namespace(**values)


def test_initializer_teacher_chunk_state_cache_isolation_and_qat_freeze() -> None:
    torch.manual_seed(1984)
    student = RNNModel(3, hidden=4, layers=1)
    student._inference_fp32 = True
    teacher = training.make_final_initializer_teacher(student)
    assert teacher is not student
    assert not teacher.training
    assert all(not parameter.requires_grad for parameter in teacher.parameters())
    assert getattr(teacher, "_qat_preset", "none") == "none"

    inputs = torch.randn(2, 12, 3)
    with torch.no_grad():
        expected, _ = training.forward_chunk(teacher, inputs, slice(0, 12), None)
    cached_values = torch.full((2, 12, 2), 17.0)
    device_targets = cached_values.clone()
    teacher_state = None
    for start in (0, 4, 8):
        rows = slice(start, start + 4)
        teacher_state = training.replace_final_initializer_chunk(
            teacher, inputs, rows, teacher_state, device_targets
        )
        assert not teacher_state.requires_grad
    torch.testing.assert_close(device_targets, expected[..., :2], rtol=1e-6, atol=1e-7)
    assert torch.equal(cached_values, torch.full_like(cached_values, 17.0))

    teacher_before = training.final_initializer_state_sha256(teacher)
    with torch.no_grad():
        prediction_before, _ = training.forward_chunk(teacher, inputs, slice(0, 12), None)
    configure_qat(student, "recurrent_w4", learn_scales=True)
    optimizer = torch.optim.SGD(student.parameters(), lr=0.01)
    optimizer.zero_grad(set_to_none=True)
    output, _ = training.forward_chunk(student, inputs, slice(0, 4), None)
    output.square().mean().backward()
    optimizer.step()
    invalidate_qat_cache(student)
    assert any(parameter.grad is not None for parameter in student.parameters())

    with torch.no_grad():
        prediction_after, _ = training.forward_chunk(teacher, inputs, slice(0, 12), None)
    assert training.final_initializer_state_sha256(teacher) == teacher_before
    torch.testing.assert_close(prediction_after, prediction_before, rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in teacher.parameters())


def test_initializer_distillation_target_clips_both_sources() -> None:
    labels = torch.tensor([[[-3.0, -1.0], [0.5, 3.0]]])
    predictions = torch.tensor([[[3.0, -2.0], [1.5, -4.0]]])
    target = training.final_initializer_distillation_target(labels, predictions)
    expected = torch.tensor([[[1.6, -1.9], [1.4, -1.6]]])
    torch.testing.assert_close(target, expected, rtol=0, atol=2e-7)


def test_initializer_distillation_guard_accepts_only_supported_recipe() -> None:
    assert training.final_initializer_distillation_error(_valid_args()) is None
    assert training.final_initializer_distillation_error(
        _valid_args(final_distill_initializer=False, final_model_adaptation=False)
    ) is None


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"final_model_adaptation": False}, "final-model adaptation"),
        ({"distill_cache_root": ""}, "validated --distill-cache-root"),
        ({"distill_alpha": 0.8}, "fixes alpha=0.9"),
        ({"distill_family_weight": 0.1}, "family targets"),
        ({"update_rows": 4000}, "default chunk updates"),
        ({"origin_view": "prefix-fixed"}, "default chunk updates"),
        ({"ema_teacher": 0.1}, "EMA, SWA, or TD"),
        ({"swa_frac": 0.5}, "EMA, SWA, or TD"),
        ({"td_alpha": 0.1}, "EMA, SWA, or TD"),
        ({"resume_state": "/tmp/state.pt"}, "training-state continuation"),
    ],
)
def test_initializer_distillation_guard_rejects_unsupported_arms(
    changes: dict[str, object], message: str
) -> None:
    error = training.final_initializer_distillation_error(_valid_args(**changes))
    assert error is not None and message in error
