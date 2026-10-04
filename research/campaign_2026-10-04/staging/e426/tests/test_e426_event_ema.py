from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from wnn.event_ema import (
    ENDPOINT,
    endpoint_events_by_update,
    inference_ema_module,
    make_event_ema_schedule,
    per_update_event_counts,
    update_ema_with_decay,
)
from wnn.training_state import capture_training_state, restore_training_state


class TinyFastEMA:
    """FastEMA-compatible test object with an independently checkpointed clock."""

    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        import copy

        self.module = copy.deepcopy(model).eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)
        self.decay = decay
        self.n = 0

    @torch.no_grad()
    def update_parameters(self, model: torch.nn.Module) -> None:
        with torch.no_grad():
            for average, current in zip(self.module.parameters(), model.parameters(), strict=True):
                if self.n == 0:
                    average.copy_(current)
                else:
                    average.lerp_(current, 1.0 - self.decay)
            self.n += 1


class EmptySampler:
    pass


def _loader() -> SimpleNamespace:
    return SimpleNamespace(
        persistent_workers=False,
        sampler=EmptySampler(),
        batch_size=2,
        drop_last=True,
        num_workers=0,
        dataset=[0, 1],
    )


def _advance(
    model: torch.nn.Module,
    teacher: TinyFastEMA,
    evaluation: TinyFastEMA,
    *,
    step: int,
    teacher_decay: float,
    evaluation_decay: float,
) -> None:
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(float(step))
    update_ema_with_decay(teacher, model, teacher_decay)
    update_ema_with_decay(evaluation, model, evaluation_decay)


def test_sequence_event_counts_are_train_input_only_and_update_aligned(tmp_path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    endpoint = float(ENDPOINT)
    path = tmp_path / "fit.parquet"
    table = pa.table(
        {
            "a6": [0.0, endpoint, 0.0, endpoint],
            "t0": [999.0, 999.0, 999.0, 999.0],
        }
    )
    pq.write_table(table, path, row_group_size=4)

    counts = endpoint_events_by_update(path, [0], update_rows=2, sequence_length=4)
    assert counts.tolist() == [[1, 1]]


def test_epoch_batch_schedule_and_event_retention_normalization() -> None:
    sequence_counts = np.asarray([[1, 0], [2, 1], [0, 3], [1, 2]], dtype=np.int64)
    observed = per_update_event_counts(sequence_counts, [2, 0, 3, 1], batch_size=2)
    assert observed == (1, 3, 3, 3)

    base_decay = 0.99
    aligned = make_event_ema_schedule(
        observed,
        base_decay=base_decay,
        updates_per_epoch=4,
        mode="aligned",
        seed=86_173,
        epoch=2,
    )
    assert aligned.applied_counts == observed
    assert aligned.total_events == sum(observed)
    assert np.isclose(np.prod(aligned.decays), base_decay**4, rtol=1e-12, atol=0.0)

    shuffled = make_event_ema_schedule(
        observed,
        base_decay=base_decay,
        updates_per_epoch=4,
        mode="shuffled",
        seed=86_173,
        epoch=2,
    )
    repeated = make_event_ema_schedule(
        observed,
        base_decay=base_decay,
        updates_per_epoch=4,
        mode="shuffled",
        seed=86_173,
        epoch=2,
    )
    assert sorted(shuffled.applied_counts) == sorted(observed)
    assert shuffled.applied_counts == repeated.applied_counts
    assert shuffled.decays == repeated.decays
    assert np.isclose(np.prod(shuffled.decays), base_decay**4, rtol=1e-12, atol=0.0)


def test_teacher_and_evaluation_ema_replay_resume_independently() -> None:
    torch.manual_seed(426)
    uninterrupted_model = torch.nn.Linear(3, 2)
    uninterrupted_teacher = TinyFastEMA(uninterrupted_model, decay=0.91)
    uninterrupted_evaluation = TinyFastEMA(uninterrupted_model, decay=0.99)
    _advance(
        uninterrupted_model,
        uninterrupted_teacher,
        uninterrupted_evaluation,
        step=1,
        teacher_decay=0.8,
        evaluation_decay=0.99,
    )

    optimizer = torch.optim.SGD(uninterrupted_model.parameters(), lr=0.1, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    saved = capture_training_state(
        model=uninterrupted_model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        loader=_loader(),
        contract={"epochs": 3, "teacher_clock": "aligned"},
        next_epoch=1,
        progress={"step_count": 1},
        teacher_ema=uninterrupted_teacher,
        evaluation_ema=uninterrupted_evaluation,
    )

    _advance(
        uninterrupted_model,
        uninterrupted_teacher,
        uninterrupted_evaluation,
        step=2,
        teacher_decay=0.7,
        evaluation_decay=0.99,
    )

    resumed_model = torch.nn.Linear(3, 2)
    resumed_teacher = TinyFastEMA(resumed_model, decay=0.91)
    resumed_evaluation = TinyFastEMA(resumed_model, decay=0.99)
    resumed_optimizer = torch.optim.SGD(resumed_model.parameters(), lr=0.1, momentum=0.9)
    resumed_scheduler = torch.optim.lr_scheduler.LambdaLR(resumed_optimizer, lambda _step: 1.0)
    resumed_scaler = torch.amp.GradScaler("cpu", enabled=False)
    restore_training_state(
        saved,
        model=resumed_model,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
        scaler=resumed_scaler,
        loader=_loader(),
        teacher_ema=resumed_teacher,
        evaluation_ema=resumed_evaluation,
    )
    _advance(
        resumed_model,
        resumed_teacher,
        resumed_evaluation,
        step=2,
        teacher_decay=0.7,
        evaluation_decay=0.99,
    )

    for expected, actual in zip(
        uninterrupted_model.state_dict().values(), resumed_model.state_dict().values(), strict=True
    ):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for expected, actual in zip(
        uninterrupted_teacher.module.state_dict().values(),
        resumed_teacher.module.state_dict().values(),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for expected, actual in zip(
        uninterrupted_evaluation.module.state_dict().values(),
        resumed_evaluation.module.state_dict().values(),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert resumed_teacher.n == uninterrupted_teacher.n == 2
    assert resumed_evaluation.n == uninterrupted_evaluation.n == 2


def test_inference_selects_fixed_evaluation_ema_not_consistency_teacher() -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    teacher = TinyFastEMA(model, decay=0.7)
    evaluation = TinyFastEMA(model, decay=0.99)
    with torch.no_grad():
        teacher.module.weight.fill_(3.0)
        evaluation.module.weight.fill_(7.0)

    chosen = inference_ema_module(evaluation)
    assert chosen is evaluation.module
    assert chosen is not teacher.module
    assert chosen.weight.item() == 7.0
