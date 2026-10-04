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

from wnn.training_controls import DistilledLoss, distilled_macro_losses
from wnn.training_sampling import EpochOrderSampler
from wnn.training_state import (
    atomic_save_training_state,
    capture_training_state,
    load_training_state,
    restore_training_state,
)


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


def test_uniform_event_counts_collapse_all_three_teacher_clocks() -> None:
    base_decay = 0.93
    schedules = [
        make_event_ema_schedule(
            (4, 4, 4, 4),
            base_decay=base_decay,
            updates_per_epoch=4,
            mode=mode,
            seed=901,
            epoch=1,
        )
        for mode in ("constant", "aligned", "shuffled")
    ]
    for schedule in schedules:
        assert schedule.decays == (base_decay,) * 4
        assert schedule.applied_counts == (4, 4, 4, 4)

    final_weights = []
    for schedule in schedules:
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.zero_()
        teacher = TinyFastEMA(model, decay=base_decay)
        for update, decay in enumerate(schedule.decays, start=1):
            with torch.no_grad():
                model.weight.fill_(float(update))
            update_ema_with_decay(teacher, model, decay)
        final_weights.append(teacher.module.weight.detach().clone())
    for value in final_weights[1:]:
        torch.testing.assert_close(value, final_weights[0], rtol=0, atol=0)


def test_zero_event_decay_preserves_first_call_copy_then_freezes() -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    teacher = TinyFastEMA(model, decay=0.8)

    # Match FastEMA/AveragedModel semantics: the first update initializes the
    # average from the student even when its event-derived decay is one.
    with torch.no_grad():
        model.weight.fill_(1.0)
    update_ema_with_decay(teacher, model, 1.0)
    torch.testing.assert_close(teacher.module.weight, model.weight, rtol=0, atol=0)
    assert teacher.n == 1

    with torch.no_grad():
        model.weight.fill_(2.0)
    update_ema_with_decay(teacher, model, 1.0)
    torch.testing.assert_close(teacher.module.weight, torch.ones_like(model.weight), rtol=0, atol=0)
    assert teacher.n == 2


def test_zero_event_window_freezes_teacher_but_advances_student_and_eval_ema() -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    teacher = TinyFastEMA(model, decay=0.8)
    evaluation = TinyFastEMA(model, decay=0.8)

    with torch.no_grad():
        model.weight.fill_(1.0)
    update_ema_with_decay(teacher, model, 0.8)
    update_ema_with_decay(evaluation, model, 0.8)
    teacher_before = teacher.module.weight.detach().clone()
    evaluation_before = evaluation.module.weight.detach().clone()

    with torch.no_grad():
        model.weight.fill_(2.0)
    update_ema_with_decay(teacher, model, 1.0)
    update_ema_with_decay(evaluation, model, 0.8)

    torch.testing.assert_close(teacher.module.weight, teacher_before, rtol=0, atol=0)
    assert not torch.equal(model.weight, teacher.module.weight)
    assert not torch.equal(evaluation.module.weight, evaluation_before)


def test_three_teacher_clocks_keep_the_row_clock_evaluation_ema_fixed() -> None:
    observed = (0, 1, 4, 3)
    base_decay = 0.9
    evaluation_decay = 0.97
    final_teachers = []
    final_evaluations = []
    for mode in ("constant", "aligned", "shuffled"):
        schedule = make_event_ema_schedule(
            observed,
            base_decay=base_decay,
            updates_per_epoch=len(observed),
            mode=mode,
            seed=31,
            epoch=0,
        )
        model = torch.nn.Linear(1, 1, bias=False)
        model.weight.data.zero_()
        teacher = TinyFastEMA(model, decay=base_decay)
        evaluation = TinyFastEMA(model, decay=evaluation_decay)
        for update, decay in enumerate(schedule.decays, start=1):
            with torch.no_grad():
                model.weight.fill_(float(update))
            update_ema_with_decay(teacher, model, decay)
            update_ema_with_decay(evaluation, model, evaluation_decay)
        final_teachers.append(teacher.module.weight.item())
        final_evaluations.append(evaluation.module.weight.item())

    assert final_teachers[0] != final_teachers[1]
    assert final_evaluations[0] == final_evaluations[1] == final_evaluations[2]


def test_epoch_sampler_maps_asymmetric_train_valid_counts_and_drops_tail() -> None:
    # The count matrix follows exactly this concatenation order, even when
    # TRAIN and VALID both contain an absolute row-group ID of zero.
    source_rows = (("train", 0), ("train", 1), ("train", 2), ("valid", 0), ("valid", 1))
    sequence_counts = np.asarray([[0, 1], [1, 0], [10, 0], [0, 100], [7, 8]], dtype=np.int64)
    sampler = EpochOrderSampler(5, seed=83, batch_size=2, drop_last=True)
    sampler.set_epoch(3)
    order = list(iter(sampler))
    expected = sequence_counts[order[:4]].reshape(2, 2, 2).sum(axis=1).reshape(-1)

    observed = per_update_event_counts(sequence_counts, order, batch_size=2, drop_last=True)

    assert len(order) == 5
    assert len(observed) == 4
    np.testing.assert_array_equal(observed, expected)
    assert sum(observed) == int(sequence_counts[order[:4]].sum())
    assert sum(observed) != int(sequence_counts.sum())
    assert len(source_rows) == len(sequence_counts)


def test_teacher_and_evaluation_ema_replay_resume_independently(tmp_path) -> None:
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
    checkpoint_path = tmp_path / "epoch-state.pt"
    atomic_save_training_state(saved, checkpoint_path)
    saved = load_training_state(checkpoint_path)

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


class TinyCausalTeacher(torch.nn.Module):
    """Small recurrent model for testing teacher carry and causal isolation."""

    def __init__(self) -> None:
        super().__init__()
        self.gru = torch.nn.GRU(3, 5, batch_first=True)
        self.head = torch.nn.Linear(5, 2)

    def forward(
        self, inputs: torch.Tensor, state: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden, next_state = self.gru(inputs, state)
        return self.head(hidden), next_state


def _chunked_forward(
    model: TinyCausalTeacher, inputs: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    state = None
    outputs = []
    for start, stop in ((0, 2), (2, 7), (7, 11), (11, 15)):
        output, state = model(inputs[:, start:stop], state)
        outputs.append(output)
    return torch.cat(outputs, dim=1), state


def test_teacher_state_is_causal_chunked_and_reset_between_sequences() -> None:
    torch.manual_seed(4260)
    teacher = TinyCausalTeacher().eval()
    evaluation = TinyCausalTeacher().eval()
    evaluation.load_state_dict(teacher.state_dict())
    with torch.no_grad():
        evaluation.gru.weight_hh_l0.add_(0.5)
    inputs = torch.randn(1, 15, 3)
    changed_future = inputs.clone()
    changed_future[:, 7:] = torch.randn_like(changed_future[:, 7:])

    with torch.no_grad():
        whole, _ = teacher(inputs, None)
        chunked, teacher_state = _chunked_forward(teacher, inputs)
        suffix_changed, _ = teacher(changed_future, None)
        _, evaluation_state = _chunked_forward(evaluation, inputs)
        reset, _ = teacher(inputs, None)
        teacher(inputs[:, :4], None)
        after_other_sequence, _ = teacher(inputs, None)

    torch.testing.assert_close(chunked, whole, rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(suffix_changed[:, :7], whole[:, :7], rtol=0, atol=0)
    torch.testing.assert_close(reset, whole, rtol=0, atol=0)
    torch.testing.assert_close(after_other_sequence, whole, rtol=0, atol=0)
    assert teacher_state.data_ptr() != evaluation_state.data_ptr()
    assert not torch.equal(teacher_state, evaluation_state)


def test_consistency_teacher_targets_are_detached_from_student_backward() -> None:
    torch.manual_seed(4261)
    student = torch.nn.Linear(3, 23)
    teacher = torch.nn.Linear(3, 23)
    inputs = torch.randn(1, 4, 3)
    labels = torch.zeros(1, 4, 2)
    cached_teacher = torch.zeros(1, 4, 2)
    row_weights = torch.ones(1, 4)

    def student_forward(window: slice, state: torch.Tensor | None):
        return student(inputs[:, window]), state

    def teacher_forward(window: slice, state: torch.Tensor | None):
        return teacher(inputs[:, window]), state

    losses = distilled_macro_losses(
        forward=student_forward,
        ema_forward=teacher_forward,
        state=None,
        ema_state=None,
        labels=labels,
        teacher=cached_teacher,
        row_weights=row_weights,
        next_change=None,
        structural_labels=None,
        next_labels=None,
        rows=slice(0, 4),
        micro_rows=2,
        detach_rows=2,
        config=DistilledLoss(0.0, 0, 1.0, False, 0.0, 2, 12, 0.0, 21),
    )
    for contribution, _, _ in losses:
        contribution.backward()

    assert student.weight.grad is not None
    assert torch.count_nonzero(student.weight.grad) > 0
    assert all(parameter.grad is None for parameter in teacher.parameters())
