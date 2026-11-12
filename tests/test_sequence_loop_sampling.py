from __future__ import annotations

import torch
from torch import nn

from wnn.training import sequence_loop
from wnn.training.sequence_distributed import DistributedContext
from wnn.training.sequence_weight_ema import SequenceWeightEMA


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.0]))


class _Dataset:
    def __len__(self) -> int:
        return 100


class _Log:
    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []

    def write(self, row: dict[str, object]) -> None:
        self.rows.append(row)


def test_train_epoch_samples_actual_updates_one_fifty_and_one_hundred(monkeypatch) -> None:
    class _Event:
        def __init__(self, **_: object) -> None:
            pass

        def record(self) -> None:
            pass

        def elapsed_time(self, _: object) -> float:
            return 1.0

    monkeypatch.setattr(torch.cuda, "Event", _Event)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _device: 0)
    monkeypatch.setattr(sequence_loop, "SEQUENCE_ROWS", 1)
    monkeypatch.setattr(sequence_loop, "local_sequence_indices", lambda size, *_: list(range(size)))
    monkeypatch.setattr(sequence_loop, "collate_sequences", lambda *_: object())
    monkeypatch.setattr(
        sequence_loop,
        "move_and_validate_batch",
        lambda *_: {
            "y": torch.ones((1, 1, 2)),
            "teacher": torch.zeros((1, 1, 2)),
            "mask": torch.ones((1, 1), dtype=torch.bool),
            "hints": torch.empty((1, 1, 0)),
            "x": torch.zeros((1, 1, 1)),
        },
    )
    monkeypatch.setattr(
        sequence_loop,
        "_window_denominators",
        lambda *_: torch.tensor([1.0, 0.0]),
    )

    def backward_window(model, *_args, **_kwargs):
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        return None, torch.tensor([1.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=torch.float64)

    monkeypatch.setattr(sequence_loop, "_backward_window", backward_window)
    model = _TinyModel()
    weight_ema = SequenceWeightEMA(model, decay=0.9)
    post_step_weights: list[float] = []

    def update_ema() -> None:
        post_step_weights.append(float(model.weight.detach().item()))
        weight_ema.update(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)
    log = _Log()

    result = sequence_loop.train_epoch(
        _Dataset(),
        model,
        optimizer,
        scheduler,
        {
            "seed": 4,
            "batch_size": 1,
            "chunk_rows": 1,
            "accumulate_chunks": 1,
            "model": {"hint_dim": 0, "attention_backend": "math"},
            "grad_clip": 1.0,
            "kd_weight": 0.0,
            "hint_weight": 0.0,
            "optimizer_update_diagnostics_every": 50,
        },
        context,
        log,
        epoch=0,
        global_step=0,
        after_optimizer_update=update_ema,
    )

    assert result.complete and result.updates == 100
    assert [row["step"] for row in log.rows] == list(range(1, 101))
    sampled = [
        row["optimizer_update_diagnostics"]["sampled_update_step"]
        for row in log.rows
        if "optimizer_update_diagnostics" in row
    ]
    assert sampled == [1, 50, 100]
    assert scheduler.last_epoch == 100
    assert len(post_step_weights) == weight_ema.updates == result.updates
    assert post_step_weights[0] < 1.0
