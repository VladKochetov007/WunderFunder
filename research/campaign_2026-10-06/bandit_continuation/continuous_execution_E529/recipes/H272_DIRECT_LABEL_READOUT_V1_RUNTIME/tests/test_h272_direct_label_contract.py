from __future__ import annotations

import torch
from train_conditional_readout import _build_batch


def test_batch_builder_rejects_appended_teacher_columns() -> None:
    raw = torch.zeros((1, 3, 112), dtype=torch.float32)
    labels_and_stale_teacher = torch.zeros((1, 3, 4), dtype=torch.float32)
    scored = torch.ones((1, 3), dtype=torch.bool)
    propensity = torch.ones((1, 3), dtype=torch.float32)
    batch = (raw, labels_and_stale_teacher, scored, propensity)

    try:
        _build_batch(batch, torch.device("cpu"), None, None)
    except ValueError as error:
        assert "exactly two original label channels" in str(error)
    else:
        raise AssertionError("H272 must reject the former teacher-augmented target layout")
