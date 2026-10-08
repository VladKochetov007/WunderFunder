from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import train_parent_readout as runner


def test_production_batch_builder_rejects_appended_teacher_columns() -> None:
    raw = torch.zeros((1, 3, 112), dtype=torch.float32)
    labels_and_stale_teacher = torch.zeros((1, 3, 4), dtype=torch.float32)
    scored = torch.ones((1, 3), dtype=torch.bool)
    propensity = torch.ones((1, 3), dtype=torch.float32)
    batch = (raw, labels_and_stale_teacher, scored, propensity)

    with pytest.raises(ValueError, match="only the original two label channels"):
        runner._batch_inputs(batch, torch.device("cpu"), None, SimpleNamespace())
