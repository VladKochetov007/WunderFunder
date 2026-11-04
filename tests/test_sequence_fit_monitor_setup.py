from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from wnn.training.sequence_distributed import DistributedContext
from wnn.training.sequence_fit_monitor_setup import initialize_monitor
from wnn.training.sequence_monitor_identity import mask_comparison


def test_train_eval_mask_must_match_actual_fitting_mask(tmp_path: Path) -> None:
    data = SimpleNamespace(fit=SimpleNamespace(training_mask="is_scored"))
    with pytest.raises(ValueError, match="match the actual fitting objective mask"):
        initialize_monitor(
            tmp_path,
            tmp_path,
            {"train_eval_mask": "all_after_warmup"},
            [],
            data,
            {},
            False,
            DistributedContext(0, 0, 1, torch.device("cpu"), False),
        )


def test_mask_comparison_is_accurate_for_legacy_and_extended_monitors() -> None:
    assert mask_comparison("is_scored") == (
        "train_eval and valid both use the official is_scored mask"
    )
    assert "masks differ" in mask_comparison("all_after_warmup")
