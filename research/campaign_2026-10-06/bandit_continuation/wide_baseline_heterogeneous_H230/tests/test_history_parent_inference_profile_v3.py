"""Regression checks for raw SequenceFile width versus engineered model width."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[5]
SOURCE_ROOT = (
    PROJECT_ROOT
    / "research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529"
    / "h230_source_v4_history_swap"
)
sys.path.insert(0, str(SOURCE_ROOT / "scripts"))
sys.path.insert(0, str(SOURCE_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import export_all_data_parent as exporter  # noqa: E402
from profile_history_parent_inference_v3 import validate_raw_feature_batch  # noqa: E402


def test_feature_only_groups_yield_sealed_raw_112_fixture(monkeypatch, tmp_path):
    calls = []

    class MockSequenceFile:
        def __init__(self, path):
            calls.append(("open", Path(path)))

        def read(self, group, *, targets, mask):
            calls.append(("read", group, targets, mask))
            return SimpleNamespace(
                features=np.zeros(
                    (exporter.train_rnn.SEQUENCE_LENGTH, exporter.train_rnn.N_FEATURES),
                    dtype=np.float32,
                )
            )

    monkeypatch.setattr(exporter, "SequenceFile", MockSequenceFile)
    dataset = exporter.FeatureOnlyGroups(tmp_path / "fixture.parquet", [77])
    raw = dataset[0]

    assert raw.dtype == torch.float32
    assert tuple(raw.shape) == (
        exporter.train_rnn.SEQUENCE_LENGTH,
        exporter.train_rnn.N_FEATURES,
    )
    validate_raw_feature_batch(
        raw.unsqueeze(0),
        batch_size=1,
        sequence_length=exporter.train_rnn.SEQUENCE_LENGTH,
        raw_feature_width=exporter.train_rnn.N_FEATURES,
    )
    assert calls[-1] == ("read", 77, False, False)


def test_sealed_preprocessor_maps_raw_112_to_checkpoint_width_211():
    raw_width = exporter.train_rnn.N_FEATURES
    model_input_width = exporter.train_rnn.build_inputs(
        torch.zeros((1, 3, raw_width), dtype=torch.float32)
    ).shape[-1]

    assert raw_width == 112
    assert exporter.train_rnn.N_EVENT_FEATURES == 99
    assert model_input_width == raw_width + exporter.train_rnn.N_EVENT_FEATURES == 211


def test_211_is_rejected_as_raw_sequencefile_width():
    bad_raw = torch.zeros((1, 20000, 211), dtype=torch.float32)

    with pytest.raises(ValueError, match=r"expected=float32 \(1, 20000, 112\)"):
        validate_raw_feature_batch(
            bad_raw,
            batch_size=1,
            sequence_length=20000,
            raw_feature_width=exporter.train_rnn.N_FEATURES,
        )
