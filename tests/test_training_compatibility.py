from __future__ import annotations

import inspect
from pathlib import Path

import pytest
from torch.utils.data import Dataset

from scripts import train_rnn
from wnn.training.environment import TrainingEnvironment


def test_legacy_sequences_export_remains_a_class_and_injects_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Preserve the old public class protocol and caller overrides without reading data."""
    monkeypatch.setattr(train_rnn, "ROOT", tmp_path)
    dataset_class = train_rnn.Sequences

    assert isinstance(dataset_class, type)
    assert issubclass(dataset_class, Dataset)
    assert dataset_class is train_rnn.Sequences
    signature = inspect.signature(dataset_class)
    assert tuple(signature.parameters)[:2] == ("split", "indices")
    assert signature.parameters["environment"].kind is inspect.Parameter.KEYWORD_ONLY

    dataset = dataset_class("valid", [])
    assert isinstance(dataset, dataset_class)
    assert dataset.path == tmp_path / "datasets/valid.parquet"

    explicit_environment = TrainingEnvironment(root=tmp_path / "explicit")
    explicit_dataset = dataset_class("valid", [], environment=explicit_environment)
    assert explicit_dataset.environment is explicit_environment
