from __future__ import annotations

import inspect
import multiprocessing
import pickle
from multiprocessing.connection import Connection
from pathlib import Path

import pytest
import torch
from torch.utils.data import Dataset

from scripts import train_rnn
from wnn.data import N_FEATURES
from wnn.training.compatibility import LegacyTrainerAPI
from wnn.training.datasets import Sequences as CanonicalSequences
from wnn.training.environment import TrainingEnvironment


def _spawned_dataset_probe(dataset: CanonicalSequences, sender: Connection) -> None:
    """Read the canonical dataset state from an independently spawned worker."""
    marker = getattr(dataset, "compatibility_marker", lambda: None)()
    sender.send(
        (
            type(dataset) is CanonicalSequences,
            type(dataset).__module__,
            type(dataset).__qualname__,
            marker,
            dataset.path.as_posix(),
        )
    )
    sender.close()


class ImportableSequenceFacade(train_rnn.Sequences):
    """Verify subclass methods survive facade pickling."""

    def compatibility_marker(self) -> str:
        """Return an override that a base-class fallback would lose."""
        return "subclass-method-preserved"


def test_legacy_sequences_export_remains_a_class_and_injects_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Preserve the old public class protocol and caller overrides without reading data."""
    monkeypatch.setattr(train_rnn, "ROOT", tmp_path)
    dataset_class = train_rnn.Sequences

    assert isinstance(dataset_class, type)
    assert issubclass(dataset_class, Dataset)
    assert issubclass(dataset_class, CanonicalSequences)
    assert dataset_class is train_rnn.Sequences
    signature = inspect.signature(dataset_class)
    assert tuple(signature.parameters)[:2] == ("split", "indices")
    assert signature.parameters["environment"].kind is inspect.Parameter.KEYWORD_ONLY

    dataset = dataset_class("valid", [])
    assert isinstance(dataset, dataset_class)
    assert dataset.path == tmp_path / "datasets/valid.parquet"
    restored = pickle.loads(pickle.dumps(dataset))
    assert type(restored) is dataset_class
    assert restored.environment.root == tmp_path

    explicit_environment = TrainingEnvironment(root=tmp_path / "explicit")
    explicit_dataset = dataset_class("valid", [], environment=explicit_environment)
    assert explicit_dataset.environment is explicit_environment


def test_explicit_environment_skips_incomplete_legacy_namespace() -> None:
    """Class and function adapters do not inspect legacy globals when overridden."""
    api = LegacyTrainerAPI({"__name__": "incomplete_legacy_trainer"})
    environment = TrainingEnvironment()
    dataset = api.attribute("Sequences")("valid", [], environment=environment)
    assert dataset.environment is environment

    build_inputs = api.attribute("build_inputs")
    values = torch.zeros(1, 2, N_FEATURES)
    inputs = build_inputs(values, environment=environment)
    assert inputs.shape[:2] == values.shape[:2]


def test_custom_name_sequence_facade_pickles_through_spawn(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Spawn workers receive canonical sequence instances with captured state."""
    monkeypatch.setattr(train_rnn, "ROOT", tmp_path)
    custom_name = "unregistered_legacy_trainer"
    api = LegacyTrainerAPI({**vars(train_rnn), "__name__": custom_name})
    legacy_class = api.attribute("Sequences")
    dataset = legacy_class("valid", [])

    assert issubclass(legacy_class, CanonicalSequences)
    assert isinstance(dataset, CanonicalSequences)
    assert legacy_class.__module__ == custom_name

    restored = pickle.loads(pickle.dumps(dataset))
    assert type(restored) is CanonicalSequences
    assert restored.environment.root == tmp_path
    assert restored.path == tmp_path / "datasets/valid.parquet"

    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_spawned_dataset_probe, args=(dataset, sender))
    process.start()
    sender.close()
    try:
        assert receiver.poll(30), "spawn worker did not return dataset state"
        worker_result = receiver.recv()
    finally:
        receiver.close()
        process.join(30)
        if process.is_alive():
            process.terminate()
            process.join()

    assert process.exitcode == 0
    assert worker_result == (
        True,
        CanonicalSequences.__module__,
        CanonicalSequences.__qualname__,
        None,
        str(tmp_path / "datasets/valid.parquet"),
    )


def test_importable_sequence_subclass_keeps_override_through_pickle_and_spawn(
    tmp_path: Path,
) -> None:
    """Retain a caller subclass and its method through ordinary and spawn pickling."""
    dataset = ImportableSequenceFacade("valid", [], environment=TrainingEnvironment(root=tmp_path))
    restored = pickle.loads(pickle.dumps(dataset))

    assert type(restored) is ImportableSequenceFacade
    assert restored.compatibility_marker() == "subclass-method-preserved"

    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_spawned_dataset_probe, args=(dataset, sender))
    process.start()
    sender.close()
    try:
        assert receiver.poll(30), "spawn worker did not return subclass state"
        worker_result = receiver.recv()
    finally:
        receiver.close()
        process.join(30)
        if process.is_alive():
            process.terminate()
            process.join()

    assert process.exitcode == 0
    assert worker_result == (
        False,
        ImportableSequenceFacade.__module__,
        ImportableSequenceFacade.__qualname__,
        "subclass-method-preserved",
        str(tmp_path / "datasets/valid.parquet"),
    )
