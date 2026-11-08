from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import numpy as np
import pytest

from wnn.training.sequence_reference_cache import ReferenceCache, validate_reference_identity
from wnn.training.sequence_reference_contract import canonical_sha256, members_sha256
from wnn.training.sequence_reference_pins import (
    ANCESTRY_LIMITATION,
    ARITHMETIC,
    COMPOSITION,
    PR49_SOURCE_COMMIT,
    PRIMARY_SPLIT_SHA256,
    PUBLISHED_RUNS,
)


def reference_fixture() -> tuple[dict, list[tuple[str, int]]]:
    """Construct structural identity data; no fixture claims to be real predictions."""
    fitting = [("train", g) for g in range(10325)]
    fitting_hash = members_sha256(fitting)
    evaluation_hash = members_sha256([("valid", g) for g in range(1873)])
    producer = {
        "kind": "pr49-tl3-reference-v1",
        "split_sha256": PRIMARY_SPLIT_SHA256,
        "fit_members_sha256": fitting_hash,
        "evaluation_members_sha256": evaluation_hash,
        "composition": copy.deepcopy(COMPOSITION),
        "arithmetic": ARITHMETIC,
        "inference_source_commit": PR49_SOURCE_COMMIT,
        "training_predictions": "in-sample",
        "ancestry_limitation": ANCESTRY_LIMITATION,
        "ancestors": [
            {
                "run": run,
                **pin,
                "fit_members_sha256": fitting_hash,
                "evaluation_members_sha256": evaluation_hash,
                "trainer_source_sha256": "e" * 64,
            }
            for run, pin in sorted(PUBLISHED_RUNS.items())
        ],
    }
    identity = {
        "manifest_sha256": "a" * 64,
        "reference_sha256": canonical_sha256(producer),
        "producer": producer,
    }
    return identity, fitting


def test_reference_hash_is_path_independent_and_manifest_specific() -> None:
    """Different cache manifests retain one reference but not one payload identity."""
    identity, fitting = reference_fixture()
    validate_reference_identity(identity, PRIMARY_SPLIT_SHA256, fitting)
    evaluation = copy.deepcopy(identity)
    evaluation["manifest_sha256"] = "b" * 64
    validate_reference_identity(evaluation, PRIMARY_SPLIT_SHA256, fitting)
    assert evaluation["reference_sha256"] == identity["reference_sha256"]
    assert evaluation["manifest_sha256"] != identity["manifest_sha256"]


@pytest.mark.parametrize(
    "mutation", ["checkpoint", "missing_parent", "split", "mixture", "precision", "fit"]
)
def test_reference_rejects_rehashed_foreign_ancestry(mutation: str) -> None:
    """Rehashing caller metadata cannot admit another teacher, cohort or composition."""
    identity, fitting = reference_fixture()
    producer = identity["producer"]
    if mutation == "checkpoint":
        producer["ancestors"][0]["checkpoint_sha256"] = "f" * 64
    elif mutation == "missing_parent":
        producer["ancestors"].pop()
    elif mutation == "split":
        producer["split_sha256"] = "f" * 64
    elif mutation == "mixture":
        producer["composition"][0]["weight"] = 0.33
    elif mutation == "precision":
        producer["arithmetic"] = "clip-members-before-mixing"
    else:
        fitting[0] = ("valid", 0)
    identity["reference_sha256"] = canonical_sha256(producer)
    with pytest.raises(ValueError):
        validate_reference_identity(identity, PRIMARY_SPLIT_SHA256, fitting)


@pytest.mark.parametrize("name", ["../outside.npy", "/outside.npy", "nested/value.npy"])
def test_reference_read_rejects_path_escape(tmp_path: Path, name: str) -> None:
    """Exercise file resolution without constructing a multi-gigabyte reference cohort."""
    cache = object.__new__(ReferenceCache)
    cache.root = tmp_path
    cache.entries = {("train", 0): {"file": name}}
    with pytest.raises(ValueError, match="escapes"):
        cache.read("train", 0, 0)


@pytest.mark.parametrize("fault", ["float64", "shape", "nonfinite", "hints"])
def test_reference_read_rejects_bad_payload(tmp_path: Path, fault: str) -> None:
    """A valid filename cannot bypass dtype, shape, finite or no-hints checks."""
    values = np.zeros((20000, 2), dtype=np.float32)
    if fault == "float64":
        values = values.astype(np.float64)
    elif fault == "shape":
        values = values[:19999]
    elif fault == "nonfinite":
        values[0, 0] = np.nan
    np.save(tmp_path / "array.npy", values, allow_pickle=False)
    cache = object.__new__(ReferenceCache)
    cache.root = tmp_path
    cache.entries = {
        ("train", 0): {
            "file": "array.npy",
            "sha256": hashlib.sha256((tmp_path / "array.npy").read_bytes()).hexdigest(),
        }
    }
    with pytest.raises(ValueError):
        cache.read("train", 0, 1 if fault == "hints" else 0)


def test_reference_read_checks_consumed_bytes_after_admission(tmp_path: Path) -> None:
    """An in-place payload edit cannot silently change later training targets."""
    values = np.zeros((20000, 2), dtype=np.float32)
    path = tmp_path / "array.npy"
    np.save(path, values, allow_pickle=False)
    cache = object.__new__(ReferenceCache)
    cache.root = tmp_path
    cache.entries = {
        ("train", 0): {
            "file": "array.npy",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    }
    prediction, hints = cache.read("train", 0, 0)
    assert prediction.shape == (20000, 2) and hints.shape == (20000, 0)
    values[1, 0] = 1.0
    np.save(path, values, allow_pickle=False)
    with pytest.raises(ValueError, match="changed after cache admission"):
        cache.read("train", 0, 0)
