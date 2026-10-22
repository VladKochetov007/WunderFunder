from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from wnn.data import N_FEATURES
from wnn.splits import CampaignSplit
from wnn.training import sequence_numeric_preprocessing as preprocessing
from wnn.training.sequence_numeric import (
    NUMERIC_INPUT_WIDTH,
    NUMERIC_MISSING_OFFSET,
    NumericBasis,
    StreamingNumericInputs,
    numeric_inputs,
)


def test_numeric_inputs_impute_and_match_streaming_rows() -> None:
    raw = np.arange(5 * N_FEATURES, dtype=np.float32).reshape(5, N_FEATURES)
    raw[1, 4] = np.nan
    raw[3, 8] = np.inf
    batched = numeric_inputs(raw)
    stream = StreamingNumericInputs()
    streamed = np.stack([stream.update(row) for row in raw])

    assert batched.shape == (5, NUMERIC_INPUT_WIDTH)
    assert batched.dtype == np.float32
    np.testing.assert_allclose(batched, streamed, rtol=0.0, atol=1e-7)
    assert batched[1, 4] == 0.0
    assert batched[3, 8] == 0.0
    assert batched[1, NUMERIC_MISSING_OFFSET + 4] == 1.0
    assert batched[3, NUMERIC_MISSING_OFFSET + 8] == 1.0
    np.testing.assert_array_equal(batched[:3], numeric_inputs(raw[:3]))
    stream.reset()
    np.testing.assert_allclose(
        stream.update(raw[0]), numeric_inputs(raw[:1])[0], rtol=0.0, atol=1e-7
    )


def test_numeric_inputs_reject_derived_float32_overflow() -> None:
    raw = np.zeros((2, N_FEATURES), dtype=np.float32)
    raw[0, 0] = np.finfo(np.float32).max
    raw[1, 0] = -np.finfo(np.float32).max

    with pytest.raises(ValueError, match="overflowed"):
        numeric_inputs(raw)


def test_numeric_basis_clamps_and_zeros_tied_intervals() -> None:
    basis = NumericBasis(hidden=2, mode="uniform16")
    knots = torch.zeros((N_FEATURES, 17), dtype=torch.float32)
    knots[0, 2:] = torch.arange(1, 16, dtype=torch.float32)
    basis.install_knots(knots)
    with torch.no_grad():
        basis.projection.weight.zero_()
        basis.projection.weight[0, 0] = 1.0
        basis.projection.weight[1, 1] = 1.0
    values = torch.zeros((3, N_FEATURES), dtype=torch.float32)
    values[:, 0] = torch.tensor([0.0, 0.5, 50.0])

    projected = basis(values)

    assert projected.dtype == torch.float32
    torch.testing.assert_close(projected, torch.tensor([[0.0, 0.0], [0.0, 0.5], [0.0, 1.0]]))


def test_numeric_basis_constructor_preserves_global_rng_and_checks_knots() -> None:
    torch.manual_seed(73)
    state = torch.random.get_rng_state().clone()
    NumericBasis(hidden=4, mode="quantile16")
    assert torch.equal(state, torch.random.get_rng_state())

    basis = NumericBasis(hidden=4, mode="quantile16")
    invalid = torch.ones((N_FEATURES, 17), dtype=torch.float32)
    invalid[0, 4] = -1.0
    with pytest.raises(ValueError, match="nondecreasing"):
        basis.install_knots(invalid)


def test_numeric_artifact_uses_only_fitting_members_and_rejects_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "data"
    root.mkdir()
    sources = {
        "datasets/train.parquet": {"bytes": 1, "mtime_ns": 1},
        "datasets/valid.parquet": {"bytes": 1, "mtime_ns": 1},
    }
    manifest = tmp_path / "campaign.json"
    manifest_bytes = json.dumps({"sources": sources}, sort_keys=True).encode()
    manifest.write_bytes(manifest_bytes)
    split = CampaignSplit(
        manifest,
        hashlib.sha256(manifest_bytes).hexdigest(),
        0,
        2,
        (1,),
        (2,),
        (3,),
        (("train", 1, 0), ("valid", 2, 1), ("valid", 3, 2)),
    )
    train_rows = np.full((2, N_FEATURES), 1.0, dtype=np.float32)
    train_rows[1] = 3.0
    rows = {
        ("train", 1): train_rows,
        ("valid", 2): np.full((1, N_FEATURES), 5.0, dtype=np.float32),
        ("valid", 3): np.full((1, N_FEATURES), 1000.0, dtype=np.float32),
    }
    calls: list[tuple[str, int, bool, bool]] = []

    class FakeSequenceFile:
        def __init__(self, path: Path) -> None:
            self.source = Path(path).stem

        def read(self, group: int, *, targets: bool, mask: bool) -> SimpleNamespace:
            calls.append((self.source, group, targets, mask))
            return SimpleNamespace(features=rows[(self.source, group)])

    monkeypatch.setattr(preprocessing, "load_campaign_split", lambda _path, _root: split)
    monkeypatch.setattr(preprocessing, "SequenceFile", FakeSequenceFile)
    output = tmp_path / "numeric.json"
    artifact = preprocessing.prepare_numeric_artifact(
        root, manifest, output, reservoir_rows=2, seed=221022
    )

    assert calls == [("train", 1, False, False), ("valid", 2, False, False)]
    assert artifact["preparation_identity"]["full_scanned_rows"] == 3
    assert artifact["mean"][0] == 3.0
    assert artifact["mean"][NUMERIC_MISSING_OFFSET:] == [0.0] * N_FEATURES
    assert artifact["std"][NUMERIC_MISSING_OFFSET:] == [1.0] * N_FEATURES
    loaded = preprocessing.load_numeric_artifact(
        output, split.sha256, [("train", 1), ("valid", 2)], sources
    )
    assert loaded["preparation_hash"] == artifact["preparation_hash"]

    tampered = json.loads(output.read_text())
    tampered["mean"][0] = -99.0
    output.write_text(json.dumps(tampered, sort_keys=True))
    with pytest.raises(ValueError, match="hash"):
        preprocessing.load_numeric_artifact(
            output, split.sha256, [("train", 1), ("valid", 2)], sources
        )
