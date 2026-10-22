"""Pure NumPy consumer checks for online relationship sequence caches."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from wnn.training.online_relationship_cache import (
    CACHE_SCHEMA,
    PARENT,
    RAW,
    ROWS,
    TARGET,
    WEIGHT,
    WIDTH,
    SequenceCache,
    load_sequence_cache,
    row_refs_sha256,
    sequence_cache_paths,
)

from wnn.splits import file_sha256

RECIPE_SHA256 = "a" * 64


def _write_eval_cache(
    directory: Path,
    *,
    target_value: float = 0.0,
    weight_value: float = 0.0,
) -> tuple[Path, Path, dict[str, object]]:
    """Write one real NPY sequence and the delivery sidecar fields it consumes."""
    directory.mkdir(parents=True, exist_ok=True)
    data_path, sidecar_path = sequence_cache_paths(directory, "feature_only_eval")
    data = np.lib.format.open_memmap(data_path, mode="w+", dtype="<f4", shape=(1, ROWS, WIDTH))
    data[:] = 0.0
    data[0, :, RAW.start] = np.float32(0.25)
    data[0, :, PARENT.start] = np.float32(-1.5)
    data[0, :, PARENT.start + 1] = np.float32(2.5)
    data[0, 17, TARGET.start] = np.float32(target_value)
    data[0, 19, WEIGHT] = np.float32(weight_value)
    data.flush()
    data._mmap.close()

    row_refs = [{"split": "valid", "row_group": 13}]
    metadata: dict[str, object] = {
        "schema": CACHE_SCHEMA,
        "scope": "feature_only_eval",
        "prediction_scope": "feature_only_eval",
        "fold": 0,
        "seed": 1001,
        "recipe_id": "test-online-relationship-cache",
        "recipe_sha256": RECIPE_SHA256,
        "split_manifest": "fixtures/fold0.json",
        "split_manifest_sha256": "b" * 64,
        "subset_manifest": "fixtures/subset-f0.json",
        "subset_manifest_sha256": "c" * 64,
        "subset_members_sha256": "d" * 64,
        "row_refs": row_refs,
        "membership_sha256": row_refs_sha256(row_refs),
        "complete_membership": True,
        "scope_label": "feature_only_eval",
        "source_datasets": {},
        "parent": {},
        "propensity": None,
        "parent_mean": [0.0] * 211,
        "parent_std": [1.0] * 211,
        "row_weight_contract": None,
        "row_weight_batch_semantics": None,
        "read_contract": {"targets": False, "scored_mask": False},
        "data_file": data_path.name,
        "shape": [1, ROWS, WIDTH],
        "dtype": "<f4",
        "layout": {
            "raw": [0, 112],
            "target": [112, 114],
            "row_weight": [114, 115],
            "parent_prediction": [115, 117],
        },
        "zero_columns": [112, 113, 114],
        "parent_prediction_file": "evaluation_predictions.npy",
        "parent_prediction_sha256": "e" * 64,
        "source_sha256": {},
        "data_sha256": file_sha256(data_path),
        "sampled_profile": False,
    }
    sidecar_path.write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")
    return data_path, sidecar_path, metadata


def _load(data_path: Path, sidecar_path: Path) -> SequenceCache:
    """Call the delivery cache consumer with the frozen evaluation identity."""
    return load_sequence_cache(
        data_path,
        sidecar_path,
        expected_scope="feature_only_eval",
        expected_recipe_sha256=RECIPE_SHA256,
        expected_fold=0,
    )


def test_cache_yields_four_views_with_source_order(tmp_path: Path) -> None:
    """The mmap reader preserves source refs and exposes four zero-copy arrays."""
    data_path, sidecar_path, metadata = _write_eval_cache(tmp_path)

    cache = _load(data_path, sidecar_path)
    records = list(cache.iter_with_refs())

    assert cache.row_refs == ({"split": "valid", "row_group": 13},)
    assert len(records) == 1
    ref, sequence = records[0]
    assert ref == metadata["row_refs"][0]
    assert sequence.raw.shape == (ROWS, 112)
    assert sequence.target.shape == (ROWS, 2)
    assert sequence.row_weight.shape == (ROWS,)
    assert sequence.parent_prediction.shape == (ROWS, 2)
    assert sequence.raw.dtype == np.dtype("<f4")
    assert np.all(sequence.raw[:, 0] == np.float32(0.25))
    assert np.all(sequence.parent_prediction[:, 0] == np.float32(-1.5))
    assert np.all(sequence.parent_prediction[:, 1] == np.float32(2.5))
    for view in sequence:
        assert np.shares_memory(view, cache.array)


@pytest.mark.parametrize("damage", ["bad_magic", "truncated_header"])
def test_corrupt_or_truncated_npy_header_is_rejected(tmp_path: Path, damage: str) -> None:
    """The real mmap loader rejects broken NPY headers even with a matching hash."""
    data_path, sidecar_path, metadata = _write_eval_cache(tmp_path)
    with data_path.open("r+b") as stream:
        if damage == "bad_magic":
            stream.write(b"notnpy")
        else:
            stream.truncate(8)
    metadata["data_sha256"] = file_sha256(data_path)
    sidecar_path.write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError):
        _load(data_path, sidecar_path)


@pytest.mark.parametrize(("target_value", "weight_value"), [(1.0, 0.0), (0.0, 1.0)])
def test_evaluation_cache_rejects_nonzero_targets_or_weights(
    tmp_path: Path, target_value: float, weight_value: float
) -> None:
    """Evaluation payloads cannot expose targets or fitting weights to consumers."""
    data_path, sidecar_path, _ = _write_eval_cache(
        tmp_path, target_value=target_value, weight_value=weight_value
    )
    cache = _load(data_path, sidecar_path)

    with pytest.raises(ValueError, match="contains target or weight"):
        list(cache.iter_sequences())
