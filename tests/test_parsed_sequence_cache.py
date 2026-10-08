from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from wnn.data import FEATURE_COLUMNS, SEQUENCE_LENGTH
from wnn.parsed_sequence_cache import (
    CacheAdmissionError,
    build_parsed_cache,
    canonical_json_bytes,
    estimate_array_bytes,
    prepare_membership,
    sha256_file,
)
from wnn.parsed_sequence_cache_reader import (
    ParsedCacheError,
    ParsedSequenceCache,
    verify_parsed_cache,
)


def _write_json(path: Path, value: dict) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _write_parquet(path: Path, seq_ids: list[int], offset: int, *, include_mask: bool) -> None:
    columns = {
        "seq_ix": pa.array(
            np.concatenate([np.full(SEQUENCE_LENGTH, value, dtype=np.int64) for value in seq_ids])
        )
    }
    for column_index, name in enumerate(FEATURE_COLUMNS):
        columns[name] = pa.array(
            np.concatenate(
                [
                    np.full(
                        SEQUENCE_LENGTH,
                        offset + sequence_index + column_index / 1000,
                        dtype=np.float32,
                    )
                    for sequence_index in range(len(seq_ids))
                ]
            )
        )
    columns["t0"] = pa.array(
        np.concatenate(
            [
                np.full(SEQUENCE_LENGTH, index / 10, dtype=np.float32)
                for index in range(len(seq_ids))
            ]
        )
    )
    columns["t1"] = pa.array(
        np.concatenate(
            [
                np.full(SEQUENCE_LENGTH, -index / 10, dtype=np.float32)
                for index in range(len(seq_ids))
            ]
        )
    )
    if include_mask:
        columns["is_scored"] = pa.array(
            np.concatenate([np.arange(SEQUENCE_LENGTH) % 2 == 0 for _ in seq_ids])
        )
    pq.write_table(pa.table(columns), path, row_group_size=SEQUENCE_LENGTH, compression="zstd")


@pytest.fixture
def cache_inputs(tmp_path: Path) -> tuple[Path, Path, Path, str, str]:
    source_root = tmp_path / "source"
    data_dir = source_root / "datasets"
    data_dir.mkdir(parents=True)
    train_path = data_dir / "train.parquet"
    valid_path = data_dir / "valid.parquet"
    _write_parquet(train_path, [100, 101], 0, include_mask=False)
    _write_parquet(valid_path, [200, 201, 202], 100, include_mask=True)

    source_paths = {
        "datasets/train.parquet": train_path,
        "datasets/valid.parquet": valid_path,
    }
    source_identities = {
        name: {"bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
        for name, path in source_paths.items()
    }
    split = {
        "version": 1,
        "fold": 0,
        "n_folds": 4,
        "sources": source_identities,
        "sequences": {
            "train": [
                {"row_group": 0, "seq_ix": 100, "component": 10},
                {"row_group": 1, "seq_ix": 101, "component": 20},
            ],
            "valid": [
                {"row_group": 0, "seq_ix": 200, "component": 10},
                {"row_group": 1, "seq_ix": 201, "component": 20},
                {"row_group": 2, "seq_ix": 202, "component": 30},
            ],
        },
        "fit": {"train": [0, 1], "valid": [0, 1]},
        "evaluation": {"valid": [2]},
    }
    split_path = tmp_path / "fold0.json"
    split_sha = _write_json(split_path, split)
    registry = {
        "version": 1,
        "split_manifest_sha256": split_sha,
        "sources": {
            name: {"original_identity": source_identities[name], "sha256": sha256_file(path)}
            for name, path in source_paths.items()
        },
    }
    registry_path = tmp_path / "portable_inputs.json"
    registry_sha = _write_json(registry_path, registry)
    return source_root, split_path, registry_path, split_sha, registry_sha


def _make_membership(
    tmp_path: Path, split_path: Path, registry_path: Path, split_sha: str, registry_sha: str
) -> tuple[Path, str, dict]:
    value = prepare_membership(
        split_manifest_path=split_path,
        portable_registry_path=registry_path,
        expected_split_sha256=split_sha,
        expected_registry_sha256=registry_sha,
        fold=0,
        fraction=1.0,
        seed=19,
    )
    path = tmp_path / "membership.json"
    raw = canonical_json_bytes(value) + b"\n"
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest(), value


def test_cache_is_source_qualified_component_preserving_and_mmap_readable(
    tmp_path: Path, cache_inputs: tuple[Path, Path, Path, str, str]
) -> None:
    source_root, split_path, registry_path, split_sha, registry_sha = cache_inputs
    membership_path, membership_sha, membership = _make_membership(
        tmp_path, split_path, registry_path, split_sha, registry_sha
    )
    assert membership["member_counts"] == {"fit_train": 2, "fit_valid": 2, "evaluation_valid": 1}
    assert membership["selection"]["selected_component_ids"] == [10, 20]
    assert membership["memberships"]["evaluation_valid"][0]["seq_ix"] == 202
    assert membership["memberships"]["evaluation_valid"][0]["sequence_id"].startswith("sha256:")

    output = tmp_path / "cache"
    manifest = build_parsed_cache(
        membership_path=membership_path,
        expected_membership_sha256=membership_sha,
        source_root=source_root,
        output_dir=output,
        output_reserve_bytes=0,
        checkpoint_reserve_bytes=0,
        free_reserve_bytes=0,
    )
    assert manifest["format_version"] == 1
    assert manifest["member_counts"] == membership["member_counts"]
    assert manifest["telemetry"]["observed_parse_once"] is True
    assert manifest["telemetry"]["parsing"]["sequence_read_invocations"] == 5
    assert manifest["telemetry"]["parsing"]["duplicate_source_row_group_read_count"] == 0
    assert (
        manifest["telemetry"]["parsing"]["source_layout"]["datasets/train.parquet"][
            "missing_train_mask_policy"
        ]
        == "all_false_as_train_rnn_Sequences_getitem"
    )
    assert manifest["telemetry"]["mmap_validation"]["mapped_npy_file_count"] == 6
    assert all(entry["writeback"]["sync_count"] >= 1 for entry in manifest["arrays"].values())
    assert manifest["disk_admission"]["array_bytes"] == estimate_array_bytes(membership)
    assert sha256_file(output / "membership.json") == membership_sha
    assert (
        sha256_file(output / "source_verification.json")
        == manifest["source_verification_artifact"]["sha256"]
    )

    train_features = np.load(output / "train_features.npy", mmap_mode="r", allow_pickle=False)
    valid_features = np.load(output / "valid_features.npy", mmap_mode="r", allow_pickle=False)
    valid_targets = np.load(output / "valid_targets.npy", mmap_mode="r", allow_pickle=False)
    valid_scored = np.load(output / "valid_scored.npy", mmap_mode="r", allow_pickle=False)
    train_scored = np.load(output / "train_scored.npy", mmap_mode="r", allow_pickle=False)
    try:
        assert train_features.shape == (2, SEQUENCE_LENGTH, len(FEATURE_COLUMNS))
        assert valid_features.shape == (3, SEQUENCE_LENGTH, len(FEATURE_COLUMNS))
        assert train_features[0, 0, 0] == np.float32(0)
        assert valid_features[2, 0, 0] == np.float32(102)
        assert np.allclose(valid_targets[2, 0], [0.2, -0.2], rtol=0, atol=1e-7)
        assert valid_scored[2, :4].tolist() == [True, False, True, False]
        assert not train_scored.any()
    finally:
        for array in (train_features, valid_features, valid_targets, valid_scored, train_scored):
            array._mmap.close()


def test_disk_guard_rejects_before_creating_cache_directory(
    tmp_path: Path, cache_inputs: tuple[Path, Path, Path, str, str]
) -> None:
    source_root, split_path, registry_path, split_sha, registry_sha = cache_inputs
    membership_path, membership_sha, _ = _make_membership(
        tmp_path, split_path, registry_path, split_sha, registry_sha
    )
    output = tmp_path / "rejected-cache"
    with pytest.raises(CacheAdmissionError, match="insufficient disk headroom"):
        build_parsed_cache(
            membership_path=membership_path,
            expected_membership_sha256=membership_sha,
            source_root=source_root,
            output_dir=output,
            output_reserve_bytes=10**18,
            checkpoint_reserve_bytes=0,
            free_reserve_bytes=0,
        )
    assert not output.exists()


def test_parsed_cache_reader_reuses_verified_producer_arrays_and_enforces_roles(
    tmp_path: Path, cache_inputs: tuple[Path, Path, Path, str, str]
) -> None:
    source_root, split_path, registry_path, split_sha, registry_sha = cache_inputs
    membership_path, membership_sha, membership = _make_membership(
        tmp_path, split_path, registry_path, split_sha, registry_sha
    )
    output = tmp_path / "cache"
    manifest = build_parsed_cache(
        membership_path=membership_path,
        expected_membership_sha256=membership_sha,
        source_root=source_root,
        output_dir=output,
        output_reserve_bytes=0,
        checkpoint_reserve_bytes=0,
        free_reserve_bytes=0,
    )
    manifest_path = output / "manifest.json"
    manifest_sha = sha256_file(manifest_path)
    validation_path = tmp_path / "cache-validation.json"
    receipt, receipt_sha = verify_parsed_cache(manifest_path, manifest_sha, validation_path)
    reader = ParsedSequenceCache.open(
        manifest_path,
        expected_manifest_sha256=manifest_sha,
        validation_receipt_path=validation_path,
        expected_validation_receipt_sha256=receipt_sha,
        expected_cache_key=receipt["cache_key"],
        expected_split_manifest_sha256=split_sha,
        expected_fold=0,
    )
    try:
        assert reader.groups("fit_train") == [0, 1]
        assert set(reader.preverified_sources.sources) == {
            "datasets/train.parquet",
            "datasets/valid.parquet",
        }
        assert (
            reader.preverified_sources.sources["datasets/train.parquet"]["sha256"]
            == manifest["sources"]["train"]["sha256"]
        )
        assert reader.preverified_sources.split_manifest_sha256 == split_sha
        assert reader.preverified_sources.cache_manifest_sha256 == manifest_sha
        assert isinstance(reader.arrays["train_features"], np.memmap)
        assert not reader.arrays["train_features"].flags.writeable
        train = reader.read("train", 0, "fit_train")
        evaluation = reader.read("valid", 2, "evaluation_valid")
        assert train.seq_ix == 100
        assert train.features.shape == (SEQUENCE_LENGTH, len(FEATURE_COLUMNS))
        assert not train.scored.any()
        assert evaluation.seq_ix == 202
        assert evaluation.targets is not None
        assert np.allclose(evaluation.targets[0], [0.2, -0.2], rtol=0, atol=1e-7)
        assert evaluation.scored is not None
        assert evaluation.scored[:4].tolist() == [True, False, True, False]
        with pytest.raises(ParsedCacheError, match="not in role"):
            reader.read("valid", 2, "fit_valid")
    finally:
        reader.close()


def test_parsed_cache_reader_rejects_array_mutation_after_cohort_validation(
    tmp_path: Path, cache_inputs: tuple[Path, Path, Path, str, str]
) -> None:
    source_root, split_path, registry_path, split_sha, registry_sha = cache_inputs
    membership_path, membership_sha, _ = _make_membership(
        tmp_path, split_path, registry_path, split_sha, registry_sha
    )
    output = tmp_path / "cache"
    build_parsed_cache(
        membership_path=membership_path,
        expected_membership_sha256=membership_sha,
        source_root=source_root,
        output_dir=output,
        output_reserve_bytes=0,
        checkpoint_reserve_bytes=0,
        free_reserve_bytes=0,
    )
    manifest_path = output / "manifest.json"
    manifest_sha = sha256_file(manifest_path)
    validation_path = tmp_path / "cache-validation.json"
    receipt, receipt_sha = verify_parsed_cache(manifest_path, manifest_sha, validation_path)
    array_path = output / "train_features.npy"
    with array_path.open("r+b") as stream:
        stream.seek(-1, 2)
        last = stream.read(1)
        stream.seek(-1, 2)
        stream.write(bytes([last[0] ^ 1]))
        stream.flush()
    with pytest.raises(ParsedCacheError, match="changed after cohort validation"):
        ParsedSequenceCache.open(
            manifest_path,
            expected_manifest_sha256=manifest_sha,
            validation_receipt_path=validation_path,
            expected_validation_receipt_sha256=receipt_sha,
            expected_cache_key=receipt["cache_key"],
            expected_split_manifest_sha256=split_sha,
            expected_fold=0,
        )
