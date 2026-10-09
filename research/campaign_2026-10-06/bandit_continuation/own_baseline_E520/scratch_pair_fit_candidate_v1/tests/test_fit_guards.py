"""Focused synthetic guards for the admitted E520 pair implementation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import evaluate_e520_pair as grader
import numpy as np
import pytest
from fit_guards import (
    FitGateError,
    stream_memmap_for_test,
    trainer_argv,
    validate_controller_admission,
    validate_prediction_chunk,
)
from support import input_identity

from wnn.portability import verify_portable_source


def test_fixed_argv_is_scratch_raw_only_and_checkpoint_pinned() -> None:
    seed3 = trainer_argv(3, "e520_raw_baseline_s3_f0")
    seed4 = trainer_argv(4, "e520_raw_baseline_s4_f0")
    assert seed3[seed3.index("--seed") + 1] == "3"
    assert seed4[seed4.index("--seed") + 1] == "4"
    assert seed3[seed3.index("--checkpoint-epoch") + 1] == "3"
    assert "--raw-only" in seed3
    assert seed3[seed3.index("--init") + 1] == ""
    assert seed3[seed3.index("--distill") + 1] == ""
    assert seed3.count("--seed") == 1
    with pytest.raises(ValueError):
        trainer_argv(5, "e520_raw_baseline_s5_f0")


def test_admission_fails_closed_without_external_live_receipts() -> None:
    with pytest.raises(FitGateError, match="stale, incomplete, or mismatched"):
        validate_controller_admission(
            b"{}",
            hashlib.sha256(b"{}").hexdigest(),
            review_sha256="a" * 64,
            hashes={"FIT_STAGE_CONFIG.json": "b" * 64},
            config={
                "recipe": {},
                "input_identity": {"static_hashes": {}},
                "resource_policy": {
                    "account_reserve_fraction": 0.2,
                    "max_worker_wall_seconds": 10_200,
                },
                "evaluation": {"anchor_evaluation_report_sha256": "c" * 64},
            },
        )


def test_streaming_memmap_covers_row_batches_and_time_chunks(tmp_path: Path) -> None:
    output = tmp_path / "pred.npy"
    first = np.full((2, 4, 2), 1.25, dtype=np.float32)
    second = np.full((2, 4, 2), -0.5, dtype=np.float32)
    receipt = stream_memmap_for_test(
        output,
        [(0, 0, first), (0, 4, second)],
        rows=2,
        sequence_rows=8,
        chunk_rows=4,
    )
    assert receipt["shape"] == [2, 8, 2]
    predictions = np.load(output, mmap_mode="r", allow_pickle=False)
    assert predictions.shape == (2, 8, 2)
    assert np.all(predictions[:, :4] == 1.25)
    assert np.all(predictions[:, 4:] == -0.5)
    predictions._mmap.close()


def test_streaming_memmap_rejects_missing_or_overlapping_chunks(tmp_path: Path) -> None:
    sample = np.ones((2, 4, 2), dtype=np.float32)
    with pytest.raises(FitGateError, match="every output slice"):
        stream_memmap_for_test(
            tmp_path / "missing.npy",
            [(0, 0, sample)],
            rows=2,
            sequence_rows=8,
            chunk_rows=4,
        )
    with pytest.raises(FitGateError, match="repeated"):
        stream_memmap_for_test(
            tmp_path / "overlap.npy",
            [(0, 0, sample), (0, 0, sample)],
            rows=2,
            sequence_rows=4,
            chunk_rows=4,
        )


def test_streaming_memmap_rejects_nonfinite_or_wrong_shape(tmp_path: Path) -> None:
    with pytest.raises(FitGateError, match="non-finite"):
        validate_prediction_chunk(
            np.full((1, 4, 2), np.nan, dtype=np.float32), expected_shape=(1, 4, 2)
        )
    with pytest.raises(FitGateError, match="shape"):
        validate_prediction_chunk(np.ones((1, 3, 2), dtype=np.float32), expected_shape=(1, 4, 2))
    with pytest.raises(ValueError, match="exactly chunkable"):
        stream_memmap_for_test(
            tmp_path / "bad-shape.npy",
            [],
            rows=2,
            sequence_rows=7,
            chunk_rows=4,
        )


def test_portable_registry_binding_is_exact_and_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "source"
    relative = input_identity.PORTABLE_RELATIVE
    manifest = source_root / relative
    manifest.parent.mkdir(parents=True)
    manifest.write_bytes(b'{"version":1,"sources":{}}\n')
    receipt_path = tmp_path / "input_identity.json"
    identity: dict[str, Any] = {"expected": {}, "observed": {}}
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    monkeypatch.setattr(input_identity, "PORTABLE_SHA256", digest)
    monkeypatch.delenv("WNN_PORTABLE_DATA_MANIFEST", raising=False)

    result = input_identity.bind_verified_portable_manifest(source_root, receipt_path, identity)

    assert result == manifest
    assert Path(json.loads(receipt_path.read_text())["portable_manifest_env"]) == manifest
    assert Path(os.environ["WNN_PORTABLE_DATA_MANIFEST"]) == manifest

    monkeypatch.delenv("WNN_PORTABLE_DATA_MANIFEST", raising=False)
    monkeypatch.setattr(input_identity, "PORTABLE_SHA256", "0" * 64)
    failed_receipt = tmp_path / "failed_identity.json"
    failed_identity: dict[str, Any] = {"expected": {}, "observed": {}}
    with pytest.raises(input_identity.ProfileAbort, match="portable_manifest_binding"):
        input_identity.bind_verified_portable_manifest(source_root, failed_receipt, failed_identity)
    recorded = json.loads(failed_receipt.read_text())
    assert recorded["expected"]["portable_manifest_binding"]["sha256"] == "0" * 64
    assert recorded["observed"]["portable_manifest_binding"]["sha256"] == digest
    assert "WNN_PORTABLE_DATA_MANIFEST" not in os.environ


def test_anchor_report_frozen_pair_mask_hashes_and_exposure() -> None:
    spec = {
        "anchor": {
            "reference_grader_spec_sha256": "a" * 64,
            "reference_grader_script_sha256": "b" * 64,
            "pair_blend": "equal 0.5/0.5 raw seed predictions before official metric clipping",
            "data_exposure": "Fold-0 evaluation groups are exposed development data.",
        },
        "split": {
            "sha256": "c" * 64,
            "valid_target_cache_sha256": "d" * 64,
            "valid_target_row_groups_sha256": "e" * 64,
        },
        "metric": {"selection": "need_prediction AND is_scored"},
    }
    report = {
        "metric_id": grader.METRIC_ID,
        "grader_spec": {"sha256": "a" * 64},
        "grader_source_sha256": "b" * 64,
        "split": {"sha256": "c" * 64},
        "targets": {"sha256": "d" * 64, "row_groups_sha256": "e" * 64},
        "selection": "need_prediction AND is_scored",
        "pair_blend": spec["anchor"]["pair_blend"],
        "data_exposure": spec["anchor"]["data_exposure"],
    }
    grader._validate_anchor_report(report, spec)
    report["selection"] = "is_scored"
    with pytest.raises(grader.GraderInputError, match="comparison contract"):
        grader._validate_anchor_report(report, spec)


def test_grader_binds_repository_portable_registry_after_hash_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry_rel = "research/campaign_2026-10-04/portable_inputs.json"
    registry = tmp_path / registry_rel
    registry.parent.mkdir(parents=True)
    registry.write_text('{"version":1,"sources":{}}\n')
    digest = hashlib.sha256(registry.read_bytes()).hexdigest()
    spec = {"input_identity": {"asset_hashes": {registry_rel: digest}}}
    monkeypatch.setenv("WNN_PORTABLE_DATA_MANIFEST", "/stale/registry.json")

    assert grader._bind_repository_portable_manifest(tmp_path, spec) == registry
    assert Path(os.environ["WNN_PORTABLE_DATA_MANIFEST"]) == registry

    monkeypatch.setenv("WNN_PORTABLE_DATA_MANIFEST", "/stale/registry.json")
    spec["input_identity"]["asset_hashes"][registry_rel] = "0" * 64
    with pytest.raises(grader.GraderInputError, match="SHA-256 mismatch"):
        grader._bind_repository_portable_manifest(tmp_path, spec)
    assert "WNN_PORTABLE_DATA_MANIFEST" not in os.environ


def test_portable_source_uses_content_hash_when_file_mtime_moves(tmp_path: Path) -> None:
    source = tmp_path / "relocated.json"
    source.write_bytes(b"unchanged portable bytes")
    original_identity = {"bytes": source.stat().st_size, "mtime_ns": 1}
    content_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "version": 1,
                "sources": {
                    "data.json": {
                        "original_identity": original_identity,
                        "sha256": content_sha,
                    }
                },
            }
        )
    )

    verify_portable_source(source, "data.json", original_identity, registry)
    source.write_bytes(b"different portable bytes")
    with pytest.raises(ValueError, match="content changed"):
        verify_portable_source(source, "data.json", original_identity, registry)


def test_official_mask_excludes_the_first_99_steps() -> None:
    scored = np.ones((1, grader.TARGET_ROWS), dtype=np.bool_)
    selected = grader._official_mask(scored)
    assert not selected[:, :99].any()
    assert selected[:, 99:].all()
