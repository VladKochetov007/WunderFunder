from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from worker import (
    expected_inputs,
    link_staged_inputs,
    validate_prediction,
    validate_staged_manifest,
)


def _input_contract() -> list[dict[str, object]]:
    return [
        {
            "path": f"inputs/file_{index}.bin",
            "sha256": f"{index:064x}",
            "size_bytes": index + 1,
        }
        for index in range(10)
    ]


def _stage(tmp_path: Path) -> tuple[Path, dict[str, dict[str, object]], str]:
    input_root = tmp_path / "inputs"
    records = _input_contract()
    expected = {str(item["path"]): item for item in records}
    manifest: list[dict[str, object]] = []
    for index, item in enumerate(records):
        path = input_root / str(item["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(bytes([index]) * int(item["size_bytes"]))
        manifest.append(dict(item))
    return input_root, expected, json.dumps(manifest)


def test_worker_requires_exact_ten_input_records() -> None:
    with pytest.raises(ValueError, match="ten-file"):
        expected_inputs({"inputs": _input_contract()[:-1]})


def test_stage_manifest_accepts_only_frozen_paths_sizes_and_hashes(tmp_path: Path) -> None:
    input_root, expected, raw_manifest = _stage(tmp_path)

    receipt = validate_staged_manifest(raw_manifest, input_root, expected)

    assert len(receipt) == 10
    assert receipt[0]["verification"] == "vast_campaign_remote_sha256_and_local_size"


def test_stage_manifest_rejects_changed_remote_digest(tmp_path: Path) -> None:
    input_root, expected, raw_manifest = _stage(tmp_path)
    rows = json.loads(raw_manifest)
    rows[0]["sha256"] = "f" * 64

    with pytest.raises(ValueError, match="supervisor identity"):
        validate_staged_manifest(json.dumps(rows), input_root, expected)


def test_stage_manifest_rejects_missing_duplicate_and_extra_rows(tmp_path: Path) -> None:
    input_root, expected, raw_manifest = _stage(tmp_path)
    rows = json.loads(raw_manifest)

    for malformed in (rows[:-1], [*rows, {**rows[0], "path": "extra"}]):
        with pytest.raises(ValueError, match="input set"):
            validate_staged_manifest(json.dumps(malformed), input_root, expected)
    with pytest.raises(ValueError, match="missing or duplicated"):
        validate_staged_manifest(json.dumps([*rows, rows[0]]), input_root, expected)


def test_linking_preserves_project_paths_and_refuses_replacement(tmp_path: Path) -> None:
    input_root, expected, raw_manifest = _stage(tmp_path)
    validate_staged_manifest(raw_manifest, input_root, expected)
    project = tmp_path / "project"

    link_staged_inputs(project, input_root, expected)

    first = project / "inputs/file_0.bin"
    assert first.is_symlink()
    assert first.resolve() == (input_root / "inputs/file_0.bin").resolve()
    with pytest.raises(FileExistsError, match="refusing to replace"):
        link_staged_inputs(project, input_root, expected)


def test_stage_manifest_rejects_symlinks_and_wrong_local_sizes(tmp_path: Path) -> None:
    input_root, expected, raw_manifest = _stage(tmp_path)
    first = input_root / "inputs/file_0.bin"
    first.unlink()
    target = tmp_path / "target.bin"
    target.write_bytes(b"x")
    first.symlink_to(target)

    with pytest.raises(ValueError, match="absent or has wrong size"):
        validate_staged_manifest(raw_manifest, input_root, expected)


def _prediction_fixture(tmp_path: Path) -> tuple[Path, Path, Path, str]:
    project = tmp_path / "project"
    cache = project / "cache"
    cache.mkdir(parents=True)
    target_path = cache / "train_targets.npz"
    np.savez(target_path, row_group=np.array([17], dtype=np.int64))
    prediction_path = cache / "pred_run_train.npy"
    predictions = np.lib.format.open_memmap(
        prediction_path,
        mode="w+",
        dtype=np.float32,
        shape=(1, 20_000, 2),
    )
    predictions.fill(np.float32(0.25))
    predictions.flush()
    del predictions
    digest = hashlib.sha256(prediction_path.read_bytes()).hexdigest()
    sidecar_path = prediction_path.with_suffix(".json")
    sidecar_path.write_text(
        json.dumps(
            {
                "run": "run",
                "epoch": 3,
                "inference_precision": "fp32",
                "checkpoint_sha256": "c" * 64,
                "prediction_sha256": digest,
                "row_groups": [17],
            }
        ),
        encoding="utf-8",
    )
    return project, prediction_path, sidecar_path, digest


def _validate_fixture(project: Path) -> dict[str, object]:
    return validate_prediction(
        project,
        "train",
        "run",
        "run",
        1,
        "c" * 64,
        "t" * 64,
    )


def test_prediction_accepts_real_npy_header_and_matches_sidecar_hash(tmp_path: Path) -> None:
    project, prediction_path, _, digest = _prediction_fixture(tmp_path)

    receipt = _validate_fixture(project)

    mapped = np.load(prediction_path, mmap_mode="r", allow_pickle=False)
    assert mapped.shape == (1, 20_000, 2)
    assert mapped.offset == 128
    assert prediction_path.stat().st_size == mapped.offset + mapped.nbytes
    assert receipt["prediction_sha256"] == digest
    assert receipt["size_bytes"] == prediction_path.stat().st_size


def test_prediction_rejects_truncated_real_npy_payload(tmp_path: Path) -> None:
    project, prediction_path, _, _ = _prediction_fixture(tmp_path)
    with prediction_path.open("r+b") as stream:
        stream.truncate(prediction_path.stat().st_size - 1)

    with pytest.raises((ValueError, OSError)):
        _validate_fixture(project)


def test_prediction_rejects_trailing_bytes_after_real_npy_payload(tmp_path: Path) -> None:
    project, prediction_path, _, _ = _prediction_fixture(tmp_path)
    with prediction_path.open("ab") as stream:
        stream.write(b"x")

    with pytest.raises(ValueError, match="byte size"):
        _validate_fixture(project)


def test_prediction_rejects_sidecar_hash_mismatch(tmp_path: Path) -> None:
    project, _, sidecar_path, _ = _prediction_fixture(tmp_path)
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["prediction_sha256"] = "f" * 64
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")

    with pytest.raises(ValueError, match="file hash differs"):
        _validate_fixture(project)
