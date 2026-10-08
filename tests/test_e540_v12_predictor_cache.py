from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from scripts import e540_kaggle_v11_export_worker as worker
from scripts import predict_run
from wnn.data import FEATURE_COLUMNS, SEQUENCE_LENGTH, SequenceFile
from wnn.data import Sequence as WnnSequence

LINEAR_EPOCH0_SHA256 = "17c91be768a7f41cd4c6ae0b2bbce26bcf7d6fbf5e4f6509e38c4be6bd44403a"
RUN_NAME = "e540v8-linear-s540-fp32"
ROOT = Path(__file__).resolve().parents[1]


def test_call_predictor_creates_missing_cache_parent_before_invocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(worker, "ROOT", tmp_path)
    log_path = tmp_path / "e540_v11_export" / "linear-prediction.log"
    log_path.parent.mkdir(parents=True)
    observed: list[bool] = []

    def main() -> None:
        observed.append((tmp_path / "cache").is_dir())

    module = SimpleNamespace(main=main)
    assert not (tmp_path / "cache").exists()

    worker.call_predictor(module, worker.prediction_argv("linear"), log_path)

    assert observed == [True]


class _CpuTorchProxy:
    """Keep predict_run's requested CUDA device on CPU inside this test only."""

    def __init__(self, torch_module: object) -> None:
        self._torch = torch_module

    def __getattr__(self, name: str) -> object:
        return getattr(self._torch, name)

    def device(self, name: str) -> object:
        if name != "cuda":
            raise AssertionError(f"production predictor requested unexpected device {name!r}")
        return self._torch.device("cpu")


def _write_one_sequence(path: Path, group_id: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: dict[str, pa.Array] = {
        "seq_ix": pa.array(np.full(SEQUENCE_LENGTH, group_id, dtype=np.int64))
    }
    zero_feature = np.zeros(SEQUENCE_LENGTH, dtype=np.float32)
    for name in FEATURE_COLUMNS:
        columns[name] = pa.array(zero_feature)
    table = pa.table(columns)
    pq.write_table(
        table,
        path,
        compression="zstd",
        row_group_size=SEQUENCE_LENGTH,
        use_dictionary=False,
    )


def test_real_linear_checkpoint_predictor_writes_finite_feature_only_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_env = os.environ.get("E540_V12_LINEAR_EPOCH0")
    if checkpoint_env is None:
        pytest.skip("set E540_V12_LINEAR_EPOCH0 to the reviewed own V8 epoch-0 checkpoint")
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert not torch.cuda.is_available()
    checkpoint_source = Path(checkpoint_env).resolve()
    assert hashlib.sha256(checkpoint_source.read_bytes()).hexdigest() == LINEAR_EPOCH0_SHA256
    subset_source = ROOT / worker.SUBSET_PATH
    subset_raw = subset_source.read_bytes()
    subset = json.loads(subset_raw)
    actual_groups = subset["evaluation_groups"]
    assert subset["evaluation_group_count"] == 468
    assert len(actual_groups) == 468
    group_id = actual_groups[0]

    project_root = tmp_path / "extracted-project"
    checkpoint = project_root / "artifacts" / "runs" / RUN_NAME / "model_ep0.pt"
    checkpoint.parent.mkdir(parents=True)
    shutil.copyfile(checkpoint_source, checkpoint)
    manifest = project_root / worker.SUBSET_PATH
    manifest.parent.mkdir(parents=True)
    manifest.write_bytes(subset_raw)
    parquet_path = project_root / "datasets" / "valid.parquet"
    _write_one_sequence(parquet_path, group_id)
    parquet_columns = pq.ParquetFile(parquet_path).schema_arrow.names
    assert "t0" not in parquet_columns
    assert "is_scored" not in parquet_columns
    assert not (project_root / "cache").exists()

    original_sequence_file = SequenceFile
    observed_reads: list[tuple[int, bool, bool]] = []

    class _MappedSequenceFile:
        def __init__(self, path: str | Path) -> None:
            self._file = original_sequence_file(path)

        def read(self, index: int, *, targets: bool = True, mask: bool = True) -> WnnSequence:
            observed_reads.append((index, targets, mask))
            if index != group_id:
                raise AssertionError("predictor changed the selected evaluation group")
            return self._file.read(0, targets=targets, mask=mask)

    original_load_groups = predict_run.load_e540_evaluation_groups

    def one_authenticated_group(path: Path, checkpoint_payload: dict[str, object]) -> list[int]:
        groups = original_load_groups(path, checkpoint_payload)
        if groups != actual_groups:
            raise AssertionError("checkpoint membership differs from the frozen full manifest")
        return groups[:1]

    def single_process_loader_budget(batch_bytes: int, max_workers: int) -> tuple[int, int]:
        del batch_bytes, max_workers
        return 0, 0

    for name in (
        "SENTINEL_PREPROCESSOR",
        "RAW_ONLY",
        "ONLINE_NORM",
        "HEURISTICS",
        "RESEARCH_FEATURES",
        "INPUT_PROPENSITY",
        "READOUT",
        "READOUT_TILTED",
    ):
        monkeypatch.setattr(predict_run.train_rnn, name, getattr(predict_run.train_rnn, name))

    monkeypatch.setattr(worker, "ROOT", project_root)
    monkeypatch.setattr(predict_run, "ROOT", project_root)
    monkeypatch.setattr(predict_run, "SequenceFile", _MappedSequenceFile)
    monkeypatch.setattr(predict_run, "load_e540_evaluation_groups", one_authenticated_group)
    monkeypatch.setattr(predict_run, "dataloader_budget", single_process_loader_budget)
    monkeypatch.setattr(predict_run, "torch", _CpuTorchProxy(torch))
    monkeypatch.chdir(project_root)
    log_path = project_root / "e540_v11_export" / "linear-prediction.log"
    log_path.parent.mkdir(parents=True)

    worker.call_predictor(predict_run, worker.prediction_argv("linear"), log_path)

    output = project_root / "cache" / f"pred_{RUN_NAME}-eval_valid.npy"
    partial = output.with_name(output.name + ".partial")
    sidecar = output.with_suffix(".json")
    assert output.is_file()
    assert not partial.exists()
    assert sidecar.is_file()
    prediction = np.load(output, mmap_mode="r", allow_pickle=False)
    assert prediction.shape == (1, SEQUENCE_LENGTH, 2)
    assert prediction.dtype == np.dtype(np.float32)
    assert np.isfinite(prediction).all()
    assert observed_reads == [(group_id, False, False)]
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    assert metadata["checkpoint_sha256"] == LINEAR_EPOCH0_SHA256
    assert metadata["inference_precision"] == "fp32"
    assert metadata["row_groups"] == [group_id]

    receipt_path = os.environ.get("E540_V12_FIXTURE_RECEIPT")
    if receipt_path is not None:
        receipt = {
            "schema": "wnn.e540-v12-real-predictor-write-fixture.v1",
            "status": "PASS",
            "production_source_sha256": {
                "worker": hashlib.sha256(Path(worker.__file__).read_bytes()).hexdigest(),
                "predictor": hashlib.sha256(Path(predict_run.__file__).read_bytes()).hexdigest(),
                "trainer": hashlib.sha256(
                    Path(predict_run.train_rnn.__file__).read_bytes()
                ).hexdigest(),
            },
            "predictor_argv": worker.prediction_argv("linear"),
            "normalized_sys_argv": [
                "predict_run.py",
                *worker.normalize_entry_argv("predict_run.py", worker.prediction_argv("linear")),
            ],
            "test_adaptation": (
                "The production predictor authenticated the complete frozen subset and "
                "checkpoint first; a test-only selector then retained one actual eligible "
                "group ID and mapped that ID to synthetic Parquet row group 0."
            ),
            "device": "CPU",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_available": bool(torch.cuda.is_available()),
            "checkpoint_sha256": LINEAR_EPOCH0_SHA256,
            "subset_manifest_sha256": hashlib.sha256(subset_raw).hexdigest(),
            "full_evaluation_group_count": len(actual_groups),
            "full_evaluation_membership_sha256": subset["evaluation_membership_sha256"],
            "evaluation_group_id": group_id,
            "synthetic_parquet_sha256": hashlib.sha256(parquet_path.read_bytes()).hexdigest(),
            "synthetic_row_groups": 1,
            "synthetic_parquet_has_no_targets_or_mask": True,
            "reader_requests": {"targets": False, "mask": False},
            "inference_precision": metadata["inference_precision"],
            "prediction_shape": list(prediction.shape),
            "prediction_dtype": str(prediction.dtype),
            "prediction_finite": bool(np.isfinite(prediction).all()),
            "prediction_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "sidecar_sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest(),
            "prediction_log_sha256": hashlib.sha256(log_path.read_bytes()).hexdigest(),
            "partial_path_absent_after_atomic_rename": not partial.exists(),
            "cache_directory_absent_before_call": True,
            "cache_directory_created_by_worker_call_predictor": True,
        }
        destination = Path(receipt_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
