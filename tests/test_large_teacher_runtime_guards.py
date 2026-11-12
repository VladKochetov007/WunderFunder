"""Source checks for large-teacher recipe guards and profile memory receipts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from wnn.training.sequence_config import (
    load_config,
    validate_expected_fold,
    validate_runtime_requirements,
)
from wnn.training.sequence_distributed import DistributedContext
from wnn.training.sequence_provenance import run_identity, validate_profile_receipt


def _recipe(tmp_path: Path, values: dict[str, object]) -> dict[str, object]:
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(values), encoding="utf-8")
    return load_config(path)


def test_runtime_guards_are_normalized_with_historical_defaults(tmp_path: Path) -> None:
    defaults = _recipe(tmp_path, {})
    assert defaults["required_world_size"] is None
    assert defaults["required_device"] is None
    assert defaults["expected_fold"] is None
    assert defaults["training_mask"] == "is_scored"
    assert defaults["normalization_scope"] == "prefix"
    default_identity = run_identity(
        defaults,
        {},
        {"sha256": "source", "files": {}},
        {},
        1,
    )

    configured = _recipe(
        tmp_path,
        {
            "required_world_size": 8,
            "required_device": "B300",
            "expected_fold": 2,
            "training_mask": "all_after_warmup",
            "normalization_scope": "all_fit",
        },
    )
    assert configured["required_world_size"] == 8
    assert configured["required_device"] == "B300"
    assert configured["expected_fold"] == 2
    assert configured["training_mask"] == "all_after_warmup"
    assert configured["normalization_scope"] == "all_fit"
    configured_identity = run_identity(
        configured,
        {},
        {"sha256": "source", "files": {}},
        {},
        1,
    )
    assert default_identity["config_sha256"] != configured_identity["config_sha256"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("required_world_size", True),
        ("required_world_size", 0),
        ("required_device", "B200"),
        ("expected_fold", True),
        ("expected_fold", -1),
        ("training_mask", "all_rows"),
        ("normalization_scope", "training_only"),
    ],
)
def test_runtime_guards_reject_invalid_recipe_values(
    tmp_path: Path, field: str, value: object
) -> None:
    with pytest.raises(ValueError):
        _recipe(tmp_path, {field: value})


def test_runtime_requirements_are_pure_and_check_rank_local_device() -> None:
    config = {
        "required_world_size": 8,
        "required_device": "B300",
    }
    assert validate_runtime_requirements(config, 8, "NVIDIA B300")
    with pytest.raises(ValueError, match="world size"):
        validate_runtime_requirements(config, 1, "NVIDIA B300")
    with pytest.raises(ValueError, match="B300"):
        validate_runtime_requirements(config, 8, "NVIDIA H100")

    assert validate_runtime_requirements(
        {"required_world_size": None, "required_device": None}, 1, "CPU test"
    )


def test_expected_fold_guard_accepts_matching_fold_and_rejects_drift() -> None:
    config = {"expected_fold": 2}
    assert validate_expected_fold(config, 2)
    with pytest.raises(ValueError, match="requires fold 2"):
        validate_expected_fold(config, 0)
    assert validate_expected_fold({"expected_fold": None}, 0)


def test_runtime_identity_records_driver_uuid_and_device_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from wnn.training import sequence_provenance

    properties = SimpleNamespace(major=10, minor=0, uuid=b"GPU-test", total_memory=100)
    monkeypatch.setattr(
        sequence_provenance.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="590.00\n"),
    )
    monkeypatch.setattr(
        sequence_provenance.torch.cuda, "get_device_properties", lambda _: properties
    )
    monkeypatch.setattr(sequence_provenance.torch.cuda, "get_device_name", lambda _: "NVIDIA B300")
    monkeypatch.setattr(sequence_provenance.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(sequence_provenance.torch.backends.cudnn, "version", lambda: 9010)
    monkeypatch.setattr(
        sequence_provenance.torch, "get_float32_matmul_precision", lambda: "highest"
    )
    monkeypatch.setattr(sequence_provenance.torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(sequence_provenance.torch.backends.cudnn, "allow_tf32", False)
    monkeypatch.setenv("NVIDIA_TF32_OVERRIDE", "0")
    monkeypatch.setenv("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "0")
    context = DistributedContext(0, 0, 1, sequence_provenance.torch.device("cuda", 0), False)

    runtime = sequence_provenance.runtime_identity(context)

    assert runtime["nvidia_driver"] == "590.00"
    assert (
        runtime["cudnn_rnn_fp32_precision"]
        == sequence_provenance.torch.backends.cudnn.rnn.fp32_precision
    )
    assert runtime["nvidia_tf32_override"] == "0"
    assert runtime["torch_allow_tf32_cublas_override"] == "0"
    assert runtime["devices"] == [
        {
            "rank": 0,
            "local_rank": 0,
            "name": "NVIDIA B300",
            "capability": [10, 0],
            "uuid": "GPU-test",
            "total_memory_bytes": 100,
        }
    ]


def _profile_receipt() -> tuple[dict[str, object], dict[str, object]]:
    identity: dict[str, object] = {
        "world_size": 1,
        "model_parameters": {"total": 4, "trainable": 4},
        "runtime": {"devices": [{"rank": 0, "uuid": "GPU-test", "total_memory_bytes": 100}]},
    }
    receipt: dict[str, object] = {
        "receipt_schema": "wnn.sequence-profile.v1",
        "status": "profile_complete",
        "approved": True,
        "identity": identity,
        "updates": 1,
        "forwarded_rows": 2,
        "real_rows": 2,
        "valid_rows": 2,
        "epochs_traversed": 0,
        "elapsed_seconds": 1.0,
        "scientific_checkpoint_written": False,
        "errors": [],
        "error_count": 0,
        "losses": {"label": 1.0, "distillation": 0.0, "hint": 0.0, "total": 1.0},
        "model_parameters": identity["model_parameters"],
        "memory_by_rank": [
            {
                "rank": 0,
                "peak_allocated_bytes": 10,
                "peak_reserved_bytes": 20,
                "device_total_memory_bytes": 100,
                "host_peak_rss_bytes": 50,
            }
        ],
        "peak_memory_bytes": 10,
        "peak_reserved_bytes": 20,
        "device_total_memory_bytes": [100],
        "host_peak_rss_bytes": [50],
    }
    return identity, receipt


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("peak_allocated_bytes", 21, "allocated memory"),
        ("peak_reserved_bytes", 101, "reserved memory"),
        ("peak_allocated_bytes", math.inf, "positive integer byte counts"),
        ("device_total_memory_bytes", 0, "positive integer byte counts"),
    ],
)
def test_profile_receipt_requires_finite_positive_memory_within_device_capacity(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    identity, receipt = _profile_receipt()
    memory_by_rank = receipt["memory_by_rank"]
    assert isinstance(memory_by_rank, list) and isinstance(memory_by_rank[0], dict)
    memory_by_rank[0][field] = value
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        validate_profile_receipt(path, identity)


def test_profile_receipt_binds_per_rank_memory_to_runtime_identity(tmp_path: Path) -> None:
    identity, receipt = _profile_receipt()
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    assert len(validate_profile_receipt(path, identity)) == 64

    memory_by_rank = receipt["memory_by_rank"]
    assert isinstance(memory_by_rank, list) and isinstance(memory_by_rank[0], dict)
    memory_by_rank[0]["device_total_memory_bytes"] = 99
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match the runtime identity"):
        validate_profile_receipt(path, identity)
