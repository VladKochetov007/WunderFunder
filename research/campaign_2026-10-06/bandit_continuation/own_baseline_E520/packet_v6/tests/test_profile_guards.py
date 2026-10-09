"""Synthetic boundary tests for the E520 profile-only admission and export guards."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from export_guards import (
    ExportContractError,
    chunk_ranges,
    prediction_storage_bytes,
    validate_chunk_shape,
    validate_streaming_export_preconditions,
)
from prepare_profile_packet import validate_review as validate_builder_review
from profile_worker import (
    ProfileAbort,
    enforce_resource_caps,
    parse_nvidia_process_bytes,
    persist_identity_check,
    require_profile_only,
    runtime_identity,
    validate_export_shape,
    validate_export_tensors,
    validate_external_review,
)


def _review_bytes(reviewer_id: str = "controller-independent") -> bytes:
    payload = {
        "experiment_id": "E520_PROFILE_LOADER_HASH_ADVICE_V2",
        "scope": "profile-only",
        "status": "approved",
        "reviewer": {"id": reviewer_id, "role": "independent-controller"},
        "reviewed_hashes": {
            name: "a" * 64
            for name in (
                "profile_worker.py",
                "profile_kernel.py",
                "prepare_profile_packet.py",
                "export_guards.py",
                "tests/conftest.py",
                "tests/test_profile_guards.py",
                "tests/test_portable_manifest_binding.py",
                "tests/test_profile_bundle.py",
                "tests/test_loader_hash_advice.py",
                "E520_PACKET_PLAN.json",
                "source_archive.tar",
            )
        },
    }
    return json.dumps(payload, sort_keys=True).encode()


def _safe_resources() -> dict[str, int | float | None]:
    return {
        "rss_peak_bytes": 1_000_000_000,
        "cuda_allocated_bytes": 1_000_000_000,
        "cuda_reserved_bytes": 1_500_000_000,
        "cuda_allocated_peak_bytes": 1_100_000_000,
        "cuda_reserved_peak_bytes": 1_600_000_000,
        "gpu_process_bytes": 1_600_000_000,
        "gpu_free_bytes": 10_000_000_000,
        "host_available_bytes": 8_000_000_000,
        "host_available_fraction": 0.30,
        "cgroup_available_bytes": 8_000_000_000,
        "cgroup_available_fraction": 0.30,
        "disk_free_bytes": 10_000_000_000,
    }


def _export_provenance() -> dict[str, str]:
    return {
        "split_manifest_sha256": "a" * 64,
        "checkpoint_sha256": "b" * 64,
        "target_order_sha256": "c" * 64,
    }


def test_profile_command_is_exact_and_rejects_fit_or_export_flags() -> None:
    require_profile_only(["--profile-warmup", "3", "--profile-steps", "20"])
    with pytest.raises(ProfileAbort, match="exactly 20"):
        require_profile_only(["--profile-warmup", "3", "--profile-steps", "19"])
    with pytest.raises(ProfileAbort, match="fit/save/export"):
        require_profile_only(
            ["--profile-warmup", "3", "--profile-steps", "20", "--save-training-state"]
        )


def test_external_review_is_hash_bound_and_cannot_self_approve() -> None:
    content = _review_bytes()
    digest = hashlib.sha256(content).hexdigest()
    approved = validate_external_review(content, digest, "weak_diversity_kaggle_actor")
    assert approved["status"] == "approved"
    with pytest.raises(ProfileAbort, match="hash mismatch"):
        validate_external_review(content, "0" * 64, "weak_diversity_kaggle_actor")
    self_review = _review_bytes("weak_diversity_kaggle_actor")
    with pytest.raises(ProfileAbort, match="independent profile-only"):
        validate_external_review(
            self_review, hashlib.sha256(self_review).hexdigest(), "weak_diversity_kaggle_actor"
        )


def test_packet_builder_requires_controller_published_review_hash() -> None:
    content = _review_bytes()
    hashes = json.loads(content)["reviewed_hashes"]
    digest = hashlib.sha256(content).hexdigest()
    assert validate_builder_review(content, digest, hashes)["status"] == "approved"
    with pytest.raises(ValueError, match="exact independent-controller"):
        validate_builder_review(content, "0" * 64, hashes)


def test_resource_gates_abort_before_any_retry() -> None:
    enforce_resource_caps(_safe_resources(), elapsed_s=10)
    low_bytes = _safe_resources()
    low_bytes["cgroup_available_bytes"] = 4_294_967_295
    with pytest.raises(ProfileAbort, match="available_bytes<4294967296"):
        enforce_resource_caps(low_bytes, elapsed_s=10)
    low_fraction = _safe_resources()
    low_fraction["host_available_fraction"] = 0.199
    with pytest.raises(ProfileAbort, match="available_fraction<0.2"):
        enforce_resource_caps(low_fraction, elapsed_s=10)
    over_limit = _safe_resources()
    over_limit["cuda_reserved_bytes"] = 2_500_000_001
    with pytest.raises(ProfileAbort, match="cuda_reserved_bytes"):
        enforce_resource_caps(over_limit, elapsed_s=10)
    peak_over_limit = _safe_resources()
    peak_over_limit["cuda_allocated_peak_bytes"] = 2_000_000_001
    with pytest.raises(ProfileAbort, match="cuda_allocated_peak_bytes"):
        enforce_resource_caps(peak_over_limit, elapsed_s=10)
    unmeasured_process = _safe_resources()
    unmeasured_process["gpu_process_bytes"] = None
    with pytest.raises(ProfileAbort, match="allocation is unavailable"):
        enforce_resource_caps(unmeasured_process, elapsed_s=10)
    missing_peak = _safe_resources()
    missing_peak["cuda_reserved_peak_bytes"] = None
    with pytest.raises(ProfileAbort, match="peak APIs are unavailable"):
        enforce_resource_caps(missing_peak, elapsed_s=10)
    over_time = _safe_resources()
    with pytest.raises(ProfileAbort, match="wall cap"):
        enforce_resource_caps(over_time, elapsed_s=1_801)


def test_export_probe_uses_fixed_chunk_bytes_without_full_prediction_storage() -> None:
    sizes = validate_export_shape(batch=16, rows=4_000, features=128, outputs=2)
    assert sizes["input_float32_bytes"] == 32_768_000
    assert sizes["output_float32_bytes_if_written"] == 512_000
    with pytest.raises(ProfileAbort, match="chunk_rows=4000"):
        validate_export_shape(batch=16, rows=20_000, features=128, outputs=2)


def test_actual_export_tensors_fail_closed_on_shape_dtype_or_nonfinite_output() -> None:
    contract = {
        "raw_shape": (16, 4_000, 80),
        "model_input_shape": (16, 4_000, 80),
        "output_shape": (16, 4_000, 2),
        "raw_features": 80,
        "model_features": 80,
        "raw_dtype": "torch.float32",
        "model_input_dtype": "torch.float32",
        "output_dtype": "torch.float32",
        "outputs_finite": True,
    }
    validate_export_tensors(**contract)
    with pytest.raises(ProfileAbort, match="output shape mismatch"):
        validate_export_tensors(**{**contract, "output_shape": (16, 4_000, 1)})
    with pytest.raises(ProfileAbort, match="output dtype"):
        validate_export_tensors(**{**contract, "output_dtype": "torch.float16"})
    with pytest.raises(ProfileAbort, match="non-finite"):
        validate_export_tensors(**{**contract, "outputs_finite": False})


def test_missing_gpu_pid_is_unavailable_and_identity_mismatch_is_persisted(
    tmp_path: Path,
) -> None:
    output = "123, 256\n"
    assert parse_nvidia_process_bytes(123, output) == 256 * 1024 * 1024
    assert parse_nvidia_process_bytes(456, output) is None
    receipt_path = tmp_path / "input_identity.json"
    identity: dict[str, object] = {"phase": "input_identity", "status": "in_progress"}
    with pytest.raises(ProfileAbort, match="identity mismatch"):
        persist_identity_check(
            receipt_path,
            identity,
            "assets-manifest.json",
            {"sha256": "a" * 64},
            {"sha256": "b" * 64},
        )
    persisted = json.loads(receipt_path.read_text())
    assert persisted["expected"]["assets-manifest.json"]["sha256"] == "a" * 64
    assert persisted["observed"]["assets-manifest.json"]["sha256"] == "b" * 64


def test_runtime_identity_records_actual_hardware_and_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args[0], 0, "Observed GPU, 1.2, 8192, GPU-uuid\n", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            device_count=lambda: 1,
            get_device_name=lambda _: "Observed GPU",
            get_device_properties=lambda _: SimpleNamespace(
                total_memory=8_589_934_592, major=8, minor=6
            ),
        ),
        __version__="test-torch",
        version=SimpleNamespace(cuda="test-cuda"),
        backends=SimpleNamespace(cudnn=SimpleNamespace(version=lambda: 90100)),
    )
    identity = runtime_identity(fake_torch)
    assert identity["selected_gpu_name"] == "Observed GPU"
    assert identity["torch_version"] == "test-torch"
    assert identity["cuda_runtime_version"] == "test-cuda"
    assert identity["nvidia_smi_identity"]["status"] == "ok"


def test_stream_export_preconditions_bind_provenance_and_disk() -> None:
    needed = validate_streaming_export_preconditions(
        rows=10_606,
        free_disk_bytes=6_000_000_000,
        reserve_bytes=1_500_000_000,
        split_sha256="a" * 64,
        expected_split_sha256="a" * 64,
        row_groups_sha256="d" * 64,
        expected_row_groups_sha256="d" * 64,
        provenance=_export_provenance(),
        checkpoint_sha256="b" * 64,
        target_order_sha256="c" * 64,
    )
    assert needed == prediction_storage_bytes(10_606)
    with pytest.raises(ExportContractError, match="row-group digest changed"):
        validate_streaming_export_preconditions(
            rows=10_606,
            free_disk_bytes=6_000_000_000,
            reserve_bytes=1_500_000_000,
            split_sha256="e" * 64,
            expected_split_sha256="a" * 64,
            row_groups_sha256="d" * 64,
            expected_row_groups_sha256="d" * 64,
            provenance=_export_provenance(),
            checkpoint_sha256="b" * 64,
            target_order_sha256="c" * 64,
        )
    with pytest.raises(ExportContractError, match="disk cannot"):
        validate_streaming_export_preconditions(
            rows=10_606,
            free_disk_bytes=3_000_000_000,
            reserve_bytes=1_500_000_000,
            split_sha256="a" * 64,
            expected_split_sha256="a" * 64,
            row_groups_sha256="d" * 64,
            expected_row_groups_sha256="d" * 64,
            provenance=_export_provenance(),
            checkpoint_sha256="b" * 64,
            target_order_sha256="c" * 64,
        )


def test_export_chunks_and_shapes_stay_bounded() -> None:
    assert chunk_ranges(8_001) == [(0, 4_000), (4_000, 8_000), (8_000, 8_001)]
    validate_chunk_shape((16, 4_000, 2), (16, 4_000, 2))
    with pytest.raises(ExportContractError, match="shape mismatch"):
        validate_chunk_shape((16, 20_000, 2), (16, 4_000, 2))
