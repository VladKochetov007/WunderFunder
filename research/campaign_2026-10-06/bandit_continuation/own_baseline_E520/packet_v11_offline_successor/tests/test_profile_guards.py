"""Synthetic boundary tests for the E520 profile-only admission and export guards."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import profile_worker
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
    GPU_QUERY_RAW_MAX_BYTES,
    PACKET_ID,
    PROFILE_WALL_CAP_S,
    REVIEWED_PACKET_FILES,
    GpuTelemetrySampler,
    LiveGuard,
    ProfileAbort,
    enforce_resource_caps,
    nvidia_smi_gpu_sample,
    parse_nvidia_process_bytes,
    parse_nvidia_smi_gpu_rows,
    persist_identity_check,
    require_physical_gpu_inventory,
    require_profile_only,
    resource_snapshot,
    runtime_identity,
    validate_export_shape,
    validate_export_tensors,
    validate_external_review,
)

TEST_SELECTED_GPU_UUID = "GPU-selected-test"


def _review_bytes(reviewer_id: str = "controller-independent") -> bytes:
    payload = {
        "experiment_id": PACKET_ID,
        "scope": "profile-only",
        "status": "approved",
        "reviewer": {"id": reviewer_id, "role": "independent-controller"},
        "reviewed_hashes": {name: "a" * 64 for name in REVIEWED_PACKET_FILES},
    }
    return json.dumps(payload, sort_keys=True).encode()


def _safe_resources() -> dict[str, int | float | str | bool | None]:
    return {
        "rss_peak_bytes": 1_000_000_000,
        "cuda_allocated_bytes": 1_000_000_000,
        "cuda_reserved_bytes": 1_500_000_000,
        "cuda_allocated_peak_bytes": 1_100_000_000,
        "cuda_reserved_peak_bytes": 1_600_000_000,
        "gpu_process_bytes": 1_600_000_000,
        "gpu_process_query_status": "matched_pid",
        "gpu_selected_physical_gpu_uuid": TEST_SELECTED_GPU_UUID,
        "gpu_bound_physical_gpu_uuid": TEST_SELECTED_GPU_UUID,
        "gpu_free_bytes": 13_000_000_000,
        "gpu_total_bytes": 15_000_000_000,
        "gpu_memory_info_total_bytes": 15_000_000_000,
        "gpu_memory_totals_match": True,
        "gpu_device_used_upper_bound_bytes": 2_000_000_000,
        "gpu_resource_query_status": "ok",
        "host_available_bytes": 8_000_000_000,
        "host_available_fraction": 0.30,
        "cgroup_available_bytes": 8_000_000_000,
        "cgroup_available_fraction": 0.30,
        "disk_free_bytes": 10_000_000_000,
        "cuda_initialized": True,
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
    changed_device = _safe_resources()
    changed_device["gpu_selected_physical_gpu_uuid"] = "GPU-other"
    with pytest.raises(ProfileAbort, match="UUID changed or is unavailable"):
        enforce_resource_caps(changed_device, elapsed_s=10)
    unmeasured_process = _safe_resources()
    unmeasured_process["gpu_process_bytes"] = None
    unmeasured_process["gpu_process_query_status"] = "pid_not_reported"
    enforce_resource_caps(unmeasured_process, elapsed_s=10)
    over_conservative_bound = _safe_resources()
    over_conservative_bound["gpu_process_bytes"] = None
    over_conservative_bound["gpu_process_query_status"] = "pid_not_reported"
    over_conservative_bound["gpu_device_used_upper_bound_bytes"] = 3_000_000_001
    with pytest.raises(ProfileAbort, match="device-wide GPU usage upper bound"):
        enforce_resource_caps(over_conservative_bound, elapsed_s=10)
    unavailable_bound = _safe_resources()
    unavailable_bound["gpu_process_bytes"] = None
    unavailable_bound["gpu_process_query_status"] = "query_error"
    unavailable_bound["gpu_memory_totals_match"] = False
    with pytest.raises(ProfileAbort, match="total/free measurement is unavailable"):
        enforce_resource_caps(unavailable_bound, elapsed_s=10)
    invalid_idle_marker = _safe_resources()
    invalid_idle_marker["gpu_process_bytes"] = None
    invalid_idle_marker["gpu_process_query_status"] = "pre_cuda_idle"
    with pytest.raises(ProfileAbort, match="pre-CUDA idle marker is invalid"):
        enforce_resource_caps(invalid_idle_marker, elapsed_s=10)
    missing_peak = _safe_resources()
    missing_peak["cuda_reserved_peak_bytes"] = None
    with pytest.raises(ProfileAbort, match="allocator current/peak APIs are unavailable"):
        enforce_resource_caps(missing_peak, elapsed_s=10)
    over_time = _safe_resources()
    with pytest.raises(ProfileAbort, match="wall cap"):
        enforce_resource_caps(over_time, elapsed_s=PROFILE_WALL_CAP_S + 1)


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


def test_empty_process_table_stays_unavailable_and_device_bound_is_computed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        profile_worker,
        "_nvidia_process_measurement",
        lambda _pid: {
            "bytes": None,
            "status": "pid_not_reported",
            "raw": "",
            "raw_byte_length": 0,
            "raw_sha256": hashlib.sha256(b"").hexdigest(),
            "raw_truncated": False,
            "stderr": "",
            "stderr_byte_length": 0,
            "stderr_sha256": hashlib.sha256(b"").hexdigest(),
            "stderr_truncated": False,
            "return_code": 0,
            "diagnostic": None,
        },
    )
    fake_cuda = SimpleNamespace(
        is_available=lambda: True,
        get_device_properties=lambda _index: SimpleNamespace(total_memory=15_000_000_000),
        mem_get_info=lambda _index: (14_000_000_000, 15_000_000_000),
        memory_allocated=lambda _index: 100_000_000,
        memory_reserved=lambda _index: 200_000_000,
        max_memory_allocated=lambda _index: 100_000_000,
        max_memory_reserved=lambda _index: 200_000_000,
    )
    monkeypatch.setattr(
        profile_worker,
        "nvidia_smi_gpu_sample",
        lambda **_kwargs: {
            "status": "ok",
            "raw_output_sha256": "d" * 64,
            "rows": [
                {
                    "uuid": TEST_SELECTED_GPU_UUID,
                    "memory_total_mib": 15_360,
                    "memory_used_mib": 120,
                }
            ],
        },
    )
    snapshot = resource_snapshot(
        SimpleNamespace(cuda=fake_cuda),
        tmp_path,
        expected_physical_gpu_uuid=TEST_SELECTED_GPU_UUID,
    )
    assert snapshot["gpu_process_bytes"] is None
    assert snapshot["gpu_process_query_status"] == "pid_not_reported"
    assert snapshot["gpu_device_used_upper_bound_bytes"] == 2_106_127_360
    assert snapshot["gpu_free_bytes"] == 14_000_000_000


def test_pre_cuda_idle_is_only_emitted_without_a_torch_module(tmp_path: Path) -> None:
    before_torch = resource_snapshot(None, tmp_path)
    unavailable_cuda = resource_snapshot(
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)), tmp_path
    )
    assert before_torch["gpu_process_query_status"] == "pre_cuda_idle"
    assert before_torch["cuda_initialized"] is False
    assert unavailable_cuda["gpu_process_query_status"] == "cuda_unavailable"
    assert unavailable_cuda["gpu_allocation_measurement_basis"] == "unavailable"


def test_process_query_nonzero_exit_preserves_raw_failure_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failed(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args[0], 9, "partial process rows", "NVML denied")

    monkeypatch.setattr(subprocess, "run", failed)
    measurement = profile_worker._nvidia_process_measurement(123)
    assert measurement["bytes"] is None
    assert measurement["status"] == "query_error"
    assert measurement["raw"] == "partial process rows"
    assert measurement["stderr"] == "NVML denied"
    assert measurement["return_code"] == 9


def test_runtime_identity_binds_visibility_token_to_the_matching_physical_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args[0],
            0,
            "0, First Inventory Row, 1.2, GPU-first-row, 0, 15360, N/A, N/A\n"
            f"1, Observed GPU, 1.2, {TEST_SELECTED_GPU_UUID}, 120, 15360, 37, 22\n",
            "",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            device_count=lambda: 1,
            get_device_name=lambda _: "Observed GPU",
            get_device_properties=lambda _: SimpleNamespace(
                total_memory=15_636_037_632, major=8, minor=6
            ),
            mem_get_info=lambda _: (15_526_068_224, 15_636_037_632),
        ),
        __version__="test-torch",
        version=SimpleNamespace(cuda="test-cuda"),
        backends=SimpleNamespace(cudnn=SimpleNamespace(version=lambda: 90100)),
    )
    identity = runtime_identity(fake_torch)
    assert identity["selected_gpu_name"] == "Observed GPU"
    assert identity["torch_version"] == "test-torch"
    assert identity["cuda_runtime_version"] == "test-cuda"
    inventory = identity["physical_gpu_inventory_and_utilization"]
    assert inventory["status"] == "ok"
    assert len(inventory["rows"]) == 2
    assert inventory["rows"][0]["utilization_gpu_percent"] is None
    assert inventory["rows"][1]["utilization_gpu_percent"] == 37
    assert identity["selected_physical_gpu_binding_status"] == "bound_from_cuda_visibility_token"
    assert identity["selected_physical_gpu_uuid"] == TEST_SELECTED_GPU_UUID
    assert identity["selected_physical_gpu_index"] == 1
    assert identity["selected_physical_gpu_memory_used_mib"] == 120
    assert identity["selected_physical_gpu_memory_total_mib"] == 15360
    assert identity["selected_cuda_free_bytes"] == 15_526_068_224
    require_physical_gpu_inventory(identity)


def test_live_resource_snapshot_preserves_startup_uuid_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(profile_worker, "_read_meminfo", lambda: (12 * 1024**3, 16 * 1024**3))
    monkeypatch.setattr(profile_worker, "_cgroup_headroom", lambda: (12 * 1024**3, 16 * 1024**3))
    fake_cuda = SimpleNamespace(
        is_available=lambda: True,
        get_device_properties=lambda _index: SimpleNamespace(total_memory=15_636_037_632),
        mem_get_info=lambda _index: (15_526_068_224, 15_636_037_632),
        memory_allocated=lambda _index: 100_000_000,
        memory_reserved=lambda _index: 200_000_000,
        max_memory_allocated=lambda _index: 100_000_000,
        max_memory_reserved=lambda _index: 200_000_000,
    )

    def inventory(**_kwargs: object) -> dict[str, object]:
        return {
            "status": "ok",
            "raw_output_sha256": "d" * 64,
            "rows": [
                {
                    "uuid": "GPU-other",
                    "memory_total_mib": 15_360,
                    "memory_used_mib": 0,
                },
                {
                    "uuid": TEST_SELECTED_GPU_UUID,
                    "memory_total_mib": 15_360,
                    "memory_used_mib": 120,
                },
            ],
        }

    monkeypatch.setattr(profile_worker, "nvidia_smi_gpu_sample", inventory)
    snapshot = resource_snapshot(
        SimpleNamespace(cuda=fake_cuda),
        tmp_path,
        expected_physical_gpu_uuid=TEST_SELECTED_GPU_UUID,
    )
    assert snapshot["gpu_selected_physical_gpu_uuid"] == TEST_SELECTED_GPU_UUID
    assert snapshot["gpu_bound_physical_gpu_uuid"] == TEST_SELECTED_GPU_UUID
    enforce_resource_caps(snapshot, elapsed_s=10)

    changed_inventory = {
        "status": "ok",
        "raw_output_sha256": "e" * 64,
        "rows": [
            {
                "uuid": "GPU-other",
                "memory_total_mib": 15_360,
                "memory_used_mib": 120,
            }
        ],
    }
    monkeypatch.setattr(
        profile_worker, "nvidia_smi_gpu_sample", lambda **_kwargs: changed_inventory
    )
    changed = resource_snapshot(
        SimpleNamespace(cuda=fake_cuda),
        tmp_path,
        expected_physical_gpu_uuid=TEST_SELECTED_GPU_UUID,
    )
    assert changed["gpu_bound_physical_gpu_uuid"] == TEST_SELECTED_GPU_UUID
    assert changed["gpu_resource_query_status"] == "invalid_memory_domains"
    with pytest.raises(ProfileAbort, match="device-memory measurement"):
        enforce_resource_caps(changed, elapsed_s=11)


def test_gpu_parser_keeps_n_a_utilization_null_and_physical_rows() -> None:
    rows = parse_nvidia_smi_gpu_rows(
        "0, GPU A, 1.2, GPU-A, 100, 15360, 44, 31\n1, GPU B, 1.2, GPU-B, 0, 15360, N/A, N/A\n"
    )
    assert [row["index"] for row in rows] == [0, 1]
    assert rows[0]["utilization_gpu_percent"] == 44
    assert rows[1]["utilization_gpu_percent"] is None
    assert len(rows) == 2


def test_gpu_query_failure_is_diagnostic_and_never_fabricates_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("nvidia-smi not found")

    monkeypatch.setattr(subprocess, "run", unavailable)
    sample = nvidia_smi_gpu_sample()
    assert sample["status"] == "query_error"
    assert sample["rows"] == []
    assert "FileNotFoundError" in sample["diagnostic"]
    assert sample["raw_output"] == ""
    assert sample["raw_stderr"] == ""
    with pytest.raises(ProfileAbort, match="physical GPU inventory query"):
        require_physical_gpu_inventory({"physical_gpu_inventory_and_utilization": sample})


def test_gpu_query_nonzero_exit_preserves_raw_failure_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failed(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args[0], 17, "partial stdout", "driver denied")

    monkeypatch.setattr(subprocess, "run", failed)
    sample = nvidia_smi_gpu_sample()
    assert sample["status"] == "query_error"
    assert sample["return_code"] == 17
    assert sample["raw_output"] == "partial stdout"
    assert sample["raw_stderr"] == "driver denied"


def test_gpu_query_bounds_large_raw_text_and_keeps_exact_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_stderr = "x" * (GPU_QUERY_RAW_MAX_BYTES + 17)

    def failed(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args[0], 17, "", raw_stderr)

    monkeypatch.setattr(subprocess, "run", failed)
    sample = nvidia_smi_gpu_sample()
    assert len(sample["raw_stderr"]) == GPU_QUERY_RAW_MAX_BYTES
    assert sample["raw_stderr_byte_length"] == len(raw_stderr)
    assert sample["raw_stderr_sha256"] == hashlib.sha256(raw_stderr.encode()).hexdigest()
    assert sample["raw_stderr_truncated"] is True


def test_sync_guard_persists_exact_failed_snapshot_before_raising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = _safe_resources()
    snapshot["gpu_process_bytes"] = None
    snapshot["gpu_process_query_status"] = "pid_not_reported"
    snapshot["gpu_process_query_raw"] = ""
    snapshot["gpu_process_query_stderr"] = "restricted"
    snapshot["gpu_device_used_upper_bound_bytes"] = 3_000_000_001
    monkeypatch.setattr(
        profile_worker,
        "resource_snapshot",
        lambda *_args, **_kwargs: snapshot,
    )
    guard = LiveGuard(None, tmp_path, tmp_path, time.monotonic())

    with pytest.raises(ProfileAbort, match="device-wide GPU usage upper bound"):
        guard.check()

    observed = json.loads((tmp_path / "resource.json").read_text(encoding="utf-8"))
    abort = json.loads((tmp_path / "resource_abort.json").read_text(encoding="utf-8"))
    assert observed["snapshot"]["gpu_process_query_raw"] == ""
    assert observed["snapshot"]["gpu_device_used_upper_bound_bytes"] == 3_000_000_001
    assert observed["phase"] == "profile_active"
    assert abort["snapshot"] == snapshot
    assert abort["phase"] == "profile_active"
    assert "device-wide GPU usage upper bound" in abort["reason"]


def test_gpu_telemetry_sampler_persists_pre_and_post_samples(
    tmp_path: Path,
) -> None:
    receipt_path = tmp_path / "gpu_telemetry.json"
    sampler = GpuTelemetrySampler(
        receipt_path,
        interval_seconds=5.0,
        max_samples=4,
        sample_fn=lambda: {"status": "query_error", "rows": [], "diagnostic": "offline"},
    )
    sampler.start()
    sampler.stop()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["phase"] == "trainer_profile_gpu_telemetry"
    assert [sample["sample_phase"] for sample in receipt["samples"]] == [
        "before_trainer_main",
        "after_trainer_main",
    ]
    assert all(sample["status"] == "query_error" for sample in receipt["samples"])
    assert all("utilization_gpu_percent" not in sample for sample in receipt["samples"])


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
