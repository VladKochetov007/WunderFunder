"""Separate native precision authority from the FP32 correctness anchor."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tpu_packet_runtime as runtime  # noqa: E402


def admission():
    return {
        "schema": "wnn.loss-pair-tpu.root-admission.v1",
        "source_manifest_sha256": "a" * 64,
        "cache_manifest_sha256": runtime.CACHE_MANIFEST_SHA,
        "cache_key": runtime.CACHE_KEY,
        "gpu_kernel": runtime.GPU_KERNEL,
        "cpu_kernel": runtime.CPU_KERNEL,
        "push_allowed": True,
        "conditional_fit_allowed": False,
        "gpu_training_contract_sha256": "b" * 64,
        "runtime_seconds": 3300,
        "tpu_compute_precision": "fp32",
        "native_bf16_fit_allowed": False,
    }


def test_fp32_profile_authority_does_not_admit_native_precision():
    record = admission()
    runtime.check_admission(record, "a" * 64)
    record["tpu_compute_precision"] = "bf16"
    with pytest.raises(ValueError, match="separate root admission"):
        runtime.check_admission(record, "a" * 64)
    record["native_bf16_fit_allowed"] = True
    runtime.check_admission(record, "a" * 64)


def test_unknown_precision_and_wrong_producer_fail_closed():
    record = admission()
    record["tpu_compute_precision"] = "fp16"
    with pytest.raises(ValueError, match="unsupported"):
        runtime.check_admission(record, "a" * 64)
    record["tpu_compute_precision"] = "fp32"
    record["gpu_kernel"] = "vladkochetov07/wnn-loss-pair-gpu-20261009-v2"
    with pytest.raises(ValueError, match="identity differs"):
        runtime.check_admission(record, "a" * 64)
