from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / "specs/README.md").is_file()
)
PACKET_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(PACKET_ROOT / "payload_root/scripts"))

from h230_profile_then_fit import evaluate_profile  # noqa: E402


def _valid_case() -> tuple[dict[str, Any], dict[str, Any], dict[str, int]]:
    expected = {
        "device_name_substring": "T4",
        "device_capability": [7, 5],
        "cuda_bf16_supported": False,
        "orthogonalization_policy": "native_bf16_if_supported_else_fp32_v1",
        "orthogonalization_dtype": "float32",
        "muon_scope": "hidden",
        "muon_gatewise": "none",
        "muon_lr_multiplier": 3.0,
        "model_precision": "fp16",
        "distill_cache_variant": "source",
        "train_sequences": 10606,
        "valid_sequences": 1873,
        "profile_batch_count": 4,
        "gradient_observations": 80,
        "batches_per_epoch": 779,
        "epochs": 4,
        "model_shape": {
            "raw_feature_width": 112,
            "event_feature_width": 99,
            "model_input_width": 211,
            "model_output_width": 23,
        },
        "profile_args": {
            "name": "profile-seed9",
            "seed": 9,
            "batch": 16,
            "chunk": 1000,
            "epochs": 4,
            "profile_warmup": 1,
            "profile_steps": 3,
            "save_training_state": False,
        },
    }
    plan = {
        "expected": expected,
        "runtime_gate": {
            "allocator_memory_bytes": 4_300_000_000,
            "worker_timeout_seconds": 9000,
            "timing_margin_multiplier": 1.15,
            "minimum_fit_setup_reserve_seconds": 120,
            "extra_fit_contract_reserve_seconds": 180,
            "checkpoint_reserve_seconds": 300,
        },
    }
    args = {
        **expected["profile_args"],
        "max_batches": 4,
        "device": "cuda",
        "precision": "fp16",
        "optim": "muon",
        "sched": "onecycle",
        "muon_scope": "hidden",
        "muon_gatewise": "none",
        "muon_lr_mult": 3.0,
        "distill_cache_variant": "source",
        "distill_cache_all_data": True,
        "include_heldout_valid": True,
        "no_eval_during_fit": True,
    }
    records = {
        "muon_runtime": {
            "device_type": "cuda",
            "device_capability": [7, 5],
            "cuda_bf16_supported": False,
            "policy": "native_bf16_if_supported_else_fp32_v1",
            "selected_dtype": "float32",
            "scope": "hidden",
            "gatewise": "none",
        },
        "grad_stats": {
            "steps": 80,
            "nonfinite_steps": 0,
            "total_q05_q50_q95": [1.0, 2.0, 3.0],
            "family_median": {"rnn": 0.5},
            "family_sq_share_mean": {"rnn": 0.5},
        },
        "profile": {"step_wall": 1000.0, "seq_per_s": 16.0, "gpu_total": 1000.0},
        "json_lines": [
            args,
            {
                "source_train_sequences": 10606,
                "eligible_valid_train_sequences": 1873,
                "used_valid_train_sequences": 1873,
                "eval_sequences": 0,
            },
        ],
    }
    shape = dict(expected["model_shape"])
    return plan, records, shape


def _evaluate(
    plan: dict[str, Any], records: dict[str, Any], shape: dict[str, int], **overrides: Any
) -> dict[str, Any]:
    parameters = {
        "device_name": "Tesla T4",
        "derived_shape": shape,
        "allocator_peak_bytes": 1_000_000_000,
        "allocator_reserved_peak_bytes": 2_000_000_000,
        "profile_wall_seconds": 12.0,
        "profile_loop_seconds": 4.0,
        "raw_preflight_seconds": 4.0,
    }
    parameters.update(overrides)
    return evaluate_profile(plan, records, **parameters)


def test_profile_gate_accepts_exact_finite_t4_profile() -> None:
    plan, records, shape = _valid_case()
    result = _evaluate(plan, records, shape)
    assert result["status"] == "passed"
    assert result["projected_profile_plus_fit_seconds"] < 9000


def test_profile_gate_rejects_device_dtype_shape_and_gradient_drift() -> None:
    cases = ("device", "dtype", "shape", "nonfinite", "missing_stats")
    for case in cases:
        plan, records, shape = _valid_case()
        overrides: dict[str, Any] = {}
        if case == "device":
            overrides["device_name"] = "A10G"
        elif case == "dtype":
            records["muon_runtime"]["selected_dtype"] = "bfloat16"
        elif case == "shape":
            shape["model_output_width"] = 31
        elif case == "nonfinite":
            records["grad_stats"]["family_median"]["rnn"] = float("nan")
        elif case == "missing_stats":
            del records["grad_stats"]["family_sq_share_mean"]
        result = _evaluate(plan, records, shape, **overrides)
        assert result["status"] == "failed_closed", case


def test_profile_gate_rejects_projection_over_worker_timeout() -> None:
    plan, records, shape = _valid_case()
    result = _evaluate(plan, records, shape, profile_loop_seconds=20.0)
    assert "projected_profile_plus_fit_exceeds_worker_timeout" in result["failures"]
