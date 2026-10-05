"""E487 one-batch CUDA precision diagnostic; not a training run."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from precision_replay import (
    ALLOCATOR_MAX_BYTES,
    CUDA_FP32_CPU_ATOL,
    CUDA_FP32_CPU_RTOL,
    DEVICE_STOP_BYTES,
    SEED,
    _jsonable_shared_control,
    capture_actual_fit_batch,
    capture_gradients,
    check_common_initialization,
    e470,
    e474,
    e485,
    orthogonalize_fp32,
    replay_crossed_gradients,
    shared_gradient_control,
    source_hashes,
    state_digest,
)

from wnn.resources import configure_torch


def memory_check() -> dict[str, int]:
    """Synchronize and enforce allocator/whole-device ceilings."""
    torch.cuda.synchronize()
    allocated_peak = int(torch.cuda.max_memory_allocated())
    free, total = torch.cuda.mem_get_info()
    observed = int(total - free)
    if allocated_peak > ALLOCATOR_MAX_BYTES:
        raise RuntimeError(f"allocator peak {allocated_peak} exceeds 2 GiB")
    if observed > DEVICE_STOP_BYTES:
        raise RuntimeError(f"observed GPU use {observed} exceeds 3 GB")
    return {
        "allocator_peak_bytes": allocated_peak,
        "observed_gpu_bytes": observed,
        "free_gpu_bytes": int(free),
        "total_gpu_bytes": int(total),
    }


def fp32_cpu_cuda_check() -> dict:
    """Compare deterministic synthetic FP32 NS results elementwise."""
    generator = torch.Generator(device="cpu").manual_seed(487_271)
    matrix = torch.randn((3, 128, 128), generator=generator, dtype=torch.float32)
    cpu = orthogonalize_fp32(matrix)
    cuda = orthogonalize_fp32(matrix.to(device="cuda"))
    torch.cuda.synchronize()
    difference = (cuda.cpu() - cpu).abs()
    bound = CUDA_FP32_CPU_ATOL + CUDA_FP32_CPU_RTOL * cpu.abs()
    maximum = float(difference.max())
    relative_rms = float(
        difference.double().square().mean().sqrt() / cpu.double().square().mean().sqrt()
    )
    passed = bool((difference <= bound).all())
    if not passed:
        raise ValueError(
            "FP32 CPU/CUDA orthogonalizer parity failed: "
            f"max_abs={maximum} relative_rms={relative_rms}"
        )
    return {
        "shape": list(matrix.shape),
        "max_abs_difference": maximum,
        "relative_rms_difference": relative_rms,
        "elementwise_atol": CUDA_FP32_CPU_ATOL,
        "elementwise_rtol": CUDA_FP32_CPU_RTOL,
        "passed": passed,
    }


def save_delta_arrays(path: Path, crossed: dict, control: dict) -> dict:
    """Persist all crossed and shared-gradient update vectors without pickle."""
    arrays = {}
    names = None
    for cell_name, cell in crossed["crossed_replays"].items():
        if names is None:
            names = sorted(cell["deltas"])
        elif names != sorted(cell["deltas"]):
            raise ValueError("crossed replay delta name inventory differs")
        for parameter_name, delta in cell["deltas"].items():
            key = f"{cell_name}__{parameter_name}"
            arrays[key] = delta.detach().cpu().numpy()
    shared_cell_count = 0
    for precision, result in control.items():
        for replay_name, cell in result["replays"].items():
            if cell is None:
                continue
            shared_cell_count += 1
            if sorted(cell["deltas"]) != names:
                raise ValueError("shared-gradient delta name inventory differs")
            for parameter_name, delta in cell["deltas"].items():
                key = f"shared_{precision}_{replay_name}__{parameter_name}"
                arrays[key] = delta.detach().cpu().numpy()
    expected_cells = 9  # 4 crossed, 4 cloned shared-gradient, 1 frozen E474.
    if names is None or shared_cell_count != 5 or len(arrays) != expected_cells * len(names):
        raise ValueError("incomplete crossed-replay delta evidence")
    arrays["parameter_names"] = np.asarray(names)
    np.savez_compressed(path, **arrays)
    return {
        "parameter_names": names,
        "crossed_replay_cells": 4,
        "shared_gradient_replay_cells": shared_cell_count,
        "array_count": len(arrays) - 1,
    }


def main() -> None:
    """Capture fit-only gradients and replay all four precision/source cells."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite E487 probe output: {args.output}")
    os.environ["WNN_GPU_MEMORY_BYTES"] = str(ALLOCATOR_MAX_BYTES)
    if not torch.cuda.is_available():
        raise RuntimeError("E487 diagnostic requires CUDA")
    torch.set_num_threads(1)
    configure_torch()
    free, total = torch.cuda.mem_get_info()
    used_before = int(total - free)
    if used_before > DEVICE_STOP_BYTES:
        raise RuntimeError(f"GPU already uses {used_before} bytes; over stop cap")
    torch.cuda.reset_peak_memory_stats()
    memory_before = memory_check()
    batch, _x, _y, _need, _nodes = capture_actual_fit_batch()
    initial = batch["initial"]
    native, custom = batch["native"], batch["custom"]
    native.to(device="cuda", dtype=torch.float32).train()
    custom.to(device="cuda", dtype=torch.float32).train()
    check_common_initialization(native, initial)
    check_common_initialization(custom, initial)
    memory_after_device_models = memory_check()
    x = batch["feature_batch"].to(device="cuda", dtype=torch.float32)
    y = batch["label_batch"].to(device="cuda", dtype=torch.float32)
    mask = batch["need_mask"].to(device="cuda")
    if not (torch.isfinite(x).all() and torch.isfinite(y).all()):
        raise ValueError("nonfinite raw fit-only probe batch")
    started = time.monotonic()
    with e474.full_ieee_float32():
        active_precision = e474.precision_metadata()
        fp32_reference = fp32_cpu_cuda_check()
        memory_after_fp32_control = memory_check()
        native_capture = capture_gradients(native, x, y, mask, "native")
        custom_capture = capture_gradients(custom, x, y, mask, "custom")
        check_common_initialization(native, initial)
        check_common_initialization(custom, initial)
        forward_receipt = {
            "max_abs_output": float(
                (native_capture["prediction"] - custom_capture["prediction"]).abs().max()
            ),
            "max_abs_state": float(
                (native_capture["final_state"] - custom_capture["final_state"]).abs().max()
            ),
            "max_abs_input_gradient": float(
                (native_capture["input_gradient"] - custom_capture["input_gradient"]).abs().max()
            ),
            "max_abs_parameter_gradient": max(
                float(
                    (native_capture["raw_gradients"][name] - custom_capture["raw_gradients"][name])
                    .abs()
                    .max()
                )
                for name in native_capture["raw_gradients"]
            ),
            "loss_native": native_capture["loss"],
            "loss_custom": custom_capture["loss"],
            "tolerances": {
                "output_state_atol": 2e-5,
                "output_state_rtol": 2e-4,
                "input_parameter_gradient_atol": 5e-5,
                "input_parameter_gradient_rtol": 5e-4,
            },
        }
        torch.testing.assert_close(
            custom_capture["prediction"], native_capture["prediction"], atol=2e-5, rtol=2e-4
        )
        torch.testing.assert_close(
            custom_capture["final_state"], native_capture["final_state"], atol=2e-5, rtol=2e-4
        )
        torch.testing.assert_close(
            custom_capture["input_gradient"],
            native_capture["input_gradient"],
            atol=5e-5,
            rtol=5e-4,
        )
        for name in native_capture["raw_gradients"]:
            torch.testing.assert_close(
                custom_capture["raw_gradients"][name],
                native_capture["raw_gradients"][name],
                atol=5e-5,
                rtol=5e-4,
                msg=name,
            )
        memory_after_gradient_capture = memory_check()
        native.zero_grad(set_to_none=True)
        custom.zero_grad(set_to_none=True)
        del x, y, mask
        captured = {"native": native_capture, "custom": custom_capture}
        control = shared_gradient_control(
            initial,
            torch.device("cuda"),
            before_replay=memory_check,
            after_replay=memory_check,
        )
        crossed = replay_crossed_gradients(
            initial,
            captured,
            torch.device("cuda"),
            before_replay=memory_check,
            after_replay=memory_check,
        )
        memory_after_replays = memory_check()
    args.output.mkdir(parents=True)
    delta_path = args.output / "replay_deltas.npz"
    delta_inventory = save_delta_arrays(delta_path, crossed, control)
    source_identity = source_hashes()
    strict_precision_pass = all(
        value["passed"] for value in crossed["native_custom_strict_gates"].values()
    )
    result = {
        "kind": "E487 one-batch CUDA Muon orthogonalization precision diagnostic",
        "status": (
            "diagnostic_complete"
            if strict_precision_pass
            else "diagnostic_complete_gate_failures_recorded"
        ),
        "fit_only": True,
        "test_rows_loaded": False,
        "seed": SEED,
        "fit_row_indices": batch["row_indices"],
        "fit_row_groups": batch["fit_row_groups"],
        "raw_feature_validation": batch["raw_feature_validation"],
        "cohort_sha256": e474.file_hash(e470.COHORT_PATH),
        "dataset_sha256": e474.DATASET_SHA256,
        "initial_state_sha256": state_digest(initial),
        "initialization_bytes_match_after_device_transfer": True,
        "parent_source_sha256": e485.parent_source_hashes(),
        "source_sha256": source_identity,
        "active_precision_inside_full_ieee_context": active_precision,
        "shared_gradient_control": _jsonable_shared_control(control),
        "fp32_orthogonalizer_cpu_cuda_control": fp32_reference,
        "memory_after_fp32_control": memory_after_fp32_control,
        "native_custom_forward_gradient_parity": forward_receipt,
        "gradient_capture": {
            source: {
                "loss": capture["loss"],
                "raw_gradient_norm": capture["raw_gradient_norm"],
                "raw_gradient_sha256": capture["raw_gradient_sha256"],
                "input_gradient_sha256": capture["input_gradient_sha256"],
            }
            for source, capture in captured.items()
        },
        "crossed_replay_arithmetic": {
            "cell_receipts": {
                key: {
                    name: value
                    for name, value in cell.items()
                    if name not in {"deltas", "updated_state"}
                }
                for key, cell in crossed["crossed_replays"].items()
            },
            "strict_native_custom_update_gates": crossed["native_custom_strict_gates"],
            "bf16_fp32_delta_sensitivity": crossed["bf16_fp32_delta_sensitivity"],
            "all_crossed_finite": crossed["all_crossed_finite"],
            "gate_failures_recorded_without_aborting": crossed[
                "expected_gate_failures_are_recorded_not_raised"
            ],
            "delta_artifact": delta_path.name,
            "delta_artifact_sha256": e474.file_hash(delta_path),
            "delta_artifact_bytes": delta_path.stat().st_size,
            "delta_arrays": delta_inventory,
        },
        "resources": {
            "allocator_limit_bytes": ALLOCATOR_MAX_BYTES,
            "observed_gpu_stop_bytes": DEVICE_STOP_BYTES,
            "before": memory_before,
            "after_device_models": memory_after_device_models,
            "after": memory_after_replays,
        },
        "memory_after_gradient_capture": memory_after_gradient_capture,
        "elapsed_seconds": time.monotonic() - started,
        "interpretation_scope": "single zero-momentum update diagnostic; no trained optimizer history or predictive claim",
        "strict_precision_pass": strict_precision_pass,
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
