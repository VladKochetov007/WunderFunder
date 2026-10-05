"""Unlaunched actual-shape E485 CUDA parity/resource probe."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
E470_DIR = ROOT / "research/campaign_2026-10-05/activation_optimizer"
E474_DIR = ROOT / "research/campaign_2026-10-05/activation_placement"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(E470_DIR))
sys.path.insert(0, str(E474_DIR))

import fit_activation_optimizer as e470  # noqa: E402
import fit_activation_placement as e474  # noqa: E402
import fit_duration as e485  # noqa: E402

from wnn.nn import RNNModel  # noqa: E402
from wnn.resources import configure_torch  # noqa: E402

ALLOCATOR_MAX_BYTES = 2 * 1024**3
OBSERVED_STOP_BYTES = 3_000_000_000
OUTPUT_ATOL = 2e-5
OUTPUT_RTOL = 2e-4
GRAD_ATOL = 5e-5
GRAD_RTOL = 5e-4
UPDATE_ATOL = 1e-4
UPDATE_RTOL = 1e-3


def require_finite(tensor: torch.Tensor, label: str) -> None:
    """Reject any nonfinite raw value before loss masking or clipping."""
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"nonfinite {label}")


def check_device_budget() -> dict[str, int]:
    """Fail closed before allocation if the live GPU is already over budget."""
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    used_bytes = total_bytes - free_bytes
    if used_bytes >= OBSERVED_STOP_BYTES:
        raise RuntimeError(f"GPU already uses {used_bytes} bytes; stop cap is 3 GB")
    if torch.cuda.memory_allocated() > ALLOCATOR_MAX_BYTES:
        raise RuntimeError("current process already exceeds the 2 GiB allocator cap")
    return {
        "free_before_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
        "observed_used_before_bytes": int(used_bytes),
    }


def check_allocator_and_observed_memory() -> dict[str, int]:
    """Enforce per-process allocator and physical-device stop thresholds."""
    torch.cuda.synchronize()
    allocated = int(torch.cuda.max_memory_allocated())
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    observed = int(total_bytes - free_bytes)
    if allocated > ALLOCATOR_MAX_BYTES:
        raise RuntimeError(f"allocator peak {allocated} exceeds 2 GiB")
    if observed > OBSERVED_STOP_BYTES:
        raise RuntimeError(f"observed GPU use {observed} exceeds 3 GB")
    return {
        "peak_allocated_bytes": allocated,
        "observed_used_after_bytes": observed,
        "total_bytes": int(total_bytes),
    }


def initial_bytes_match(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> bool:
    """Compare each live device tensor with its original CPU initialization."""
    current = model.state_dict()
    if current.keys() != state.keys():
        raise ValueError("device model state keys differ from the registered initialization")
    for name, value in current.items():
        live_cpu = value.detach().cpu().contiguous()
        initial_cpu = state[name].detach().cpu().contiguous()
        if (
            live_cpu.dtype != initial_cpu.dtype
            or live_cpu.shape != initial_cpu.shape
            or live_cpu.numpy().tobytes() != initial_cpu.numpy().tobytes()
        ):
            raise ValueError(f"CPU/device initialization bytes changed for {name}")
    return True


def run_native_parity(
    seed: int,
    initial: dict[str, torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
) -> dict:
    """Compare train-mode custom alpha-zero to native GRU through one clipped step."""
    native = RNNModel(42, hidden=e485.HIDDEN, layers=2, dropout=0.0, n_out=2)
    native.load_state_dict(initial, strict=True)
    custom = e485.initialize_arm(0.0, initial)
    native.to(device="cuda", dtype=torch.float32).train()
    custom.to(device="cuda", dtype=torch.float32).train()
    native_bytes = initial_bytes_match(native, initial)
    custom_bytes = initial_bytes_match(custom, initial)
    x_native = x.detach().clone().requires_grad_(True)
    x_custom = x.detach().clone().requires_grad_(True)
    started = time.monotonic()
    prediction_native, state_native = native(x_native)
    prediction_custom, state_custom = custom(x_custom)
    require_finite(prediction_native, "native parity prediction")
    require_finite(prediction_custom, "custom parity prediction")
    require_finite(state_native, "native parity state")
    require_finite(state_custom, "custom parity state")
    if state_native.shape != (2, 4, e485.HIDDEN) or state_custom.shape != state_native.shape:
        raise ValueError("parity state must have shape [2,4,128]")
    torch.testing.assert_close(
        prediction_custom, prediction_native, atol=OUTPUT_ATOL, rtol=OUTPUT_RTOL
    )
    torch.testing.assert_close(state_custom, state_native, atol=OUTPUT_ATOL, rtol=OUTPUT_RTOL)
    loss_native = e470.loss_for_batch(prediction_native, y, mask)
    loss_custom = e470.loss_for_batch(prediction_custom, y, mask)
    require_finite(loss_native, "native parity loss")
    require_finite(loss_custom, "custom parity loss")
    loss_native.backward()
    loss_custom.backward()
    torch.testing.assert_close(x_custom.grad, x_native.grad, atol=GRAD_ATOL, rtol=GRAD_RTOL)
    native_parameters = dict(native.named_parameters())
    custom_parameters = dict(custom.named_parameters())
    if native_parameters.keys() != custom_parameters.keys():
        raise ValueError("native and custom model parameter names differ")
    for name in native_parameters:
        if native_parameters[name].grad is None or custom_parameters[name].grad is None:
            raise ValueError(f"missing model gradient at {name}")
        require_finite(native_parameters[name].grad, f"native gradient {name}")
        require_finite(custom_parameters[name].grad, f"custom gradient {name}")
        torch.testing.assert_close(
            custom_parameters[name].grad,
            native_parameters[name].grad,
            atol=GRAD_ATOL,
            rtol=GRAD_RTOL,
            msg=name,
        )
    optimizer_native = e485.make_optimizer(native)
    optimizer_custom = e485.make_optimizer(custom)
    clipped_native, _ = e485.optimizer_update(
        native, optimizer_native, 0, 0, f"native-parity/{seed}"
    )
    clipped_custom, _ = e485.optimizer_update(
        custom, optimizer_custom, 0, 0, f"custom-parity/{seed}"
    )
    if clipped_native != clipped_custom:
        raise ValueError("native/custom parity clipped status differs")
    for name in native_parameters:
        torch.testing.assert_close(
            custom_parameters[name],
            native_parameters[name],
            atol=UPDATE_ATOL,
            rtol=UPDATE_RTOL,
            msg=name,
        )
    return {
        "seed": seed,
        "mode": "train",
        "state_shape": list(state_custom.shape),
        "cpu_initial_bytes_match_native_and_custom": native_bytes and custom_bytes,
        "loss_native": float(loss_native.detach().cpu()),
        "loss_custom": float(loss_custom.detach().cpu()),
        "clipped_step_status": clipped_native,
        "elapsed_seconds": time.monotonic() - started,
        "tolerances": {
            "output_state": {"atol": OUTPUT_ATOL, "rtol": OUTPUT_RTOL},
            "input_parameter_gradients": {"atol": GRAD_ATOL, "rtol": GRAD_RTOL},
            "post_clipped_step_parameters": {"atol": UPDATE_ATOL, "rtol": UPDATE_RTOL},
        },
    }


def one_real_fit_update(
    seed: int,
    alpha: float,
    initial: dict[str, torch.Tensor],
    x: np.ndarray,
    y: np.ndarray,
    need: np.ndarray,
    ids: list[int],
) -> dict:
    """Time one actual fit-only B4×256 update for one fixed recipe."""
    model = e485.initialize_arm(alpha, initial).to(device="cuda", dtype=torch.float32).train()
    if sum(parameter.numel() for parameter in model.parameters()) != e485.EXPECTED_PARAMETERS:
        raise ValueError("GPU probe candidate parameter count changed")
    initial_bytes_match(model, initial)
    optimizer = e485.make_optimizer(model)
    xx = torch.from_numpy(x[ids, : e485.CHUNK_ROWS]).to(device="cuda", dtype=torch.float32)
    yy = torch.from_numpy(y[ids, : e485.CHUNK_ROWS]).to(device="cuda", dtype=torch.float32)
    mask = torch.from_numpy(need[ids, : e485.CHUNK_ROWS]).to(device="cuda")
    require_finite(xx, "raw fit features")
    require_finite(yy, "raw fit labels")
    before = [parameter.detach().clone() for parameter in model.parameters()]
    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()
    prediction, state = model(xx)
    require_finite(prediction, "fit prediction")
    require_finite(state, "fit recurrent state")
    if state.shape != (2, 4, e485.HIDDEN):
        raise ValueError("GPU fit state must have shape [2,4,128]")
    loss = e470.loss_for_batch(prediction, yy, mask)
    require_finite(loss, "fit loss")
    loss.backward()
    clipped, telemetry = e485.optimizer_update(
        model, optimizer, 0, 0, f"gpu-probe/{seed}/alpha={alpha}"
    )
    changed = any(
        not torch.equal(old, new) for old, new in zip(before, model.parameters(), strict=True)
    )
    if not changed:
        raise ValueError(f"probe update did not change parameters: {seed}/alpha={alpha}")
    return {
        "seed": seed,
        "alpha": alpha,
        "fit_row_groups": ids,
        "state_shape": list(state.shape),
        "loss": float(loss.detach().cpu()),
        "clip_applied": clipped,
        "parameters_changed": changed,
        "parameters_finite": True,
        "update_seconds": time.monotonic() - started,
        "telemetry_step0": telemetry,
    }


def main() -> None:
    """Probe native parity and all six real-data update recipes; never score tests."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite probe output: {args.output}")
    if not torch.cuda.is_available():
        raise RuntimeError("E485 probe requires CUDA")
    torch.set_num_threads(1)
    configure_torch()
    budget_before = check_device_budget()
    torch.cuda.reset_peak_memory_stats()
    hashes = e485.parent_source_hashes()
    initial_states = e485.make_initial_states()
    cohort, x, y, need, _, nodes = e470.load_cohort(include_test=False)
    if len(nodes) != 24 or any(node["role"] != "fit" for node in nodes):
        raise ValueError("GPU probe may load fit rows only")
    if x.shape != (24, e485.ROWS_PER_GROUP, e485.FEATURES):
        raise ValueError(f"unexpected fit data shape: {x.shape}")
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("nonfinite fit-only data")
    parity: list[dict] = []
    updates: list[dict] = []
    with e474.full_ieee_float32():
        active_precision = e474.precision_metadata()
        for seed in e485.SEEDS:
            ids = e485.canonical_epoch_order(seed, 0)[: e485.BATCH_SIZE].tolist()
            x_gpu = torch.from_numpy(x[ids, : e485.CHUNK_ROWS]).to(
                device="cuda", dtype=torch.float32
            )
            y_gpu = torch.from_numpy(y[ids, : e485.CHUNK_ROWS]).to(
                device="cuda", dtype=torch.float32
            )
            mask_gpu = torch.from_numpy(need[ids, : e485.CHUNK_ROWS]).to(device="cuda")
            require_finite(x_gpu, "parity raw fit features")
            require_finite(y_gpu, "parity raw fit labels")
            parity.append(run_native_parity(seed, initial_states[seed], x_gpu, y_gpu, mask_gpu))
        for seed in e485.SEEDS:
            ids = e485.canonical_epoch_order(seed, 0)[: e485.BATCH_SIZE].tolist()
            for alpha in e485.ALPHAS:
                updates.append(
                    one_real_fit_update(seed, alpha, initial_states[seed], x, y, need, ids)
                )
                check_allocator_and_observed_memory()
    if len(updates) != 6 or not all(item["parameters_changed"] for item in updates):
        raise ValueError("all six fit-only probe recipes must make a finite parameter update")
    memory = check_allocator_and_observed_memory()
    timings = [item["update_seconds"] for item in updates]
    conservative_step_seconds = float(max(timings))
    projected_full_matrix_seconds = (
        conservative_step_seconds * len(updates) * e485.UPDATES_PER_TRAJECTORY * 1.25
    )
    args.output.mkdir(parents=True)
    result = {
        "kind": "E485 actual H128 B4x256 CUDA probe",
        "status": "passed",
        "fit_only": True,
        "test_rows_loaded": False,
        "cohort_sha256": e474.file_hash(e470.COHORT_PATH),
        "dataset_sha256": e474.DATASET_SHA256,
        "parent_source_sha256": hashes,
        "source_sha256": {
            name: e474.file_hash(HERE / name)
            for name in ("fit_duration.py", "tests.py", "gpu_probe.py")
        },
        "active_precision_inside_context": active_precision,
        "optimizer_cuda_orthogonalization": "existing wnn.optim BF16 orthogonalizer",
        "resource_limits": {
            "allocator_cap_bytes": ALLOCATOR_MAX_BYTES,
            "observed_stop_bytes": OBSERVED_STOP_BYTES,
            "before": budget_before,
            "after": memory,
        },
        "native_alpha_zero_parity": parity,
        "six_actual_fit_updates": updates,
        "timing_projection": {
            "update_seconds_by_arm": timings,
            "conservative_update_seconds": conservative_step_seconds,
            "projected_matrix_seconds": projected_full_matrix_seconds,
            "method": "maximum measured warm B4x256 step × 6 arms × 1,896 updates × 1.25 margin; excludes initialization, checkpoint, and scoring time",
            "admission_cap_seconds": 7_200,
            "projection_fits_cap": projected_full_matrix_seconds <= 7_200,
        },
        "fit_role_order": cohort["cohort"],
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
