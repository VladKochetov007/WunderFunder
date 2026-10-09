"""Profile native TPU BF16 against the hash-bound FP32 device gate."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def prediction_drift(
    reference: np.ndarray,
    candidate: np.ndarray,
    targets: np.ndarray,
    row_mask: np.ndarray,
) -> dict[str, object]:
    """Summarize raw FP32/BF16 prediction differences with fixed label weights."""
    if reference.shape != candidate.shape or reference.shape != targets.shape:
        raise ValueError("prediction and target arrays must share [batch,time,2] shape")
    if row_mask.shape != reference.shape[:-1]:
        raise ValueError("row mask must match prediction rows")
    if reference.shape[-1] != 2:
        raise ValueError("the benchmark expects exactly two prediction targets")
    if not (
        np.isfinite(reference).all()
        and np.isfinite(candidate).all()
        and np.isfinite(targets).all()
    ):
        raise FloatingPointError("prediction drift inputs contain nonfinite values")
    difference = candidate.astype(np.float64) - reference.astype(np.float64)
    clipped_targets = np.clip(targets.astype(np.float64), -2.0, 2.0)
    weights = np.abs(clipped_targets) * row_mask.astype(bool)[..., None]
    denominator = float(np.sum(weights))
    return {
        "mean_absolute_error_by_target": [
            float(np.mean(np.abs(difference[..., channel]))) for channel in range(2)
        ],
        "rmse_by_target": [
            float(np.sqrt(np.mean(np.square(difference[..., channel]))))
            for channel in range(2)
        ],
        "max_absolute_error_by_target": [
            float(np.max(np.abs(difference[..., channel]))) for channel in range(2)
        ],
        "weighted_mean_absolute_error": float(
            np.sum(weights * np.abs(difference)) / max(denominator, 1e-12)
        ),
        "weighted_rmse": float(
            np.sqrt(np.sum(weights * np.square(difference)) / max(denominator, 1e-12))
        ),
        "weight_sum": denominator,
        "rows_with_nonzero_weight": int(np.count_nonzero(np.any(weights > 0, axis=-1))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-state", type=Path, required=True)
    parser.add_argument("--profile-inputs", type=Path, required=True)
    parser.add_argument("--profile-inputs-receipt", type=Path, required=True)
    parser.add_argument("--fp32-receipt", type=Path, required=True)
    parser.add_argument("--onecycle-trace", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    from run_device_parity import validate_profile_receipt
    from tpu_worker import (
        _forward_step,
        _memory_snapshot,
        _read_onecycle_trace,
        _replicate,
        _shard,
        _training_step,
        sha256_file,
    )

    fp32 = json.loads(args.fp32_receipt.read_text())
    if fp32.get("status") != "pass" or fp32.get("initial_state_sha256") != sha256_file(
        args.initial_state
    ):
        raise ValueError("native BF16 requires passing actual FP32 gate for this initializer")
    profile_input_receipt = validate_profile_receipt(
        args.profile_inputs, args.profile_inputs_receipt, args.initial_state
    )
    if not fp32.get("actual_data_profile"):
        raise ValueError("synthetic FP32 profile cannot admit native actual-data profile")
    schedule = _read_onecycle_trace(args.onecycle_trace)
    schedule_sha256 = sha256_file(args.onecycle_trace)
    if fp32.get("onecycle_trace_sha256") != schedule_sha256:
        raise ValueError("native BF16 profile has a different OneCycle trace from its FP32 gate")

    prediction_meta = fp32.get("profile_predictions_fp32")
    if not isinstance(prediction_meta, dict):
        raise ValueError("FP32 gate omitted actual-input prediction fixture")
    prediction_name = Path(str(prediction_meta.get("path", "")))
    if prediction_name.is_absolute() or ".." in prediction_name.parts or not prediction_name.name:
        raise ValueError("FP32 prediction fixture path must be a simple relative name")
    fp32_prediction_path = (args.fp32_receipt.parent / prediction_name).resolve(strict=True)
    if sha256_file(fp32_prediction_path) != prediction_meta.get("sha256"):
        raise ValueError("FP32 prediction fixture differs from its receipt")

    import jax
    import jax.numpy as jnp
    from rnn_native import adamw_init, load_npz_state

    jax.config.update("jax_default_matmul_precision", "highest")
    devices = [device for device in jax.devices() if device.platform == "tpu"]
    if len(devices) != 8:
        raise ValueError("native TPU profile requires eight actual TPU devices")
    args.output_root.mkdir(parents=True, exist_ok=False)
    params = load_npz_state(args.initial_state, 211, 256, 2, 2)
    with np.load(args.profile_inputs, allow_pickle=False) as arrays:
        host_inputs = arrays["inputs"]
        host_targets = arrays["targets"]
        host_mask = arrays["row_mask"].astype(bool)
    if (
        host_inputs.shape != (32, 2000, 211)
        or host_targets.shape != (32, 2000, 2)
        or host_mask.shape != (32, 2000)
        or not np.isfinite(host_inputs).all()
        or not np.isfinite(host_targets).all()
        or np.any(host_mask[:, :99])
    ):
        raise ValueError("native BF16 actual profile fixture schema differs")
    inputs, targets, mask = [
        _shard(value, devices) for value in (host_inputs, host_targets, host_mask)
    ]

    # Use the same actual producer fixture and zero recurrent state for the
    # initial FP32-versus-BF16 prediction comparison.
    prediction_check_start = time.perf_counter()
    fp32_predictions = np.load(fp32_prediction_path, mmap_mode="r", allow_pickle=False)
    if fp32_predictions.shape != (32, 2000, 2) or not np.isfinite(fp32_predictions).all():
        raise ValueError("FP32 prediction fixture has invalid shape or values")
    bf16_params = _replicate(params, devices)
    bf16_state = _replicate(jnp.zeros((2, 4, 256), dtype=jnp.bfloat16), devices)
    bf16_predictions_device, _ = _forward_step("bf16")(
        bf16_params, inputs, bf16_state
    )
    jax.block_until_ready(bf16_predictions_device)
    bf16_predictions = np.asarray(
        jax.device_get(bf16_predictions_device), dtype=np.float32
    ).reshape(32, 2000, 2)
    prediction_check_seconds = time.perf_counter() - prediction_check_start
    prediction_drift_summary = prediction_drift(
        np.asarray(fp32_predictions, dtype=np.float32),
        bf16_predictions,
        host_targets,
        host_mask,
    )

    learning_rate, beta1 = (float(value) for value in schedule[0])
    state = _replicate(jnp.zeros((2, 4, 256), dtype=jnp.bfloat16), devices)
    rows = {}
    for loss in ("wmse", "huber"):
        initial = _replicate(params, devices)
        optimizer = _replicate(adamw_init(params), devices)
        step = _training_step(loss, "bf16")
        samples = []
        for repeat in range(6):
            started = time.perf_counter()
            result = step(
                initial,
                optimizer,
                inputs,
                state,
                targets,
                mask,
                np.full(8, learning_rate, dtype=np.float32),
                np.full(8, beta1, dtype=np.float32),
            )
            jax.block_until_ready(result)
            seconds = time.perf_counter() - started
            if repeat == 0:
                cold = seconds
            else:
                samples.append(seconds)
            if not np.asarray(jax.device_get(result[-1])).all():
                raise FloatingPointError(f"nonfinite native BF16 {loss} discarded update")
        for value in jax.tree_util.tree_leaves((result[0], result[1].first, result[1].second)):
            if value.dtype != jnp.float32:
                raise ValueError("native BF16 master parameters/AdamW accumulators must be FP32")
        scalars = {
            "learning_rate": learning_rate,
            "beta1": beta1,
            "weighted_loss": float(np.asarray(jax.device_get(result[3])).reshape(-1)[0]),
            "unscaled_global_gradient_norm": float(
                np.asarray(jax.device_get(result[4])).reshape(-1)[0]
            ),
            "gradient_clip_scale": float(np.asarray(jax.device_get(result[5])).reshape(-1)[0]),
            "finite": bool(np.asarray(jax.device_get(result[6])).all()),
        }
        fp32_step = fp32["profile"][loss]["first_discarded_update"]
        if fp32_step.get("learning_rate") != learning_rate or fp32_step.get("beta1") != beta1:
            raise ValueError("FP32 and native profiles do not use the same initial OneCycle row")

        # Persist actual updated masters and moments, not merely an initial fixture.
        saved = {
            f"leaf_{index}": np.asarray(jax.device_get(value))
            for index, value in enumerate(jax.tree_util.tree_leaves((result[0], result[1])))
        }
        checkpoint = args.output_root / f"{loss}_discarded_update_state.npz"
        np.savez(checkpoint, **saved)
        with np.load(checkpoint, allow_pickle=False) as reloaded:
            for key, value in saved.items():
                np.testing.assert_array_equal(reloaded[key], value)
                if reloaded[key].dtype != value.dtype:
                    raise ValueError("native updated-state reload changed precision")
        rows[loss] = {
            "compile_plus_first_update_seconds": cold,
            "warm_update_seconds": float(np.median(samples)),
            "warm_update_seconds_samples": samples,
            "warm_update_seconds_max": max(samples),
            "shape": [32, 2000, 211],
            "active_sequences": 32,
            "finite": True,
            "first_discarded_update": scalars,
            "fp32_comparison": {
                "weighted_loss_abs_delta": abs(
                    scalars["weighted_loss"] - fp32_step["weighted_loss"]
                ),
                "unscaled_global_gradient_norm_abs_delta": abs(
                    scalars["unscaled_global_gradient_norm"]
                    - fp32_step["unscaled_global_gradient_norm"]
                ),
                "gradient_clip_scale_abs_delta": abs(
                    scalars["gradient_clip_scale"] - fp32_step["gradient_clip_scale"]
                ),
            },
            "memory_after_full_shape": _memory_snapshot(),
            "updated_master_and_moment_reload": True,
            "discarded_checkpoint_sha256": sha256_file(checkpoint),
        }
    receipt = {
        "status": "pass",
        "compute_precision": "bf16",
        "master_precision": "fp32",
        "optimizer_accumulator_precision": "fp32",
        "loss_scaler": None,
        "profile_updates_discarded": 12,
        "fp32_receipt_sha256": sha256_file(args.fp32_receipt),
        "initial_state_sha256": sha256_file(args.initial_state),
        "profile_inputs_sha256": sha256_file(args.profile_inputs),
        "profile_inputs_receipt_sha256": sha256_file(args.profile_inputs_receipt),
        "onecycle_trace_sha256": schedule_sha256,
        "device_count": len(devices),
        "device_inventory": [
            {
                "id": str(device),
                "platform": device.platform,
                "kind": getattr(device, "device_kind", None),
                "platform_version": getattr(device, "platform_version", None),
            }
            for device in devices
        ],
        "jax_version": jax.__version__,
        "profile": rows,
        "prediction_drift_vs_fp32": {
            **prediction_drift_summary,
            "prediction_check_seconds": prediction_check_seconds,
            "fp32_prediction_sha256": prediction_meta["sha256"],
            "profile_inputs_sha256": profile_input_receipt["profile_inputs_sha256"],
            "sequence_ids": profile_input_receipt["sequence_ids"],
        },
        "synchronization": "host call plus jax.block_until_ready complete update result tree",
    }
    (args.output_root / "native_profile_receipt.json").write_text(
        json.dumps(receipt, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
