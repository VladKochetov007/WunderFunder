"""Profile native TPU BF16 only after a hash-bound actual FP32 gate passes."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-state", type=Path, required=True)
    parser.add_argument("--profile-inputs", type=Path, required=True)
    parser.add_argument("--profile-inputs-receipt", type=Path, required=True)
    parser.add_argument("--fp32-receipt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    from run_device_parity import validate_profile_receipt
    from tpu_worker import _memory_snapshot, _replicate, _shard, _training_step, sha256_file

    fp32 = json.loads(args.fp32_receipt.read_text())
    if fp32.get("status") != "pass" or fp32.get("initial_state_sha256") != sha256_file(
        args.initial_state
    ):
        raise ValueError("native BF16 requires passing actual FP32 gate for this initializer")
    validate_profile_receipt(args.profile_inputs, args.profile_inputs_receipt, args.initial_state)
    if not fp32.get("actual_data_profile"):
        raise ValueError("synthetic FP32 profile cannot admit native actual-data profile")

    import jax
    import jax.numpy as jnp
    import numpy as np
    from rnn_native import adamw_init, load_npz_state

    jax.config.update("jax_default_matmul_precision", "highest")
    devices = [device for device in jax.devices() if device.platform == "tpu"]
    if len(devices) != 8:
        raise ValueError("native TPU profile requires eight actual TPU devices")
    args.output_root.mkdir(parents=True, exist_ok=False)
    params = load_npz_state(args.initial_state, 211, 256, 2, 2)
    with np.load(args.profile_inputs, allow_pickle=False) as arrays:
        inputs, targets, mask = [
            _shard(arrays[key], devices) for key in ("inputs", "targets", "row_mask")
        ]
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
                np.full(8, 0.001, dtype=np.float32),
                np.full(8, 0.95, dtype=np.float32),
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
        "device_count": 8,
        "device_inventory": [{"id": str(device), "kind": getattr(device, "device_kind", None)} for device in devices],
        "jax_version": jax.__version__,
        "profile": rows,
        "synchronization": "jax.block_until_ready complete update result tree",
    }
    (args.output_root / "native_profile_receipt.json").write_text(
        json.dumps(receipt, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
