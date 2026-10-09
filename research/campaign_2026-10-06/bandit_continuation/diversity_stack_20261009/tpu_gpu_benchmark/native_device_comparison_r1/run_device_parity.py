"""Run admitted TPU lifecycle tests and a bounded, discarded full-shape update.

This program never contacts a provider or starts a production fit. Root must
admit its actual-device execution and bind its receipt into the fit contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path


def validate_profile_receipt(
    inputs: Path, receipt_path: Path, initial_state: Path
) -> dict[str, object]:
    """Consume the actual GPU writer's hash keys before touching a TPU client."""
    receipt = json.loads(receipt_path.read_text())
    with inputs.open("rb") as stream:
        inputs_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    with initial_state.open("rb") as stream:
        initial_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    if (
        receipt.get("schema") != "wnn.loss-pair.profile-inputs-receipt.v1"
        or receipt.get("profile_inputs_sha256") != inputs_sha
        or receipt.get("initial_state_sha256") != initial_sha
        or not receipt.get("cache_manifest_sha256")
    ):
        raise ValueError("actual GPU profile fixture byte/init/cache binding differs")
    return receipt


def main() -> None:
    """Write hash-bound measured parity evidence; never infer passing skips."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-state", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--profile-inputs",
        type=Path,
        help="Hash-bound actual parsed-cache first batch: inputs, targets, row_mask NPZ.",
    )
    parser.add_argument("--profile-inputs-receipt", type=Path)
    parser.add_argument("--warm-repeats", type=int, default=5)
    args = parser.parse_args()
    if not 3 <= args.warm_repeats <= 10:
        raise ValueError("warm repeats must be bounded between three and ten")
    here = Path(__file__).resolve().parent
    args.output_root.mkdir(parents=True, exist_ok=False)
    junit = args.output_root / "lifecycle_tests.xml"
    env = os.environ.copy()
    env["WNN_PARITY_INITIAL_STATE"] = str(args.initial_state.resolve(strict=True))
    if args.profile_inputs:
        env["WNN_PARITY_PROFILE_INPUTS"] = str(args.profile_inputs.resolve(strict=True))
        if args.profile_inputs_receipt is None:
            raise ValueError("actual profile inputs require their producer receipt")
        validate_profile_receipt(
            args.profile_inputs, args.profile_inputs_receipt, args.initial_state
        )
    started = time.perf_counter()
    with (args.output_root / "lifecycle_tests.log").open("w") as log:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", str(here / "tests"), f"--junitxml={junit}"],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=900,
            check=False,
        )
    suites = ET.parse(junit).getroot()
    cases = list(suites.iter("testcase"))
    if result.returncode or not cases or any(list(case) for case in cases):
        raise RuntimeError("device lifecycle tests failed or skipped; inspect preserved XML/log")

    # Import after the test subprocess exits, avoiding simultaneous TPU clients.
    import jax
    import jax.numpy as jnp
    import numpy as np
    from tpu_worker import _replicate, _shard, _training_step, _write_json_atomic, sha256_file
    from rnn_native import adamw_init, load_npz_state

    jax.config.update("jax_default_matmul_precision", "highest")
    devices = [device for device in jax.devices() if device.platform == "tpu"]
    if len(devices) != 8:
        raise RuntimeError("this receipt requires eight actual TPU devices")
    params = load_npz_state(args.initial_state, 211, 256, 2, 2)
    host_x = np.random.default_rng(20261009).normal(size=(32, 2000, 211)).astype(np.float32)
    host_y = np.random.default_rng(7).normal(size=(32, 2000, 2)).astype(np.float32)
    host_mask = np.ones((32, 2000), dtype=bool)
    host_mask[:, :99] = False
    input_receipt = None
    if args.profile_inputs:
        if args.profile_inputs_receipt is None:
            raise ValueError("actual profile inputs require their producer receipt")
        input_receipt = validate_profile_receipt(
            args.profile_inputs, args.profile_inputs_receipt, args.initial_state
        )
        with np.load(args.profile_inputs, allow_pickle=False) as archive:
            host_x = archive["inputs"]
            host_y = archive["targets"]
            host_mask = archive["row_mask"].astype(bool)
        if (
            host_x.shape != (32, 2000, 211)
            or host_y.shape != (32, 2000, 2)
            or host_mask.shape != (32, 2000)
            or not np.isfinite(host_x).all()
            or not np.isfinite(host_y).all()
        ):
            raise ValueError("actual profile fixture shape or finiteness differs")
    # Time the real full batch; check zero-weight remainder reduction separately.
    padded_mask = host_mask.copy()
    padded_mask[30:] = False
    state = _replicate(jnp.zeros((2, 4, 256), dtype=jnp.float32), devices)
    profile = {}
    for loss in ("wmse", "huber"):
        step = _training_step(loss, "fp32")
        device_params = _replicate(params, devices)
        optimizer = _replicate(adamw_init(params), devices)
        inputs, targets, mask = [_shard(value, devices) for value in (host_x, host_y, host_mask)]
        step_start = time.perf_counter()
        output = step(
            device_params,
            optimizer,
            inputs,
            state,
            targets,
            mask,
            np.full(8, 0.001, dtype=np.float32),
            np.full(8, 0.95, dtype=np.float32),
        )
        jax.block_until_ready(output)
        cold = time.perf_counter() - step_start
        warm_samples = []
        for _ in range(args.warm_repeats):
            step_start = time.perf_counter()
            warm = step(
                device_params,
                optimizer,
                inputs,
                state,
                targets,
                mask,
                np.full(8, 0.001, dtype=np.float32),
                np.full(8, 0.95, dtype=np.float32),
            )
            jax.block_until_ready(warm)
            warm_samples.append(time.perf_counter() - step_start)
            if not np.asarray(jax.device_get(warm[-1])).all():
                raise FloatingPointError(f"nonfinite {loss} synchronized warm update")
        warm_seconds = float(np.median(warm_samples))
        if not np.asarray(jax.device_get(output[-1])).all():
            raise FloatingPointError(f"nonfinite {loss} full-shape discarded update")
        # Compare the global replica reduction against one unsharded update
        # using a short prefix, preserving the padded-mask test independently.
        from jax_step import train_chunk_step

        reference = train_chunk_step(
            params,
            adamw_init(params),
            jnp.asarray(host_x[:, :101]),
            None,
            jnp.asarray(host_y[:, :101]),
            jnp.asarray(padded_mask[:, :101]),
            learning_rate=0.001,
            beta1=0.95,
            loss_kind=loss,
            precision="fp32",
        )
        reduced = step(
            device_params,
            optimizer,
            _shard(host_x[:, :101], devices),
            state,
            _shard(host_y[:, :101], devices),
            _shard(padded_mask[:, :101], devices),
            np.full(8, 0.001, dtype=np.float32),
            np.full(8, 0.95, dtype=np.float32),
        )
        jax.block_until_ready((reference, reduced))
        for index in (3, 4):
            np.testing.assert_allclose(
                np.asarray(reduced[index][0]), np.asarray(reference[index]), rtol=5e-4, atol=3e-6
            )
        for name, value in reference[0].items():
            np.testing.assert_allclose(
                np.asarray(reduced[0][name][0]),
                np.asarray(value),
                rtol=5e-4,
                atol=2e-5,
                err_msg=f"{loss} replica update {name}",
            )
        for name, value in reference[1].first.items():
            np.testing.assert_allclose(
                np.asarray(reduced[1].first[name][0]), np.asarray(value), rtol=5e-4, atol=3e-7
            )
            np.testing.assert_allclose(
                np.asarray(reduced[1].second[name][0]),
                np.asarray(reference[1].second[name]),
                rtol=1e-3,
                atol=3e-9,
            )
        from tpu_worker import _memory_snapshot

        profile[loss] = {
            "compile_plus_first_update_seconds": cold,
            "warm_update_seconds": warm_seconds,
            "warm_update_seconds_samples": warm_samples,
            "warm_update_statistic": "median",
            "warm_update_seconds_max": max(warm_samples),
            "synchronization": "jax.block_until_ready complete update result tree",
            "shape": [32, 2000, 211],
            "active_sequences": 32,
            "padded_remainder_active_sequences_checked": 30,
            "finite": True,
            "memory_after_full_shape": _memory_snapshot(),
        }
    _write_json_atomic(
        args.output_root / "device_parity_receipt.json",
        {
            "status": "pass" if input_receipt else "probe",
            "actual_data_profile": bool(input_receipt),
            "profile_inputs_receipt": input_receipt,
            "platform": "tpu",
            "accelerator_count": 8,
            "device_inventory": [str(device) for device in devices],
            "passed_checks": [
                "forward",
                "gradient",
                "adamw_update",
                "reload",
                "reset",
                "prefix_causality",
                "replica_reduction",
                "full_shape_profile",
            ],
            "jax_step_sha256": sha256_file(here / "jax_step.py"),
            "jax_core_sha256": sha256_file(here / "rnn_native.py"),
            "initial_state_sha256": sha256_file(args.initial_state),
            "test_xml_sha256": sha256_file(junit),
            "profile": profile,
            "matmul_precision": "highest",
            "elapsed_seconds": time.perf_counter() - started,
            "limits": "Representative discarded full-shape profile with pinned actual inputs."
            if input_receipt
            else "Synthetic profile only; no production-fit admission or throughput multiplier claim.",
        },
    )


if __name__ == "__main__":
    main()
