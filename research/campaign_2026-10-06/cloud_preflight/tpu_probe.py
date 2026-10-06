from __future__ import annotations

import hashlib
import json
import os
import platform
import resource
import signal
import subprocess
import sys
import time
import traceback
from contextlib import suppress
from pathlib import Path
from typing import Any

PACKET_ID = "wnn-e463-real-tpu-fullshape-preflight-20261006-v1"
OUTPUT = Path("/kaggle/working/cloud_preflight_tpu_v1")
DATASET = Path("/kaggle/input/wnn-jax-realdata-e463-tpu-20261004-v1/source_bundle")
ASSET_ROOT = Path("/kaggle/input/wunder-funder")
RUN = DATASET / "artifacts/runs/c20261003_combo_s3_f0"
CHECKPOINT = RUN / "model_ep3.pt"
NPZ = RUN / "model_ep3.jax.npz"
SIDECAR = RUN / "model_ep3.jax.npz.json"
MODEL_SOURCE = DATASET / "src/wnn_jax/rnn.py"
EXPECTED_CHECKPOINT_SHA256 = "d4a1e50b1892b4d9edd7c255908b68073a760c059ca80223e98a63c7e023371e"
EXPECTED_NPZ_SHA256 = "677f604fd86d993c3d4a66a10638755253bfe52f1f578009dbd781c73fe6d04a"
EXPECTED_SIDECAR_SHA256 = "3e65d2a88352d22bc90d88b829f05beee3faa7a66b9de41c7507f521aab12668"
EXPECTED_MODEL_SOURCE_SHA256 = "294eff11d906fe434dba2c4d0ed4741ad2191f756c4a35204474cb229ff939dc"
DIMENSIONS = (211, 256, 2, 23)
GLOBAL_BATCH = 16
LOCAL_BATCH = 2
CHUNK_ROWS = 2000
PARENT_TIMEOUT_SECONDS = 540
WORKER_TIMEOUT_SECONDS = 480
STARTED = time.monotonic()
PHASE = "python_entry"
CHILD: subprocess.Popen[bytes] | None = None
TERMINAL: dict[str, Any] = {"status": "starting"}


def write_json(path: Path, value: object) -> None:
    """Atomically persist a phase or final status for Kaggle's output collector."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    """Hash data and source incrementally to keep host memory bounded."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_limit(path: Path) -> int | None:
    try:
        raw = path.read_text().strip()
    except OSError:
        return None
    if raw == "max":
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def host_snapshot() -> dict[str, Any]:
    """Measure host and cgroup RAM before importing JAX or TPU runtime libraries."""
    memory = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        name, raw = line.split(":", 1)
        memory[name] = int(raw.strip().split()[0]) * 1024
    page_size = os.sysconf("SC_PAGE_SIZE")
    resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
    group = Path("/sys/fs/cgroup")
    limit = _read_limit(group / "memory.max")
    current = _read_limit(group / "memory.current")
    if limit is not None and current is not None:
        effective_total = min(memory["MemTotal"], limit)
        cgroup_available = max(0, limit - current)
        available = min(memory["MemAvailable"], cgroup_available)
    else:
        effective_total = memory["MemTotal"]
        available = memory["MemAvailable"]
    reserve = max(2 * 1024**3, int(effective_total * 0.20))
    return {
        "platform": platform.platform(),
        "python": sys.version,
        "cpu_count": os.cpu_count(),
        "host_mem_total_bytes": memory["MemTotal"],
        "host_mem_available_bytes": memory["MemAvailable"],
        "cgroup_memory_limit_bytes": limit,
        "cgroup_memory_current_bytes": current,
        "effective_memory_total_bytes": effective_total,
        "effective_memory_available_bytes": available,
        "required_headroom_bytes": reserve,
        "process_rss_bytes": resident_pages * page_size,
        "process_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        "headroom_passed": available >= reserve,
    }


def safe_stats(device: Any) -> dict[str, Any]:
    """Convert backend-specific TPU memory stats into a JSON-safe mapping."""
    try:
        values = device.memory_stats()
    except Exception as error:
        return {"memory_stats_error": f"{type(error).__name__}: {error}"}
    if not isinstance(values, dict):
        return {}
    result: dict[str, Any] = {}
    for key, value in values.items():
        if value is None or isinstance(value, (str, int, float, bool)):
            result[str(key)] = value
        else:
            try:
                result[str(key)] = int(value)
            except (TypeError, ValueError, OverflowError):
                result[str(key)] = str(value)
    return result


def hbm_fields(stats: dict[str, Any]) -> tuple[int | None, int | None, int | None]:
    """Read current, peak and total HBM capacity across PJRT naming variants."""
    current = stats.get("bytes_in_use")
    if current is None:
        current = stats.get("bytes_reserved")
    peak = stats.get("peak_bytes_in_use")
    if peak is None:
        peak = stats.get("peak_bytes_reserved")
    limit = stats.get("bytes_limit")
    if limit is None:
        limit = stats.get("bytes_reservable_limit")
    if limit is None:
        limit = stats.get("bytes_reserved_limit")
    values = tuple(
        int(value) if isinstance(value, (int, float)) and value >= 0 else None
        for value in (current, peak, limit)
    )
    return values  # type: ignore[return-value]


def tpu_hbm_capacity(jax: Any) -> tuple[int | None, dict[str, Any]]:
    """Use PJRT's capacity stat or the official JAX TPU hardware descriptor."""
    try:
        from jax.experimental.pallas.tpu import get_tpu_info

        info = get_tpu_info()
        per_core = int(info.hbm_capacity_bytes)
        core_count = int(getattr(info, "num_cores", 1))
        return per_core * core_count, {
            "source": "jax.experimental.pallas.tpu.get_tpu_info",
            "hbm_capacity_bytes_per_tensorcore": per_core,
            "tensorcores_per_chip": core_count,
            "chip_version": str(info.chip_version),
        }
    except Exception as error:
        return None, {"source": "unavailable", "error": f"{type(error).__name__}: {error}"}


def device_records(devices: list[Any], fallback_hbm_capacity: int | None) -> list[dict[str, Any]]:
    """Inventory every visible local TPU chip and retain raw HBM telemetry."""
    records = []
    for device in devices:
        stats = safe_stats(device)
        current, peak, limit = hbm_fields(stats)
        capacity_source = "device.memory_stats" if limit is not None else None
        if limit is None and fallback_hbm_capacity is not None:
            limit = fallback_hbm_capacity
            capacity_source = "jax_tpu_hardware_info_fallback"
        records.append(
            {
                "id": int(getattr(device, "id", -1)),
                "platform": str(getattr(device, "platform", "unknown")),
                "device_kind": str(getattr(device, "device_kind", "unknown")),
                "process_index": int(getattr(device, "process_index", -1)),
                "repr": str(device),
                "hbm_current_bytes": current,
                "hbm_peak_bytes": peak,
                "hbm_capacity_bytes": limit,
                "hbm_capacity_source": capacity_source,
                "hbm_required_free_bytes": int(limit * 0.20) if limit else None,
                "memory_stats": stats,
            }
        )
    return records


def verify_inputs() -> dict[str, Any]:
    """Validate the exact source and epoch-3 JAX export attached to this packet."""
    observed = {
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "npz_sha256": sha256_file(NPZ),
        "sidecar_sha256": sha256_file(SIDECAR),
        "model_source_sha256": sha256_file(MODEL_SOURCE),
    }
    expected = {
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "npz_sha256": EXPECTED_NPZ_SHA256,
        "sidecar_sha256": EXPECTED_SIDECAR_SHA256,
        "model_source_sha256": EXPECTED_MODEL_SOURCE_SHA256,
    }
    if observed != expected:
        raise ValueError(f"E463 immutable input hash mismatch: {observed}")
    sidecar = json.loads(SIDECAR.read_text())
    if sidecar.get("npz_sha256") != EXPECTED_NPZ_SHA256:
        raise ValueError("NPZ sidecar does not bind the expected NPZ bytes")
    if sidecar.get("source_checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("NPZ sidecar does not bind the frozen epoch-3 Torch checkpoint")
    if sidecar.get("dimensions") != {
        "n_in": 211,
        "hidden": 256,
        "layers": 2,
        "n_out": 23,
    }:
        raise ValueError("NPZ sidecar model dimensions do not match the E463 model")
    return {**observed, "dimensions": sidecar["dimensions"], "sidecar": sidecar}


def check_finite(jnp: Any, value: Any, name: str) -> bool:
    """Return a synchronized finite predicate or raise with the affected tensor."""
    finite = bool(jnp.all(jnp.isfinite(value)))
    if not finite:
        raise FloatingPointError(f"nonfinite {name}")
    return finite


def run_device_worker() -> dict[str, Any]:
    """Run exact-checkpoint TPU parity and one full local-batch 2000-row update."""
    global PHASE
    PHASE = "framework_import"
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax import lax

    sys.path.insert(0, str(DATASET / "src"))
    from wnn_jax import rnn

    PHASE = "actual_tpu_inventory"
    local_devices = list(jax.local_devices())
    all_devices = list(jax.devices())
    fallback_hbm_capacity, tpu_info = tpu_hbm_capacity(jax)
    records = device_records(all_devices, fallback_hbm_capacity)
    write_json(
        OUTPUT / "tpu_inventory.json",
        {
            "jax_version": jax.__version__,
            "jax_backend": jax.default_backend(),
            "local_device_count": len(local_devices),
            "global_device_count": len(all_devices),
            "tpu_hardware_info": tpu_info,
            "devices": records,
        },
    )
    if jax.default_backend() != "tpu" or len(local_devices) != 8 or len(all_devices) != 8:
        raise RuntimeError(
            "requested TpuV5E8 must expose exactly eight local/global TPU devices; "
            f"backend={jax.default_backend()}, local={len(local_devices)}, global={len(all_devices)}"
        )
    if any(device.platform != "tpu" for device in all_devices):
        raise RuntimeError("JAX exposed a non-TPU device; CPU fallback is not accepted")
    if not ASSET_ROOT.exists():
        raise FileNotFoundError("private wunder-funder asset mount is absent")
    for device_record in records:
        current = device_record["hbm_current_bytes"]
        limit = device_record["hbm_capacity_bytes"]
        if current is None or limit is None:
            raise RuntimeError(
                f"TPU {device_record['id']} did not report measurable HBM capacity/current use"
            )
        if limit - current < int(limit * 0.20):
            raise MemoryError(f"TPU {device_record['id']} lacks 20% free HBM before the test")
    write_json(OUTPUT / "tpu_hbm_before.json", records)

    PHASE = "checkpoint_hash_and_load"
    inputs = verify_inputs()
    dimensions = DIMENSIONS
    params = rnn.load_npz_state(NPZ, *dimensions)
    optimizer = rnn.adamw_init(params)

    PHASE = "short_device_shape_and_finiteness"
    with jax.default_device(local_devices[0]):
        short_input = jnp.zeros((1, 8, DIMENSIONS[0]), dtype=jnp.float32)
        short_forward = jax.jit(lambda p, x: rnn.forward_chunk(p, x, precision="fp32"))
        short_prediction, short_state = short_forward(params, short_input)
        jax.block_until_ready((short_prediction, short_state))
        check_finite(jnp, short_prediction, "short TPU prediction")
        check_finite(jnp, short_state, "short TPU recurrent state")
    short_record = {
        "input_shape": [1, 8, DIMENSIONS[0]],
        "output_shape": list(short_prediction.shape),
        "state_shape": list(short_state.shape),
        "finite": True,
        "device": str(local_devices[0]),
    }
    if tuple(short_prediction.shape) != (1, 8, DIMENSIONS[3]):
        raise AssertionError(f"short TPU output shape changed: {short_prediction.shape}")
    if tuple(short_state.shape) != (DIMENSIONS[2], 1, DIMENSIONS[1]):
        raise AssertionError(f"short TPU state shape changed: {short_state.shape}")
    write_json(OUTPUT / "tpu_short_forward.json", short_record)

    PHASE = "fullshape_data_setup"
    rng = np.random.default_rng(20261006)
    local_inputs = rng.normal(0.0, 0.1, (LOCAL_BATCH, CHUNK_ROWS, DIMENSIONS[0])).astype(np.float32)
    local_labels = rng.normal(0.0, 0.2, (LOCAL_BATCH, CHUNK_ROWS, 2)).astype(np.float32)
    local_targets = np.clip(local_labels * 0.75, -2.0, 2.0).astype(np.float32)
    local_mask = np.ones((LOCAL_BATCH, CHUNK_ROWS), dtype=np.float32)

    # Repeat one deterministic local batch to verify all eight TPU replicas while
    # preserving the exact local per-device batch of a global batch-16 update.
    def replicated(value: Any) -> Any:
        return jnp.asarray(np.repeat(value[None, ...], 8, axis=0))

    shard_inputs = replicated(local_inputs)
    shard_labels = replicated(local_labels)
    shard_targets = replicated(local_targets)
    shard_mask = replicated(local_mask)
    shard_params = jax.tree_util.tree_map(
        lambda value: replicated(np.asarray(jax.device_get(value), dtype=np.float32)), params
    )
    shard_optimizer = jax.tree_util.tree_map(
        lambda value: replicated(np.asarray(jax.device_get(value))), optimizer
    )
    PHASE = "fullshape_forward_backward_update"
    write_json(
        OUTPUT / "tpu_fullshape_started.json",
        {
            "phase": PHASE,
            "global_batch": GLOBAL_BATCH,
            "local_batch_per_tpu": LOCAL_BATCH,
            "tpu_devices": 8,
            "chunk_rows": CHUNK_ROWS,
            "input_shape_global": [GLOBAL_BATCH, CHUNK_ROWS, DIMENSIONS[0]],
            "input_shape_per_device": [LOCAL_BATCH, CHUNK_ROWS, DIMENSIONS[0]],
            "synthetic_inputs_and_targets": True,
            "real_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        },
    )

    def update_one_replica(
        model_params: Any,
        opt_state: Any,
        sequence: Any,
        targets: Any,
        labels: Any,
        row_mask: Any,
    ) -> tuple[Any, Any, Any, Any, Any]:
        def objective(current: Any) -> tuple[Any, tuple[Any, Any]]:
            prediction, next_state = rnn.forward_chunk(current, sequence, precision="fp32")
            if prediction.shape != (LOCAL_BATCH, CHUNK_ROWS, DIMENSIONS[3]):
                raise AssertionError(f"full-shape TPU output shape changed: {prediction.shape}")
            if next_state.shape != (DIMENSIONS[2], LOCAL_BATCH, DIMENSIONS[1]):
                raise AssertionError(f"full-shape TPU recurrent state changed: {next_state.shape}")
            loss = rnn.weighted_fixed_target_mse(prediction, targets, labels, row_mask)
            finite_values = jnp.all(jnp.isfinite(prediction)) & jnp.all(jnp.isfinite(next_state))
            return loss, (finite_values, jnp.all(jnp.isfinite(next_state)))

        (loss, (finite_forward, finite_state)), gradients = jax.value_and_grad(
            objective, has_aux=True
        )(model_params)
        gradients = jax.tree_util.tree_map(lambda grad: lax.pmean(grad, "tpu_replicas"), gradients)
        mean_loss = lax.pmean(loss, "tpu_replicas")
        finite_gradients = jnp.asarray(True)
        for gradient in jax.tree_util.tree_leaves(gradients):
            finite_gradients = finite_gradients & jnp.all(jnp.isfinite(gradient))
        updated, next_optimizer = rnn.adamw_update(
            model_params,
            gradients,
            opt_state,
            learning_rate=1e-4,
            weight_decay=0.01,
        )
        return updated, next_optimizer, mean_loss, finite_forward, finite_gradients & finite_state

    update = jax.pmap(
        update_one_replica,
        axis_name="tpu_replicas",
        in_axes=(0, 0, 0, 0, 0, 0),
        devices=local_devices,
    )
    started = time.monotonic()
    updated, optimizer_after, losses, finite_outputs, finite_gradients = update(
        shard_params,
        shard_optimizer,
        shard_inputs,
        shard_targets,
        shard_labels,
        shard_mask,
    )
    jax.block_until_ready((updated, optimizer_after, losses, finite_outputs, finite_gradients))
    if not bool(jnp.all(jnp.isfinite(losses))):
        raise FloatingPointError("full-shape TPU loss is nonfinite")
    if not bool(jnp.all(finite_outputs)) or not bool(jnp.all(finite_gradients)):
        raise FloatingPointError("full-shape TPU output/state/gradient contains NaN or Inf")
    finite_updated = all(
        bool(jnp.all(jnp.isfinite(value))) for value in jax.tree_util.tree_leaves(updated)
    )
    if not finite_updated:
        raise FloatingPointError("one or more full-shape TPU updated parameters are nonfinite")
    step_counts = np.asarray(jax.device_get(optimizer_after.count)).tolist()
    if step_counts != [1] * 8:
        raise AssertionError(f"each TPU replica must complete exactly one update: {step_counts}")
    update_seconds = time.monotonic() - started
    write_json(
        OUTPUT / "tpu_fullshape_complete.json",
        {
            "status": "passed",
            "global_batch": GLOBAL_BATCH,
            "local_batch_per_tpu": LOCAL_BATCH,
            "device_count": 8,
            "chunk_rows": CHUNK_ROWS,
            "input_shape_global": [GLOBAL_BATCH, CHUNK_ROWS, DIMENSIONS[0]],
            "input_shape_per_device": [LOCAL_BATCH, CHUNK_ROWS, DIMENSIONS[0]],
            "output_shape_per_device": [LOCAL_BATCH, CHUNK_ROWS, DIMENSIONS[3]],
            "state_shape_per_device": [DIMENSIONS[2], LOCAL_BATCH, DIMENSIONS[1]],
            "optimizer_steps_per_replica": step_counts,
            "finite_loss": [float(value) for value in np.asarray(jax.device_get(losses))],
            "finite_outputs_states_gradients": True,
            "synthetic_inputs_and_targets": True,
            "update_seconds": update_seconds,
            "scientific_score_claim": None,
        },
    )

    PHASE = "export_reload_parity"
    first_replica = jax.tree_util.tree_map(lambda value: value[0], updated)
    export_path = OUTPUT / "e463_tpu_fullshape_one_step.npz"
    rnn.export_npz_state(export_path, first_replica)
    reloaded = rnn.load_npz_state(export_path, *dimensions)
    with jax.default_device(local_devices[0]):
        after_export, _ = short_forward(reloaded, short_input)
        before_export, _ = short_forward(first_replica, short_input)
        jax.block_until_ready((after_export, before_export))
    check_finite(jnp, after_export, "export/reload TPU prediction")
    max_error = float(jnp.max(jnp.abs(after_export - before_export)))
    np.testing.assert_allclose(
        np.asarray(after_export), np.asarray(before_export), rtol=1e-5, atol=1e-6
    )
    write_json(
        OUTPUT / "tpu_export_reload.json",
        {
            "status": "passed",
            "npz_sha256": sha256_file(export_path),
            "bytes": export_path.stat().st_size,
            "input_shape": [1, 8, DIMENSIONS[0]],
            "output_shape": list(after_export.shape),
            "post_reload_max_abs_error": max_error,
        },
    )

    PHASE = "hbm_headroom_validation"
    final_records = device_records(all_devices, fallback_hbm_capacity)
    hbm_failures = []
    for _, after in zip(records, final_records, strict=True):
        limit = after["hbm_capacity_bytes"]
        peak = after["hbm_peak_bytes"]
        current = after["hbm_current_bytes"]
        if limit is None or peak is None or current is None:
            hbm_failures.append(f"TPU {after['id']} missing capacity/peak/current HBM fields")
            continue
        if peak > int(limit * 0.80) or limit - current < int(limit * 0.20):
            hbm_failures.append(f"TPU {after['id']} violated 20% HBM headroom")
    write_json(OUTPUT / "tpu_hbm_after.json", final_records)
    if hbm_failures:
        raise MemoryError("; ".join(hbm_failures))
    return {
        "status": "passed",
        "classification": "actual eight-device TPUv5e E463 checkpoint/model full-shape memory and numerical preflight; one synthetic batch-16 diagnostic update; no scientific fit or score",
        "actual_hardware": records,
        "model_source_sha256": EXPECTED_MODEL_SOURCE_SHA256,
        "inputs": inputs,
        "dimensions": {
            "n_in": DIMENSIONS[0],
            "hidden": DIMENSIONS[1],
            "layers": DIMENSIONS[2],
            "n_out": DIMENSIONS[3],
        },
        "short_device_check": short_record,
        "fullshape": {
            "global_batch": GLOBAL_BATCH,
            "local_batch_per_tpu": LOCAL_BATCH,
            "chunk_rows": CHUNK_ROWS,
            "update_seconds": update_seconds,
        },
        "export_reload": {
            "sha256": sha256_file(export_path),
            "post_reload_max_abs_error": max_error,
        },
        "hbm_after": final_records,
        "elapsed_seconds": time.monotonic() - STARTED,
    }


def worker_main() -> int:
    """Run device checks in a child so import/runtime crashes retain parent logs."""
    global PHASE, TERMINAL
    PHASE = "device_worker_start"
    write_json(
        OUTPUT / "device_worker_start.json",
        {"time_epoch_seconds": time.time(), "python": sys.version, "packet": PACKET_ID},
    )
    try:
        result = run_device_worker()
    except BaseException as error:
        TERMINAL = {
            "status": "failed",
            "phase": PHASE,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        print(json.dumps({"event": "tpu_probe_failed", **TERMINAL}), flush=True)
        write_json(OUTPUT / "device_worker_terminal.json", TERMINAL)
        return 1
    TERMINAL = {"status": "passed", "phase": "complete", "result": result}
    write_json(OUTPUT / "device_worker_result.json", result)
    write_json(OUTPUT / "device_worker_terminal.json", TERMINAL)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


def stop_child_process_group(child: subprocess.Popen[bytes]) -> None:
    """Terminate the bounded worker and any XLA helper processes it started."""
    with suppress(ProcessLookupError):
        os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        with suppress(ProcessLookupError):
            os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=5)


def parent_main() -> int:
    """Run the TPU payload under an internal bound and always preserve terminal status."""
    global CHILD, PHASE, TERMINAL
    OUTPUT.mkdir(parents=True, exist_ok=False)
    write_json(
        OUTPUT / "startup.json",
        {
            "phase": "python_started_before_jax_import",
            "time_epoch_seconds": time.time(),
            "python": sys.version,
            "platform": platform.platform(),
            "packet": PACKET_ID,
            "requested_hardware": "TpuV5E8",
        },
    )
    signal.signal(
        signal.SIGALRM,
        lambda _signum, _frame: (_ for _ in ()).throw(
            TimeoutError(f"TPU payload exceeded {PARENT_TIMEOUT_SECONDS} seconds")
        ),
    )
    signal.alarm(PARENT_TIMEOUT_SECONDS)
    try:
        PHASE = "host_admission"
        host = host_snapshot()
        write_json(OUTPUT / "host_start.json", host)
        if not host["headroom_passed"]:
            raise MemoryError("host RAM is below the 20%/2GiB preflight headroom requirement")
        if not DATASET.exists() or not RUN.exists():
            raise FileNotFoundError(
                "verified private E463 source/checkpoint dataset mount is absent"
            )
        if not ASSET_ROOT.exists():
            raise FileNotFoundError("private wunder-funder dataset mount is absent")
        PHASE = "launch_bounded_tpu_worker"
        environment = os.environ.copy()
        environment.update(
            {
                "JAX_PLATFORMS": "tpu",
                "PJRT_DEVICE": "TPU",
                "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
                "PYTHONUNBUFFERED": "1",
            }
        )
        log_path = OUTPUT / "tpu_worker.log"
        with log_path.open("wb") as log:
            CHILD = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), "--device-worker"],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=environment,
                start_new_session=True,
            )
            write_json(
                OUTPUT / "worker_launch.json",
                {
                    "pid": CHILD.pid,
                    "started_epoch_seconds": time.time(),
                    "timeout_seconds": WORKER_TIMEOUT_SECONDS,
                },
            )
            try:
                return_code = CHILD.wait(timeout=WORKER_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired as error:
                PHASE = "device_worker_timeout"
                stop_child_process_group(CHILD)
                return_code = CHILD.returncode
                write_json(
                    OUTPUT / "worker_process.json",
                    {
                        "status": "timeout",
                        "timeout_seconds": WORKER_TIMEOUT_SECONDS,
                        "returncode": return_code,
                    },
                )
                raise TimeoutError(
                    f"TPU device worker exceeded {WORKER_TIMEOUT_SECONDS} seconds"
                ) from error
        write_json(OUTPUT / "worker_process.json", {"status": "exited", "returncode": return_code})
        if return_code != 0:
            child_terminal = OUTPUT / "device_worker_terminal.json"
            failure = json.loads(child_terminal.read_text()) if child_terminal.exists() else None
            raise RuntimeError(f"TPU device worker failed (exit={return_code}): {failure}")
        result = json.loads((OUTPUT / "device_worker_result.json").read_text())
        if result.get("status") != "passed":
            raise RuntimeError("TPU worker completed without a passing device result")
        PHASE = "complete"
        TERMINAL = {"status": "passed", "phase": PHASE, "worker_returncode": return_code}
        write_json(OUTPUT / "result.json", result)
        return 0
    except BaseException as error:
        TERMINAL = {
            "status": "failed",
            "phase": PHASE,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        print(json.dumps({"event": "tpu_parent_failed", **TERMINAL}), flush=True)
        return 1
    finally:
        signal.alarm(0)
        if CHILD is not None and CHILD.poll() is None:
            stop_child_process_group(CHILD)
        TERMINAL["elapsed_seconds"] = time.monotonic() - STARTED
        TERMINAL["last_phase"] = PHASE
        TERMINAL["terminal_host_memory"] = host_snapshot()
        write_json(OUTPUT / "terminal.json", TERMINAL)
        write_json(OUTPUT / "complete.json", {"completed_epoch_seconds": time.time(), **TERMINAL})


if __name__ == "__main__":
    if sys.argv[1:] == ["--device-worker"]:
        raise SystemExit(worker_main())
    raise SystemExit(parent_main())
