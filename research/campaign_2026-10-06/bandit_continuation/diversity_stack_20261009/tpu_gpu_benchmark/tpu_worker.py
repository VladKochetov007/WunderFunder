"""Finite, resumable TPU runner for the matched GRU256 loss-pair screen.

The worker consumes only pinned parsed-cache artifacts and a frozen Torch
initialization, sample order, and OneCycle trace. It never reads Parquet, creates
recipes, or contacts a provider. Provider admission and kernel packaging remain
outside this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "research/campaign_2026-10-04/staging/jax_tpu_e465/src"))

from jax_step import train_chunk_step  # noqa: E402
from wnn_jax.rnn import (  # noqa: E402
    AdamWState,
    adamw_init,
    export_npz_state,
    load_npz_state,
)

from wnn.features import event_features  # noqa: E402
from wnn.gmetric import GlobalAccumulator  # noqa: E402
from wnn.parsed_sequence_cache_reader import ParsedSequenceCache  # noqa: E402

CONTRACT_SCHEMA = "wnn.loss-pair-tpu-run-contract.v1"
ORDERS_SCHEMA = "wnn.loss-pair.sample-orders.v1"
TRACE_SCHEMA = "wnn.loss-pair.onecycle-trace.v1"
RUNNER_SCHEMA = "wnn.loss-pair-tpu-terminal.v1"
EXPECTED_RECIPE = {
    "n_in": 211,
    "hidden": 256,
    "layers": 2,
    "n_out": 2,
    "batch_size": 32,
    "chunk_rows": 2000,
    "epochs": 2,
    "seed": 20261009,
    "warmup_rows": 99,
    "fit_train_sequences": 350,
    "fit_valid_sequences_unused": 28,
    "evaluation_sequences": 468,
    "updates_per_arm": 220,
    "gradient_clip_norm": 1.0,
    "weight_decay": 0.01,
    "beta2": 0.999,
    "epsilon": 1e-8,
    "huber_delta": 1.0,
}
LOSS_KINDS = ("wmse", "huber")
HASH_BLOCK_BYTES = 8 << 20


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(HASH_BLOCK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _atomic_save_npz(path: Path, arrays: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("xb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _artifact(contract: Mapping[str, Any], name: str) -> tuple[Path, str]:
    record = contract["artifacts"][name]
    path = Path(record["path"]).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve(strict=True), str(record["sha256"])


def _verify_artifact(contract: Mapping[str, Any], name: str) -> Path:
    path, expected = _artifact(contract, name)
    if path.stat().st_size != contract["artifacts"][name]["size_bytes"]:
        raise ValueError(f"{name} byte size differs from the run contract")
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"{name} SHA-256 differs from the run contract")
    return path


def _validate_contract(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    contract = json.loads(raw)
    if not isinstance(contract, dict) or contract.get("schema") != CONTRACT_SCHEMA:
        raise ValueError("unsupported TPU run contract")
    recipe = contract.get("recipe")
    if not isinstance(recipe, dict) or any(
        recipe.get(key) != value for key, value in EXPECTED_RECIPE.items()
    ):
        raise ValueError("recipe differs from the frozen LOSS_PAIR_R2 contract")
    if recipe.get("precision") not in {"fp32", "bf16"}:
        raise ValueError(
            "precision must be fp32 for matched comparison or bf16 for the native TPU arm"
        )
    if recipe.get("losses") != list(LOSS_KINDS):
        raise ValueError("the finite runner accepts exactly the WMSE and Huber arms")
    identity = contract.get("identity")
    if not isinstance(identity, dict):
        raise ValueError("contract has no cache identity")
    expected_identity = {
        "fold": 0,
        "split_manifest_sha256": "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d",
        "membership_sha256": "dbd9ea6d7b0b752495745ea1ebd4bd858af9305c40f0ada7de7386c5359f3acd",
        "portable_registry_sha256": "90d26c7063050092eeda774c7c6298897a8fb133e4804ac13d0a281bd328ff8c",
    }
    if any(identity.get(key) != value for key, value in expected_identity.items()):
        raise ValueError("cache identity differs from the frozen fold-0 loss-pair membership")
    missing = {
        "cache_manifest",
        "cache_validation_receipt",
        "initial_state",
        "onecycle_trace",
        "sample_orders",
        "device_parity_receipt",
    } - set(contract.get("artifacts", {}))
    if missing:
        raise ValueError(f"contract omits pinned artifacts: {sorted(missing)}")
    return contract, sha256_bytes(raw)


def _read_onecycle_trace(path: Path) -> np.ndarray:
    trace = _read_json(path)
    if trace.get("schema") != TRACE_SCHEMA or trace.get("total_steps") != 220:
        raise ValueError("OneCycle trace must bind exactly 220 attempted updates")
    values = trace.get("rows")
    if not isinstance(values, list) or len(values) != 220:
        raise ValueError("OneCycle trace must contain 220 pre-update schedule rows")
    output = np.empty((220, 2), dtype=np.float64)
    for index, row in enumerate(values):
        if row.get("attempt_index") != index:
            raise ValueError("OneCycle trace update indices are not contiguous")
        expected_location = (index // 110, (index % 110) // 10, index % 10)
        if (
            tuple(row.get(key) for key in ("epoch_index", "batch_index", "chunk_index"))
            != expected_location
            or row.get("scheduler_step_pre") != index
        ):
            raise ValueError("OneCycle trace location differs from the frozen update clock")
        groups = row.get("optimizer_groups")
        if not isinstance(groups, list) or len(groups) != 1 or groups[0].get("group_index") != 0:
            raise ValueError("matched AdamW trace requires exactly optimizer group zero")
        output[index, 0] = float(groups[0]["lr"])
        output[index, 1] = float(groups[0]["beta1"])
    if (
        not np.isfinite(output).all()
        or np.any(output[:, 0] <= 0)
        or np.any((output[:, 1] <= 0) | (output[:, 1] >= 1))
    ):
        raise ValueError("OneCycle trace contains invalid lr/beta1 values")
    return output


def _read_orders(path: Path, membership: Mapping[str, Any]) -> dict[str, Any]:
    orders = _read_json(path)
    if (
        orders.get("schema") != ORDERS_SCHEMA
        or orders.get("seed") != 20261009
        or orders.get("batch_size") != 32
    ):
        raise ValueError("sample-order artifact does not bind the frozen seed and batch size")
    by_id: dict[str, dict[str, Any]] = {}
    for role in ("fit_train", "fit_valid", "evaluation_valid"):
        for member in membership["memberships"][role]:
            by_id[member["sequence_id"]] = member
    fit_ids = {member["sequence_id"] for member in membership["memberships"]["fit_train"]}
    eval_ids = {member["sequence_id"] for member in membership["memberships"]["evaluation_valid"]}
    epochs = orders.get("epochs")
    if not isinstance(epochs, dict) or set(epochs) != {"-1", "0", "1"}:
        raise ValueError("sample-order artifact must include normalization and both fit epochs")
    for epoch in ("-1", "0", "1"):
        values = epochs[epoch]
        if len(values) != 350 or len(set(values)) != 350 or set(values) != fit_ids:
            raise ValueError(f"epoch {epoch} is not a permutation of the complete fit membership")
    norm_ids = epochs["-1"][:64]
    if (
        not isinstance(norm_ids, list)
        or len(norm_ids) != 64
        or any(item not in fit_ids for item in norm_ids)
    ):
        raise ValueError("normalization IDs must name the first two complete epoch -1 batches")
    eval_order = orders.get("evaluation_ids")
    if (
        not isinstance(eval_order, list)
        or len(eval_order) != 468
        or len(set(eval_order)) != 468
        or set(eval_order) != eval_ids
    ):
        raise ValueError("evaluation order is not the fixed 468-sequence membership")
    if set(epochs["0"]) & set(eval_order):
        raise ValueError("training and evaluation memberships overlap")
    return {
        "by_id": by_id,
        "epochs": epochs,
        "evaluation_ids": eval_order,
        "normalization_ids": norm_ids,
    }


def _member_source(member: Mapping[str, Any]) -> str:
    logical = member["source"]
    if logical == "datasets/train.parquet":
        return "train"
    if logical == "datasets/valid.parquet":
        return "valid"
    raise ValueError(f"unexpected parsed-cache source identity: {logical}")


def _cache_batch(
    cache: ParsedSequenceCache,
    member_by_id: Mapping[str, Mapping[str, Any]],
    sequence_ids: list[str],
    role: str,
    *,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    batch_ids = sequence_ids[:batch_size]
    if len(batch_ids) > batch_size:
        raise ValueError("batch has more sequences than the frozen global batch")
    features = np.zeros((batch_size, 20_000, 211), dtype=np.float32)
    targets = np.zeros((batch_size, 20_000, 2), dtype=np.float32)
    scored = np.zeros((batch_size, 20_000), dtype=bool)
    for batch_index, sequence_id in enumerate(batch_ids):
        member = member_by_id[sequence_id]
        if role not in member.get("roles", [role]):
            raise ValueError(f"sequence is not in requested cache role: {sequence_id}")
        source = _member_source(member)
        row = int(member["array_index"])
        raw = cache.arrays[f"{source}_features"][row]
        # Per-sequence conversion keeps transient event-feature arrays below the
        # global 1 GB support envelope while preserving the 20k causal clock.
        features[batch_index, :, :112] = raw
        features[batch_index, :, 112:] = event_features(raw)
        targets[batch_index] = cache.arrays[f"{source}_targets"][row]
        scored[batch_index] = cache.arrays[f"{source}_scored"][row]
    return features, targets, scored, batch_ids


def _shard(value: np.ndarray, devices: list[Any]) -> jax.Array:
    if value.shape[0] != len(devices) * 4:
        raise ValueError("global batch must shard as four sequences per each of eight chips")
    local = value.reshape((len(devices), 4, *value.shape[1:]))
    return jax.device_put_sharded(
        [jnp.asarray(local[index]) for index in range(len(devices))], devices
    )


def _replicate(tree: Any, devices: list[Any]) -> Any:
    return jax.device_put_replicated(tree, devices)


def _first_replica(tree: Any) -> Any:
    return jax.tree_util.tree_map(lambda value: value[0], tree)


def _host_tree(tree: Any) -> dict[str, np.ndarray]:
    return {key: np.asarray(jax.device_get(value), dtype=np.float32) for key, value in tree.items()}


def _cgroup_memory() -> dict[str, int | str]:
    result: dict[str, int | str] = {}
    for root in (Path("/sys/fs/cgroup"), Path("/sys/fs/cgroup")):
        try:
            current = (root / "memory.current").read_text().strip()
            result["memory_current_bytes"] = int(current) if current != "max" else -1
            for line in (root / "memory.stat").read_text().splitlines():
                key, value = line.split()
                if key in {"anon", "file", "file_dirty", "file_writeback"}:
                    result[f"memory_{key}_bytes"] = int(value)
            result["path"] = str(root)
            return result
        except (OSError, ValueError):
            continue
    return result


def _memory_snapshot() -> dict[str, Any]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    device_stats = []
    for device in jax.devices():
        try:
            device_stats.append(device.memory_stats())
        except (AttributeError, RuntimeError):
            device_stats.append(None)
    return {
        "max_rss_bytes": int(usage.ru_maxrss * (1024 if sys.platform != "darwin" else 1)),
        "cgroup": _cgroup_memory(),
        "device_memory_stats": device_stats,
    }


def _checkpoint_arrays(
    params: Mapping[str, Any], optimizer: AdamWState, metadata: Mapping[str, Any]
) -> dict[str, Any]:
    arrays: dict[str, Any] = {
        f"param::{key}": value for key, value in _host_tree(_first_replica(params)).items()
    }
    arrays["optimizer::count"] = np.asarray(jax.device_get(optimizer.count[0]), dtype=np.int32)
    for key, value in optimizer.first.items():
        arrays[f"optimizer::first::{key}"] = np.asarray(jax.device_get(value[0]), dtype=np.float32)
    for key, value in optimizer.second.items():
        arrays[f"optimizer::second::{key}"] = np.asarray(jax.device_get(value[0]), dtype=np.float32)
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    return arrays


def _load_checkpoint(
    path: Path, receipt_path: Path, contract_sha: str, loss_kind: str
) -> tuple[dict[str, Any], AdamWState, dict[str, Any]]:
    receipt = _read_json(receipt_path)
    if receipt.get("sha256") != sha256_file(path):
        raise ValueError("training checkpoint hash differs from its resume receipt")
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        if (
            metadata.get("contract_sha256") != contract_sha
            or metadata.get("loss_kind") != loss_kind
        ):
            raise ValueError("checkpoint belongs to a different contract or loss arm")
        params = {
            key.removeprefix("param::"): jnp.asarray(archive[key])
            for key in archive.files
            if key.startswith("param::")
        }
        first = {
            key.removeprefix("optimizer::first::"): jnp.asarray(archive[key])
            for key in archive.files
            if key.startswith("optimizer::first::")
        }
        second = {
            key.removeprefix("optimizer::second::"): jnp.asarray(archive[key])
            for key in archive.files
            if key.startswith("optimizer::second::")
        }
        count = jnp.asarray(archive["optimizer::count"], dtype=jnp.int32)
    return params, AdamWState(count, first, second), metadata


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(record, sort_keys=True, allow_nan=False).encode() + b"\n"
    with path.open("ab") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _truncate_jsonl(path: Path, count: int) -> None:
    if not path.exists():
        if count:
            raise ValueError("checkpoint says updates exist but diagnostic log is missing")
        return
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            break
    if len(rows) < count:
        raise ValueError("diagnostic log is shorter than the committed checkpoint")
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as stream:
        for row in rows[:count]:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _save_resume_checkpoint(
    arm_dir: Path,
    params: Any,
    optimizer: AdamWState,
    *,
    contract_sha: str,
    loss_kind: str,
    precision: str,
    next_epoch: int,
    next_batch: int,
    updates_completed: int,
) -> None:
    path = arm_dir / "training_checkpoint_latest.npz"
    receipt_path = arm_dir / "training_checkpoint_latest.json"
    metadata = {
        "schema": "wnn.loss-pair-training-checkpoint.v1",
        "contract_sha256": contract_sha,
        "loss_kind": loss_kind,
        "precision": precision,
        "next_epoch": next_epoch,
        "next_batch": next_batch,
        "updates_completed": updates_completed,
        "saved_at_unix": time.time(),
    }
    _atomic_save_npz(path, _checkpoint_arrays(params, optimizer, metadata))
    _write_json_atomic(receipt_path, {"sha256": sha256_file(path), "metadata": metadata})


def _training_step(loss_kind: str, precision: str):
    def step(params, optimizer, inputs, state, targets, row_mask, learning_rate, beta1):
        return train_chunk_step(
            params,
            optimizer,
            inputs,
            state,
            targets,
            row_mask,
            learning_rate=learning_rate,
            beta1=beta1,
            weight_decay=0.01,
            beta2=0.999,
            epsilon=1e-8,
            gradient_clip_norm=1.0,
            huber_delta=1.0,
            loss_kind=loss_kind,
            precision=precision,
            replica_axis_name="replicas",
        )

    return jax.pmap(step, axis_name="replicas")


def _forward_step(precision: str):
    from wnn_jax.rnn import forward_chunk

    def step(params, inputs, state):
        return forward_chunk(params, inputs, state, precision=precision)

    return jax.pmap(step)


def _write_update_result(path: Path, result: Any) -> tuple[float, float, float, bool, float]:
    _, _, _, loss, norm, scale, finite = result
    jax.block_until_ready((loss, norm, scale, finite))
    host = [np.asarray(jax.device_get(item))[0].item() for item in (loss, norm, scale, finite)]
    return float(host[0]), float(host[1]), float(host[2]), bool(host[3]), 1.0


def _arm_terminal_receipt(arm_dir: Path, contract_sha: str) -> dict[str, Any] | None:
    receipt_path = arm_dir / "terminal_receipt.json"
    if not receipt_path.exists():
        return None
    receipt = _read_json(receipt_path)
    if receipt.get("contract_sha256") != contract_sha or receipt.get("schema") != RUNNER_SCHEMA:
        raise ValueError("existing terminal receipt belongs to another run contract")
    for name, expected in receipt.get("artifacts", {}).items():
        if sha256_file(arm_dir / name) != expected:
            raise ValueError(f"completed arm artifact changed: {name}")
    return receipt


def _run_arm(
    *,
    cache: ParsedSequenceCache,
    orders: Mapping[str, Any],
    initial_params: Mapping[str, Any],
    schedule: np.ndarray,
    contract: Mapping[str, Any],
    contract_sha: str,
    output_root: Path,
    devices: list[Any],
    loss_kind: str,
) -> dict[str, Any]:
    arm_start = time.perf_counter()
    arm_dir = output_root / loss_kind
    arm_dir.mkdir(parents=True, exist_ok=True)
    if (arm_dir / "failure_receipt.json").exists():
        raise ValueError(
            "failed arm evidence is immutable; use a reviewed corrected attempt/output directory"
        )
    existing = _arm_terminal_receipt(arm_dir, contract_sha)
    if existing is not None:
        return existing
    log_path = arm_dir / "update_diagnostics.jsonl"
    checkpoint_path = arm_dir / "training_checkpoint_latest.npz"
    checkpoint_receipt = arm_dir / "training_checkpoint_latest.json"
    precision = contract["recipe"]["precision"]
    if checkpoint_path.exists() != checkpoint_receipt.exists():
        raise ValueError("incomplete checkpoint pair; refusing an uncertain restart")
    if checkpoint_path.exists():
        params_host, optimizer_host, metadata = _load_checkpoint(
            checkpoint_path, checkpoint_receipt, contract_sha, loss_kind
        )
        updates_completed = int(metadata["updates_completed"])
        epoch_start, batch_start = int(metadata["next_epoch"]), int(metadata["next_batch"])
        if (
            not 0 <= epoch_start <= 2
            or not 0 <= batch_start <= 11
            or updates_completed != epoch_start * 110 + batch_start * 10
        ):
            raise ValueError("checkpoint progress is not at a complete sequence-batch boundary")
        _truncate_jsonl(log_path, updates_completed)
        params = _replicate(params_host, devices)
        optimizer = _replicate(optimizer_host, devices)
    else:
        if log_path.exists() or any(arm_dir.glob("model_epoch1.npz")):
            raise ValueError("partial arm artifacts exist without a committed checkpoint")
        _truncate_jsonl(log_path, 0)
        params_host = {key: jnp.asarray(value) for key, value in initial_params.items()}
        optimizer_host = adamw_init(params_host)
        params = _replicate(params_host, devices)
        optimizer = _replicate(optimizer_host, devices)
        updates_completed, epoch_start, batch_start = 0, 0, 0

    step = _training_step(loss_kind, precision)
    local_batch = 4
    state_dtype = jnp.bfloat16 if precision == "bf16" else jnp.float32
    zero_state = jax.device_put_replicated(
        jnp.zeros((2, local_batch, 256), dtype=state_dtype), devices
    )
    warm_inputs = _shard(np.zeros((32, 2000, 211), dtype=np.float32), devices)
    warm_targets = _shard(np.ones((32, 2000, 2), dtype=np.float32), devices)
    warm_mask = _shard(np.ones((32, 2000), dtype=bool), devices)
    lr_warm = np.full((len(devices),), schedule[0, 0], dtype=np.float32)
    beta_warm = np.full((len(devices),), schedule[0, 1], dtype=np.float32)
    compile_start = time.perf_counter()
    warm_result = step(
        params, optimizer, warm_inputs, zero_state, warm_targets, warm_mask, lr_warm, beta_warm
    )
    jax.block_until_ready(warm_result)
    compile_warmup_seconds = time.perf_counter() - compile_start
    # Discard the warmup update. Restore the exact initial or checkpoint state.
    if updates_completed == 0:
        params = _replicate(
            {key: jnp.asarray(value) for key, value in initial_params.items()}, devices
        )
        optimizer = _replicate(
            adamw_init({key: jnp.asarray(value) for key, value in initial_params.items()}), devices
        )
    else:
        params = _replicate(params_host, devices)
        optimizer = _replicate(optimizer_host, devices)

    fit_start = time.perf_counter()
    batch_count = 11
    global_update = updates_completed
    for epoch in range(epoch_start, 2):
        start_batch = batch_start if epoch == epoch_start else 0
        ids = orders["epochs"][str(epoch)]
        for batch_index in range(start_batch, batch_count):
            batch_ids = ids[batch_index * 32 : (batch_index + 1) * 32]
            prepare_start = time.perf_counter()
            features, targets, _scored, actual_ids = _cache_batch(
                cache, orders["by_id"], batch_ids, "fit_train"
            )
            feature_prepare_seconds = time.perf_counter() - prepare_start
            if len(actual_ids) != (30 if batch_index == 10 else 32):
                raise ValueError("fit batch sizes differ from frozen drop_last=False membership")
            row_index = np.arange(20_000, dtype=np.int32)
            recurrent_state = zero_state
            buffered = []
            batch_update_start = time.perf_counter()
            for chunk_index, chunk_start in enumerate(range(0, 20_000, 2000)):
                update_index = epoch * 110 + batch_index * 10 + chunk_index
                if update_index != global_update or update_index >= len(schedule):
                    raise ValueError("update index differs from the pinned 220-step schedule")
                chunk_stop = chunk_start + 2000
                chunk_x = features[:, chunk_start:chunk_stop]
                chunk_y = targets[:, chunk_start:chunk_stop]
                mask = np.broadcast_to(row_index[chunk_start:chunk_stop] >= 99, (32, 2000)).copy()
                if len(actual_ids) < 32:
                    mask[len(actual_ids) :] = False
                lr, beta1 = schedule[update_index]
                transfer_start = time.perf_counter()
                sharded_x = _shard(chunk_x, devices)
                sharded_y = _shard(chunk_y, devices)
                sharded_mask = _shard(mask, devices)
                transfer_seconds = time.perf_counter() - transfer_start
                step_start = time.perf_counter()
                result = step(
                    params,
                    optimizer,
                    sharded_x,
                    recurrent_state,
                    sharded_y,
                    sharded_mask,
                    np.full((len(devices),), lr, dtype=np.float32),
                    np.full((len(devices),), beta1, dtype=np.float32),
                )
                step_enqueue_seconds = time.perf_counter() - step_start
                params, optimizer, next_state, loss, grad_norm, clip_scale, finite = result
                record = {
                    "attempt_index": update_index,
                    "epoch_index": epoch,
                    "batch_index": batch_index,
                    "chunk_index": chunk_index,
                    "scheduler_step_pre": update_index,
                    "optimizer_groups": [
                        {"group_index": 0, "lr": float(lr), "beta1": float(beta1)}
                    ],
                    "learning_rate": float(lr),
                    "beta1": float(beta1),
                    "clip_threshold": 1.0,
                    "amp_scale_before": None,
                    "amp_scale_after": None,
                    "feature_prepare_seconds_for_sequence_batch": feature_prepare_seconds
                    if chunk_index == 0
                    else 0.0,
                    "host_to_device_enqueue_seconds": transfer_seconds,
                    "forward_backward_update_enqueue_seconds": step_enqueue_seconds,
                }
                buffered.append((record, (loss[0], grad_norm[0], clip_scale[0], finite[0])))
                global_update += 1
                recurrent_state = next_state
            # One synchronization at the whole-sequence boundary. Per-step
            # dispatch latency above is explicitly not accelerator execution time.
            jax.block_until_ready(
                (params, optimizer, recurrent_state, [item[1] for item in buffered])
            )
            batch_update_seconds = time.perf_counter() - batch_update_start
            scalar_rows = jax.device_get([item[1] for item in buffered])
            failure_index = None
            for (record, _), scalars in zip(buffered, scalar_rows, strict=True):
                loss_value, norm_value, scale_value, finite_value = [
                    np.asarray(value).item() for value in scalars
                ]
                finite_value = bool(finite_value)
                record.update(
                    {
                        "loss": float(loss_value) if finite_value else None,
                        "unclipped_global_grad_norm": float(norm_value) if finite_value else None,
                        "clip_scale": float(scale_value) if finite_value else None,
                        "gradient_clipped": bool(scale_value < 1.0) if finite_value else False,
                        "clipped": bool(scale_value < 1.0) if finite_value else False,
                        "finite": finite_value,
                        "completed": finite_value,
                        "skipped": not finite_value,
                        "scaler_skipped": False,
                        "synchronized_sequence_batch_update_seconds": batch_update_seconds
                        if record["chunk_index"] == 0
                        else 0.0,
                    }
                )
                _append_jsonl(log_path, record)
                if not finite_value and failure_index is None:
                    failure_index = record["attempt_index"]
            if failure_index is not None:
                # The last committed checkpoint is at a whole-batch boundary.
                # Saving mid-batch weights under that earlier sampler position
                # would replay already updated chunks on resume.
                _write_json_atomic(
                    arm_dir / "failure_receipt.json",
                    {
                        "contract_sha256": contract_sha,
                        "attempt_index": failure_index,
                        "failure": "nonfinite_update",
                        "resume": "last committed whole-batch checkpoint only",
                    },
                )
                raise FloatingPointError(
                    f"non-finite TPU update at index {failure_index}; fit failed closed"
                )
            next_epoch, next_batch = (
                (epoch + 1, 0) if batch_index == batch_count - 1 else (epoch, batch_index + 1)
            )
            _save_resume_checkpoint(
                arm_dir,
                params,
                optimizer,
                contract_sha=contract_sha,
                loss_kind=loss_kind,
                precision=precision,
                next_epoch=next_epoch,
                next_batch=next_batch,
                updates_completed=global_update,
            )
        batch_start = 0
    if global_update != 220:
        raise ValueError(f"completed {global_update} updates, expected 220")
    fit_seconds = time.perf_counter() - fit_start
    params_host = _first_replica(params)
    final_params = {key: jax.device_get(value) for key, value in params_host.items()}
    model_path = arm_dir / "model_epoch1.npz"
    temporary_model = arm_dir / "model_epoch1.tmp.npz"
    export_npz_state(temporary_model, final_params)
    with temporary_model.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary_model, model_path)

    # Reload from the serialized inference checkpoint before the fixed outer
    # export, so terminal predictions exercise the delivered artifact.
    reload_start = time.perf_counter()
    params_reloaded = load_npz_state(model_path, 211, 256, 2, 2)
    replicated_inference = _replicate(params_reloaded, devices)
    forward = _forward_step(precision)
    jax.block_until_ready(next(iter(params_reloaded.values())))
    reload_seconds = time.perf_counter() - reload_start

    prediction_path = arm_dir / "evaluation_predictions.npy"
    target_path = arm_dir / "evaluation_targets.npy"
    mask_path = arm_dir / "evaluation_scored_mask.npy"
    predictions = np.lib.format.open_memmap(
        prediction_path, mode="w+", dtype=np.float32, shape=(468, 20_000, 2)
    )
    evaluation_targets = np.lib.format.open_memmap(
        target_path, mode="w+", dtype=np.float32, shape=(468, 20_000, 2)
    )
    evaluation_scored = np.lib.format.open_memmap(
        mask_path, mode="w+", dtype=np.bool_, shape=(468, 20_000)
    )
    evaluation_start = time.perf_counter()
    evaluation_ids = orders["evaluation_ids"]
    for batch_index in range(15):
        ids = evaluation_ids[batch_index * 32 : (batch_index + 1) * 32]
        features, targets, scored, actual_ids = _cache_batch(
            cache, orders["by_id"], ids, "evaluation_valid"
        )
        state = jax.device_put_replicated(jnp.zeros((2, 4, 256), dtype=state_dtype), devices)
        batch_predictions = np.zeros((32, 20_000, 2), dtype=np.float32)
        for chunk_start in range(0, 20_000, 2000):
            chunk = features[:, chunk_start : chunk_start + 2000]
            result, state = forward(replicated_inference, _shard(chunk, devices), state)
            jax.block_until_ready(result)
            host = np.asarray(jax.device_get(result)).reshape(32, 2000, 2)
            batch_predictions[:, chunk_start : chunk_start + 2000] = host
        out_start = batch_index * 32
        out_stop = out_start + len(actual_ids)
        predictions[out_start:out_stop] = batch_predictions[: len(actual_ids)]
        evaluation_targets[out_start:out_stop] = targets[: len(actual_ids)]
        evaluation_scored[out_start:out_stop] = scored[: len(actual_ids)]
    predictions.flush()
    evaluation_targets.flush()
    evaluation_scored.flush()
    evaluation_seconds = time.perf_counter() - evaluation_start
    score_mask = np.asarray(evaluation_scored) & (np.arange(20_000)[None, :] >= 99)
    metric = GlobalAccumulator()
    metric.add(np.asarray(evaluation_targets), np.asarray(predictions), score_mask)
    score = metric.result()
    _write_json_atomic(
        arm_dir / "evaluation_sample_ids.json",
        {
            "schema": "wnn.loss-pair-evaluation-sample-ids.v1",
            "role": "evaluation_valid",
            "sequence_ids": evaluation_ids,
            "count": len(evaluation_ids),
        },
    )
    _write_json_atomic(
        arm_dir / "normalization.json",
        {
            "schema": "wnn.loss-pair-normalization.v1",
            "method": "seeded EpochOrderSampler epoch -1, batch 32, first two batches, every seventh flattened row, std clamp 1e-3",
            "source_ids": orders["normalization_ids"],
            "mean": np.asarray(final_params["mean"], dtype=np.float32).tolist(),
            "std": np.asarray(final_params["std"], dtype=np.float32).tolist(),
            "mean_sha256": sha256_bytes(np.asarray(final_params["mean"], dtype="<f4").tobytes()),
            "std_sha256": sha256_bytes(np.asarray(final_params["std"], dtype="<f4").tobytes()),
        },
    )
    diagnostics = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    if len(diagnostics) != 220 or any(not row["completed"] for row in diagnostics):
        raise ValueError("terminal diagnostics do not cover 220 finite completed updates")
    terminal = {
        "schema": RUNNER_SCHEMA,
        "status": "complete",
        "contract_sha256": contract_sha,
        "loss_kind": loss_kind,
        "precision": precision,
        "device_inventory": [
            {
                "platform": device.platform,
                "kind": getattr(device, "device_kind", None),
                "id": str(device),
            }
            for device in devices
        ],
        "accelerator_count": len(devices),
        "fit_sequences": 350,
        "evaluation_sequences": 468,
        "updates_attempted": len(diagnostics),
        "updates_completed": sum(bool(row["completed"]) for row in diagnostics),
        "updates_skipped": sum(bool(row["skipped"]) for row in diagnostics),
        "gradient_clipped_updates": sum(bool(row["gradient_clipped"]) for row in diagnostics),
        "compile_warmup_seconds": compile_warmup_seconds,
        "fit_wall_seconds": fit_seconds,
        "checkpoint_reload_seconds": reload_seconds,
        "evaluation_export_seconds": evaluation_seconds,
        "full_run_wall_seconds": time.perf_counter() - arm_start,
        "official_metric": score,
        "memory": _memory_snapshot(),
        "artifacts": {},
    }
    for artifact in (
        "model_epoch1.npz",
        "training_checkpoint_latest.npz",
        "training_checkpoint_latest.json",
        "update_diagnostics.jsonl",
        "evaluation_predictions.npy",
        "evaluation_targets.npy",
        "evaluation_scored_mask.npy",
        "evaluation_sample_ids.json",
        "normalization.json",
    ):
        terminal["artifacts"][artifact] = sha256_file(arm_dir / artifact)
    _write_json_atomic(arm_dir / "terminal_receipt.json", terminal)
    return terminal


def _pair_analysis(output_root: Path, contract_sha: str) -> dict[str, Any]:
    wmse_dir, huber_dir = output_root / "wmse", output_root / "huber"
    wmse = np.load(wmse_dir / "evaluation_predictions.npy", mmap_mode="r", allow_pickle=False)
    huber = np.load(huber_dir / "evaluation_predictions.npy", mmap_mode="r", allow_pickle=False)
    targets = np.load(wmse_dir / "evaluation_targets.npy", mmap_mode="r", allow_pickle=False)
    mask = np.load(wmse_dir / "evaluation_scored_mask.npy", mmap_mode="r", allow_pickle=False)
    if not np.array_equal(
        mask, np.load(huber_dir / "evaluation_scored_mask.npy", mmap_mode="r", allow_pickle=False)
    ):
        raise ValueError("paired arms have different fixed evaluation masks")
    use = mask & (np.arange(20_000)[None, :] >= 99)
    blend = 0.5 * (np.asarray(wmse) + np.asarray(huber))
    metric = GlobalAccumulator()
    metric.add(np.asarray(targets), blend, use)
    correlations = []
    for channel in range(2):
        left, right = np.asarray(wmse[..., channel][use]), np.asarray(huber[..., channel][use])
        value = (
            float(np.corrcoef(left, right)[0, 1])
            if left.size > 1 and np.std(left) > 0 and np.std(right) > 0
            else None
        )
        correlations.append(value if value is None or np.isfinite(value) else None)
    result = {
        "schema": "wnn.loss-pair-comparison.v1",
        "contract_sha256": contract_sha,
        "exposure": "fold-0 development evaluation; not fresh-holdout or public evidence",
        "prediction_correlation_by_target": correlations,
        "equal_weight_blend_global_wp_clip2_v1": metric.result(),
        "wmse_metric": _read_json(wmse_dir / "terminal_receipt.json")["official_metric"],
        "huber_metric": _read_json(huber_dir / "terminal_receipt.json")["official_metric"],
    }
    _write_json_atomic(output_root / "paired_comparison.json", result)
    result["artifact_sha256"] = sha256_file(output_root / "paired_comparison.json")
    return result


def run(contract_path: Path, output_root: Path) -> dict[str, Any]:
    run_start = time.perf_counter()
    jax.config.update("jax_default_matmul_precision", "highest")
    contract, contract_sha = _validate_contract(contract_path.resolve(strict=True))
    output_root.mkdir(parents=True, exist_ok=True)
    run_receipt_path = output_root / "run_receipt.json"
    if run_receipt_path.exists():
        receipt = _read_json(run_receipt_path)
        if receipt.get("contract_sha256") != contract_sha:
            raise ValueError("output directory is already bound to a different contract")
    else:
        if any(output_root.iterdir()):
            raise ValueError(
                "output directory has artifacts but no run receipt; refusing uncertain reuse"
            )
        _write_json_atomic(
            run_receipt_path,
            {
                "schema": "wnn.loss-pair-tpu-run-receipt.v1",
                "contract_sha256": contract_sha,
                "started_at_unix": time.time(),
                "runner_sha256": sha256_file(Path(__file__).resolve()),
            },
        )
    devices = [device for device in jax.devices() if device.platform == "tpu"]
    if len(devices) != 8:
        raise RuntimeError(
            f"expected one TPU v5e-8 pod with 8 visible devices; observed {len(devices)}"
        )
    setup_start = time.perf_counter()
    artifact_paths = {name: _verify_artifact(contract, name) for name in contract["artifacts"]}
    parity = _read_json(artifact_paths["device_parity_receipt"])
    required_checks = {
        "forward",
        "gradient",
        "adamw_update",
        "reload",
        "reset",
        "prefix_causality",
        "replica_reduction",
        "full_shape_profile",
    }
    if (
        parity.get("status") != "pass"
        or parity.get("platform") != "tpu"
        or parity.get("accelerator_count") != len(devices)
        or set(parity.get("passed_checks", [])) != required_checks
    ):
        raise ValueError("actual-device lifecycle and full-shape parity evidence is incomplete")
    if parity.get("jax_step_sha256") != sha256_file(Path(__file__).with_name("jax_step.py")):
        raise ValueError("device parity does not bind this update implementation")
    if parity.get("initial_state_sha256") != contract["artifacts"]["initial_state"]["sha256"]:
        raise ValueError("device profile does not bind the shared initialized model")
    if (
        not parity.get("actual_data_profile")
        or parity.get("profile_inputs_receipt", {}).get("cache_manifest_sha256")
        != contract["artifacts"]["cache_manifest"]["sha256"]
    ):
        raise ValueError("actual device profile is not bound to this parsed-cache producer")
    cache = ParsedSequenceCache.open(
        artifact_paths["cache_manifest"],
        expected_manifest_sha256=contract["artifacts"]["cache_manifest"]["sha256"],
        validation_receipt_path=artifact_paths["cache_validation_receipt"],
        expected_validation_receipt_sha256=contract["artifacts"]["cache_validation_receipt"][
            "sha256"
        ],
        expected_cache_key=contract["identity"]["cache_key"],
        expected_split_manifest_sha256=contract["identity"]["split_manifest_sha256"],
        expected_fold=0,
    )
    if cache.manifest["membership_artifact"]["sha256"] != contract["identity"]["membership_sha256"]:
        raise ValueError("parsed-cache membership hash differs from root's fold-0 pin")
    if (
        cache.manifest["portable_registry_sha256"]
        != contract["identity"]["portable_registry_sha256"]
    ):
        raise ValueError("parsed-cache portable registry differs from root's pin")
    actual_counts = cache.manifest["member_counts"]
    expected_counts = {"fit_train": 350, "fit_valid": 28, "evaluation_valid": 468}
    if actual_counts != expected_counts:
        raise ValueError("parsed-cache member counts differ from the frozen pair recipe")
    orders = _read_orders(artifact_paths["sample_orders"], cache.membership)
    schedule = _read_onecycle_trace(artifact_paths["onecycle_trace"])
    initial_params = load_npz_state(artifact_paths["initial_state"], 211, 256, 2, 2)
    setup_seconds = time.perf_counter() - setup_start
    terminal = {}
    for loss_kind in LOSS_KINDS:
        terminal[loss_kind] = _run_arm(
            cache=cache,
            orders=orders,
            initial_params=initial_params,
            schedule=schedule,
            contract=contract,
            contract_sha=contract_sha,
            output_root=output_root,
            devices=devices,
            loss_kind=loss_kind,
        )
    comparison = _pair_analysis(output_root, contract_sha)
    cache.close()
    final_receipt = {
        "schema": RUNNER_SCHEMA,
        "status": "complete",
        "contract_sha256": contract_sha,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "cache_manifest_sha256": contract["artifacts"]["cache_manifest"]["sha256"],
        "cache_validation_receipt_sha256": contract["artifacts"]["cache_validation_receipt"][
            "sha256"
        ],
        "initial_state_sha256": contract["artifacts"]["initial_state"]["sha256"],
        "onecycle_trace_sha256": contract["artifacts"]["onecycle_trace"]["sha256"],
        "sample_orders_sha256": contract["artifacts"]["sample_orders"]["sha256"],
        "device_inventory": terminal["wmse"]["device_inventory"],
        "cache_validation_and_load_seconds": setup_seconds,
        "full_worker_wall_seconds": time.perf_counter() - run_start,
        "paired_comparison": comparison,
    }
    _write_json_atomic(output_root / "terminal_receipt.json", final_receipt)
    return final_receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.contract, args.output_root)
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
