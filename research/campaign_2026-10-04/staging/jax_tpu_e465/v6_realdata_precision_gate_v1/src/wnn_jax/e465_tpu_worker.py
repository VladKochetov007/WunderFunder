"""Full-shape, fold-0 E465 AdamW continuation worker for an eight-core TPU."""

from __future__ import annotations

import hashlib
import json
import math
import os
import resource
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pyarrow.parquet as pq
from jax import lax
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from wnn_jax import e465_gate
from wnn_jax.bridge import (
    ComboTeacherMean,
    FoldGroups,
    PropensityCache,
    RawSequenceStore,
    collate_evaluation_sequences,
    collate_sequences,
    group_batch_sha256,
    load_evaluation_sequence,
    load_fold_groups,
    load_sequence_supervision,
    seeded_group_batches,
    sha256_file,
)
from wnn_jax.distributed_combo import (
    global_combo_loss_and_local_surrogate,
    pmean_and_clip_gradient,
)
from wnn_jax.e465_host_contract import validate_epoch_batches, validate_sequence_domain
from wnn_jax.onecycle import onecycle_point
from wnn_jax.rnn import (
    AdamWState,
    adamw_init,
    adamw_update,
    export_npz_state,
    forward_chunk,
    load_npz_state,
)

TPU_DEVICE_COUNT = 8
GLOBAL_BATCH_SIZE = 32
LOCAL_BATCH_SIZE = 4
CHUNK_ROWS = 1000
SEQUENCE_ROWS = 20_000
CHUNKS_PER_BATCH = SEQUENCE_ROWS // CHUNK_ROWS
FIT_TRAIN_GROUPS = 10_535
FIT_VALID_GROUPS = 1_405
EVALUATION_GROUPS = 468
ALL_VALID_GROUPS = 1_873
FIT_GROUPS = FIT_TRAIN_GROUPS + FIT_VALID_GROUPS
FULL_BATCHES = FIT_GROUPS // GLOBAL_BATCH_SIZE
DROPPED_GROUPS = FIT_GROUPS % GLOBAL_BATCH_SIZE
UPDATES_PER_ARM = FULL_BATCHES * CHUNKS_PER_BATCH
MAX_HOST_RSS_BYTES = 12 * 1024**3
ALLOCATOR_FRACTION = 0.80
DEVICE_HARD_STOP_FRACTION = 0.90
MAX_WALL_SECONDS = 8 * 60 * 60
EMA_DECAY_PER_UPDATE = 0.999
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
TEACHERS = ("c20260929_since_s3_f0", "c20260929_since_s4_f0")
PROPENSITY_RUN = "c20260929_prop_f0"
EXPECTED_PARENT_RUN = "c20261003_combo_s3_f0"
EXPECTED_PARENT_DIMENSIONS = {"n_in": 211, "hidden": 256, "layers": 2, "n_out": 23}


@dataclass(frozen=True)
class RuntimeContext:
    """Verified input identities and reusable host-side cache readers."""

    root: Path
    input_manifest_path: Path
    input_manifest_sha256: str
    input_file_hashes: dict[str, str]
    fold: FoldGroups
    initial_params: dict[str, jax.Array]
    source_hashes: dict[str, str]
    store: RawSequenceStore
    teachers: dict[str, ComboTeacherMean]
    propensity: dict[str, PropensityCache]
    device_inventory: list[dict[str, Any]]
    mapped_sharding: NamedSharding


def _json_bytes(value: object) -> bytes:
    """Return deterministic finite JSON bytes for provenance records."""
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def _write_json(path: Path, value: object) -> None:
    """Write one complete provenance record without overwriting an artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_bytes(_json_bytes(value))
    temporary.replace(path)


def _sha256_bytes(value: bytes) -> str:
    """Hash in-memory JSON or mask bytes."""
    return hashlib.sha256(value).hexdigest()


def _file_entry_map(manifest: dict[str, Any]) -> dict[str, str]:
    """Validate the input-file table and return unique relative path hashes."""
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("input manifest needs a non-empty files list")
    found: dict[str, str] = {}
    for item in entries:
        if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
            raise ValueError("each input file entry must contain only path and sha256")
        relative = str(item["path"])
        digest = str(item["sha256"])
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError(f"input file path must stay relative to project root: {relative}")
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError(f"invalid SHA-256 for {relative}")
        if relative in found:
            raise ValueError(f"duplicate input file entry: {relative}")
        found[relative] = digest
    return found


def _required_input_paths(manifest: dict[str, Any]) -> set[str]:
    """Build the required raw/cache closure from the fixed objective lineage."""
    parent = manifest.get("parent_export", {})
    fold_path = str(manifest.get("fold_manifest", ""))
    required = {
        fold_path,
        "datasets/train.parquet",
        "datasets/valid.parquet",
        "cache/train_targets.npz",
        "cache/valid_targets.npz",
        str(parent.get("npz_path", "")),
        str(parent.get("sidecar_path", "")),
        str(parent.get("source_checkpoint_path", "")),
        str(parent.get("source_provenance_path", "")),
    }
    for run in TEACHERS:
        required.add(f"artifacts/runs/{run}/model_ep3.pt")
        for split in ("train", "valid"):
            base = f"cache/pred_{run}_{split}"
            required.update((base + ".npy", base + ".json"))
    required.add(f"artifacts/runs/{PROPENSITY_RUN}/model.pt")
    for split in ("train", "valid"):
        base = f"cache/propcache_{PROPENSITY_RUN}_{split}"
        required.update((base + ".npy", base + ".json"))
    return required


def _verify_input_manifest(
    root: Path, manifest_path: Path
) -> tuple[dict[str, Any], dict[str, str]]:
    """Hash the complete data/cache closure and enforce the fixed parent lineage."""
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    if manifest.get("format") != "wnn.e465-tpu-input-manifest" or manifest.get("version") != 1:
        raise ValueError("unsupported E465 input manifest format")
    if tuple(manifest.get("teacher_runs", ())) != TEACHERS:
        raise ValueError("input manifest must bind the registered pair of fold-0 SBD teachers")
    if manifest.get("propensity_run") != PROPENSITY_RUN:
        raise ValueError("input manifest must bind the registered fold-0 propensity cache")
    files = _file_entry_map(manifest)
    missing = _required_input_paths(manifest) - set(files)
    if missing:
        raise ValueError(f"input manifest is missing required files: {sorted(missing)}")
    root_resolved = root.resolve(strict=True)
    for relative, expected in files.items():
        candidate = (root_resolved / relative).resolve(strict=True)
        if not candidate.is_relative_to(root_resolved) or not candidate.is_file():
            raise ValueError(f"input path is outside project root or not a file: {relative}")
        actual = sha256_file(candidate)
        if actual != expected:
            raise ValueError(f"input SHA-256 changed for {relative}: {actual}")

    parent = manifest["parent_export"]
    if manifest.get("fold_manifest_sha256") != files[manifest["fold_manifest"]]:
        raise ValueError("fold manifest SHA-256 differs from the input file table")
    if parent.get("run") != EXPECTED_PARENT_RUN or int(parent.get("epoch", -1)) != 3:
        raise ValueError("parent must be the fixed fold-0 combo S3 epoch-3 checkpoint")
    if parent.get("weight_source") != "evaluation_ema":
        raise ValueError("parent export must contain the fixed epoch-3 evaluation EMA")
    if parent.get("dimensions") != EXPECTED_PARENT_DIMENSIONS:
        raise ValueError("parent model dimensions differ from the 211/256/2/23 contract")
    if (
        parent.get("npz_sha256") != files[str(parent["npz_path"])]
        or parent.get("sidecar_sha256") != files[str(parent["sidecar_path"])]
        or parent.get("source_checkpoint_sha256") != files[str(parent["source_checkpoint_path"])]
        or parent.get("source_provenance_sha256") != files[str(parent["source_provenance_path"])]
    ):
        raise ValueError("parent export descriptors differ from the input file table")
    npz_sidecar_path = root_resolved / str(parent["sidecar_path"])
    sidecar = json.loads(npz_sidecar_path.read_text(encoding="utf-8"))
    if sidecar.get("format") != "wnn.jax-npz-export" or sidecar.get("source_epoch") != 3:
        raise ValueError("parent NPZ sidecar format or epoch differs")
    if sidecar.get("dimensions") != EXPECTED_PARENT_DIMENSIONS:
        raise ValueError("parent NPZ sidecar dimensions differ")
    if sidecar.get("npz_sha256") != files[str(parent["npz_path"])]:
        raise ValueError("parent NPZ sidecar hash differs from manifest")
    if sidecar.get("source_checkpoint_sha256") != files[str(parent["source_checkpoint_path"])]:
        raise ValueError("parent checkpoint hash differs from export sidecar")
    if sidecar.get("source_checkpoint") != parent["source_checkpoint_path"]:
        raise ValueError("parent checkpoint path differs from NPZ export sidecar")
    if sidecar.get("npz_path") != parent["npz_path"]:
        raise ValueError("parent NPZ path differs from its sidecar")
    provenance_path = root_resolved / str(parent["source_provenance_path"])
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    args = provenance.get("args", {})
    expected_args = {
        "name": EXPECTED_PARENT_RUN,
        "checkpoint_policy": "fixed",
        "checkpoint_epoch": 3,
        "no_eval_during_fit": True,
        "eval_ema": True,
        "split_manifest_sha256": manifest["fold_manifest_sha256"],
        "train_source": "train+valid",
        "loss": "wmse",
        "aux_struct": 0.3,
        "aux_next": 0.3,
        "distill": ",".join(TEACHERS),
        "distill_alpha": 0.5,
        "distill_boot_h": 8,
        "propensity": PROPENSITY_RUN,
        "prop_power": 2.0,
        "prop_eps": 0.03,
        "valid_true_mask": 0.03,
        "since_weight": 32.0,
        "since_floor": 0.25,
        "batch": GLOBAL_BATCH_SIZE,
        "chunk": CHUNK_ROWS,
        "hidden": 256,
        "layers": 2,
        "dropout": 0.0,
        "arch": "gru",
        "cell": "gru",
        "enc": "linear",
        "ema_teacher": 1.0,
        "ema_decay": EMA_DECAY_PER_UPDATE,
    }
    if any(args.get(name) != value for name, value in expected_args.items()):
        raise ValueError("parent provenance differs from the registered combo training recipe")
    if provenance.get("selected_epoch") != 3:
        raise ValueError("parent checkpoint provenance did not select fixed epoch 3")
    if (
        provenance.get("checkpoint_sha256", {}).get("model_ep3.pt")
        != files[str(parent["source_checkpoint_path"])]
    ):
        raise ValueError("parent provenance checkpoint hash differs from the input file table")
    teacher_entries = provenance.get("args", {}).get("distill_provenance", [])
    expected_teacher_entries = {
        (run, split): {
            "run": run,
            "split": split,
            "epoch": 3,
            "checkpoint_sha256": files[f"artifacts/runs/{run}/model_ep3.pt"],
            "prediction_sha256": files[f"cache/pred_{run}_{split}.npy"],
        }
        for run in TEACHERS
        for split in ("train", "valid")
    }
    actual_teacher_entries = {
        (item.get("run"), item.get("split")): {
            key: item.get(key)
            for key in (
                "run",
                "split",
                "epoch",
                "checkpoint_sha256",
                "prediction_sha256",
            )
        }
        for item in teacher_entries
    }
    if (
        len(teacher_entries) != len(expected_teacher_entries)
        or actual_teacher_entries != expected_teacher_entries
    ):
        raise ValueError("parent provenance SBD teacher hashes differ from the input cache closure")
    if (
        provenance.get("propensity_checkpoint_sha256")
        != files[f"artifacts/runs/{PROPENSITY_RUN}/model.pt"]
    ):
        raise ValueError("parent provenance propensity checkpoint hash differs")
    return manifest, files


def _code_hashes(stage_root: Path) -> dict[str, str]:
    """Hash the runner, V6 gate, and every staged module used by this attempt."""
    return e465_gate.source_hashes(stage_root)


def _host_rss_bytes() -> tuple[int, int]:
    """Return current and high-water resident memory from Linux procfs."""
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/self/status").read_text(encoding="ascii").splitlines():
            name, separator, tail = line.partition(":")
            if separator and name in {"VmRSS", "VmHWM"}:
                values[name] = int(tail.strip().split()[0]) * 1024
    except (OSError, ValueError, IndexError):
        values = {}
    current = values.get("VmRSS", 0)
    peak = values.get("VmHWM", int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024)
    return current, peak


def _device_memory(devices: list[jax.Device]) -> list[dict[str, Any]]:
    """Read per-device allocator telemetry and enforce the 90% HBM hard stop."""
    records: list[dict[str, Any]] = []
    for device in devices:
        raw_stats = device.memory_stats()
        if not isinstance(raw_stats, dict):
            raise RuntimeError(f"TPU {device} memory_stats is unavailable")
        stats = {str(name): int(value) for name, value in raw_stats.items()}
        limit = stats.get("bytes_limit")
        if not limit or limit < 1:
            raise RuntimeError(f"TPU {device} has no positive bytes_limit: {stats}")
        observed = max(stats.get("bytes_in_use", 0), stats.get("peak_bytes_in_use", 0))
        stop_bytes = int(limit * DEVICE_HARD_STOP_FRACTION)
        if observed >= stop_bytes:
            raise MemoryError(f"TPU {device} allocator reached hard stop: {observed}/{stop_bytes}")
        records.append(
            {
                "device": str(device),
                "memory_stats": stats,
                "allocator_target_fraction": ALLOCATOR_FRACTION,
                "hard_stop_fraction": DEVICE_HARD_STOP_FRACTION,
                "observed_bytes": observed,
                "hard_stop_bytes": stop_bytes,
            }
        )
    return records


def _guard_resources(devices: list[jax.Device]) -> dict[str, Any]:
    """Enforce host RSS and accelerator headroom while recording telemetry."""
    current_rss, peak_rss = _host_rss_bytes()
    if max(current_rss, peak_rss) >= MAX_HOST_RSS_BYTES:
        raise MemoryError(f"host RSS reached hard stop: {max(current_rss, peak_rss)}")
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    if len(affinity) > 8:
        os.sched_setaffinity(0, set(affinity[:8]))
        affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) < 8:
        raise RuntimeError(f"worker needs eight available host CPUs, found {affinity}")
    return {
        "host_rss_current_bytes": current_rss,
        "host_rss_peak_bytes": peak_rss,
        "host_rss_stop_bytes": MAX_HOST_RSS_BYTES,
        "cpu_affinity": affinity,
        "cpu_count": len(affinity),
        "devices": _device_memory(devices),
    }


def _validate_preflight(
    bundle_path: Path,
    input_manifest_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    """Consume the exact terminal V6 output collection and E465 input identity."""
    return e465_gate.validate_e463_v6_bundle(
        bundle_path,
        input_manifest_path,
        project_root,
    )


def _validate_preflight_inputs(
    report: dict[str, Any],
    context: RuntimeContext,
) -> None:
    """Recheck V6-bound raw, fold, and parent identities after hashing E465 inputs."""
    identity = report.get("v6_input_identity", {})
    raw_hashes = identity.get("raw_data_sha256", {})
    if (
        report.get("status") != "passed"
        or identity.get("input_manifest_sha256") != context.input_manifest_sha256
        or identity.get("fold_manifest_sha256") != context.fold.manifest_sha256
        or identity.get("parent_checkpoint_sha256")
        != "d4a1e50b1892b4d9edd7c255908b68073a760c059ca80223e98a63c7e023371e"
        or identity.get("parent_npz_sha256")
        != "677f604fd86d993c3d4a66a10638755253bfe52f1f578009dbd781c73fe6d04a"
        or not isinstance(raw_hashes, dict)
        or any(
            context.input_file_hashes.get(f"datasets/{split}.parquet") != raw_hashes.get(split)
            for split in ("train", "valid")
        )
    ):
        raise ValueError("E463 V6 receipt differs from fully hashed E465 input identities")


def _validate_profile(
    path: Path,
    *,
    input_manifest_sha256: str,
    source_hashes: dict[str, str],
    preflight_identity: dict[str, Any],
) -> dict[str, Any]:
    """Require an exact source/data-bound full-shape actual profile."""
    report = json.loads(path.read_text(encoding="utf-8"))
    return e465_gate.validate_profile_report(
        report,
        input_manifest_sha256=input_manifest_sha256,
        source_identity=source_hashes,
        preflight_identity=preflight_identity,
    )


def _profile_precision_gate(
    arm_reports: dict[str, dict[str, Any]],
    histories: dict[str, list[dict[str, object]]],
) -> dict[str, Any]:
    """Summarize finite actual-data updates and retain paired FP32/BF16 deltas."""
    if set(arm_reports) != {"fp32", "bf16"} or set(histories) != {"fp32", "bf16"}:
        raise ValueError("actual profile must execute both FP32 and BF16 arms")
    fp32_hashes = arm_reports["fp32"]["batch_data_sha256"]
    bf16_hashes = arm_reports["bf16"]["batch_data_sha256"]
    if fp32_hashes != bf16_hashes or len(fp32_hashes) != 2:
        raise ValueError("FP32 and BF16 profiles did not consume the same two actual host batches")
    diagnostics: list[dict[str, float | int]] = []
    for update_index, (fp32, bf16) in enumerate(
        zip(histories["fp32"], histories["bf16"], strict=True)
    ):
        fp32_loss = float(fp32["loss"])
        bf16_loss = float(bf16["loss"])
        fp32_norm = float(fp32["gradient_norm_before_clip"])
        bf16_norm = float(bf16["gradient_norm_before_clip"])
        loss_delta = abs(fp32_loss - bf16_loss)
        norm_delta = abs(fp32_norm - bf16_norm)
        diagnostics.append(
            {
                "update_index": update_index,
                "fp32_loss": fp32_loss,
                "bf16_loss": bf16_loss,
                "fp32_gradient_norm": fp32_norm,
                "bf16_gradient_norm": bf16_norm,
                "loss_abs_delta": loss_delta,
                "loss_relative_delta": loss_delta / max(abs(fp32_loss), 1e-8),
                "gradient_norm_abs_delta": norm_delta,
                "gradient_norm_relative_delta": norm_delta / max(abs(fp32_norm), 1e-8),
            }
        )
    gates = {}
    for precision in ("fp32", "bf16"):
        arm = arm_reports[precision]
        history = histories[precision]
        gates[precision] = {
            "precision": precision,
            "actual_data": True,
            "batch_count": arm["batch_count"],
            "attempted_updates": arm["attempt_count"],
            "completed_updates": arm["completed_update_count"],
            "finite_loss_gradient_checks": len(history),
            "final_params_optimizer_ema_and_recurrent_states_finite": arm[
                "final_state_finite"
            ],
            "batch_data_sha256": arm["batch_data_sha256"],
        }
    return {
        "schema": "wnn.e465-actual-data-precision-gate.v1",
        "status": "passed",
        "profile_batches_per_precision": 2,
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "sequence_rows": SEQUENCE_ROWS,
        "chunk_rows": CHUNK_ROWS,
        "chunks_per_batch": CHUNKS_PER_BATCH,
        "arms": gates,
        "paired_step_diagnostics": diagnostics,
        "numerical_review_required_before_fit": True,
        "interpretation": (
            "Actual-data finite/update checks and paired diagnostics only; no precision parity "
            "claim or fit admission without a root review bound to this profile."
        ),
    }

def _batch_sha256(batch: dict[str, np.ndarray]) -> str:
    """Hash sequence supervision with no second full-batch byte-string copy."""
    digest = hashlib.sha256()
    for name in sorted(batch):
        value = np.asarray(batch[name])
        if not value.flags.c_contiguous:
            raise ValueError(f"host batch array must be contiguous: {name}")
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(_json_bytes(list(value.shape)))
        digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def _load_host_batch(
    groups: tuple[tuple[str, int], ...],
    context: RuntimeContext,
) -> dict[str, np.ndarray]:
    """Read and collate exactly one full global batch of fixed supervision."""
    samples = [
        load_sequence_supervision(
            context.root,
            split,
            row_group,
            context.teachers[split],
            context.propensity[split],
            context.store,
            groups=context.fold,
            alpha=0.5,
            horizon=8,
        )
        for split, row_group in groups
    ]
    if len(samples) != GLOBAL_BATCH_SIZE:
        raise ValueError("fit input loader must produce a complete global batch of 32")
    result = collate_sequences(samples)
    del samples
    return result


def _iter_prefetched_batches(
    groups: tuple[tuple[tuple[str, int], ...], ...], context: RuntimeContext
):
    """Yield one batch while a single following batch is built on the host."""
    if not groups:
        raise ValueError("no full fit batches are available")
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="e465-batch-prefetch") as pool:
        pending: Future[dict[str, np.ndarray]] = pool.submit(_load_host_batch, groups[0], context)
        for index, row_groups in enumerate(groups):
            current = pending.result()
            if index + 1 < len(groups):
                pending = pool.submit(_load_host_batch, groups[index + 1], context)
            yield index, row_groups, current


def _shard_chunk(
    batch: dict[str, np.ndarray],
    start: int,
    stop: int,
    sharding: NamedSharding,
) -> dict[str, jax.Array]:
    """Convert a contiguous host chunk to the eight mapped `[4,time,...]` shards."""
    shards: dict[str, jax.Array] = {}
    for name, host_value in batch.items():
        values = np.asarray(host_value[:, start:stop])
        values = values.reshape((TPU_DEVICE_COUNT, LOCAL_BATCH_SIZE, *values.shape[1:]))
        shards[name] = jax.device_put(values, sharding)
    return shards


def _mapped_sharding(devices: list[jax.Device]) -> NamedSharding:
    """Create a leading-replica-axis sharding for the pmap input contract."""
    mesh = Mesh(np.asarray(devices, dtype=object), ("replicas",))
    return NamedSharding(mesh, PartitionSpec("replicas"))


def _replicate(tree: Any, devices: list[jax.Device], sharding: NamedSharding) -> Any:
    """Stack one pytree per replica and place it on an explicit device mesh."""
    return jax.tree_util.tree_map(
        lambda value: jax.device_put(
            jnp.broadcast_to(value, (len(devices), *value.shape)), sharding
        ),
        tree,
    )


def _make_train_step(precision: str, devices: list[jax.Device]):
    """Construct the globally pooled chunk update for one compute precision."""

    def replica_step(
        params: dict[str, jax.Array],
        optimizer: AdamWState,
        ema_params: dict[str, jax.Array],
        student_state: jax.Array,
        ema_state: jax.Array,
        batch: dict[str, jax.Array],
        learning_rate: jax.Array,
        beta1: jax.Array,
    ):
        def objective(current: dict[str, jax.Array]):
            prediction, next_student_state = forward_chunk(
                current, batch["inputs"], student_state, precision=precision
            )
            ema_prediction, next_ema_state = forward_chunk(
                ema_params, batch["inputs"], ema_state, precision=precision
            )
            pooled, local_surrogate = global_combo_loss_and_local_surrogate(
                prediction,
                batch,
                ema_prediction,
                axis_name="replicas",
                replica_count=TPU_DEVICE_COUNT,
            )
            return local_surrogate, (pooled, next_student_state, next_ema_state)

        (_, (pooled_loss, next_student_state, next_ema_state)), gradients = jax.value_and_grad(
            objective, has_aux=True
        )(params)
        gradients, gradient_norm = pmean_and_clip_gradient(
            gradients, axis_name="replicas", max_norm=MAX_GRAD_NORM
        )
        next_params, next_optimizer = adamw_update(
            params,
            gradients,
            optimizer,
            learning_rate=learning_rate,
            weight_decay=WEIGHT_DECAY,
            beta1=beta1,
            beta2=0.999,
        )
        next_ema_params = {
            name: value
            if name in {"mean", "std"}
            else EMA_DECAY_PER_UPDATE * ema_params[name] + (1.0 - EMA_DECAY_PER_UPDATE) * value
            for name, value in next_params.items()
        }
        return (
            next_params,
            next_optimizer,
            next_ema_params,
            lax.stop_gradient(next_student_state),
            lax.stop_gradient(next_ema_state),
            pooled_loss,
            gradient_norm,
            next_optimizer.count,
        )

    return jax.pmap(
        replica_step,
        axis_name="replicas",
        devices=devices,
        in_axes=(0, 0, 0, 0, 0, 0, None, None),
    )


def _first_replica(tree: Any) -> Any:
    """Remove a leading replicated-device dimension from a pytree."""
    return jax.tree_util.tree_map(lambda value: value[0], tree)


def _all_finite(tree: Any) -> bool:
    """Synchronously reject non-finite parameters, moments, or metrics."""
    leaves = jax.tree_util.tree_leaves(tree)
    return all(bool(np.isfinite(np.asarray(jax.device_get(value))).all()) for value in leaves)


def _prepare_context(
    *,
    root: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
    input_hashes: dict[str, str],
    stage_root: Path,
    devices: list[jax.Device],
) -> RuntimeContext:
    """Validate manifest/cache identities and initialize fixed parent parameters."""
    fold_path = root / manifest["fold_manifest"]
    fold = load_fold_groups(fold_path)
    if manifest.get("fold_manifest_sha256") != fold.manifest_sha256:
        raise ValueError("fold manifest hash differs from input manifest")
    if (len(fold.train), len(fold.valid_fit), len(fold.evaluation)) != (
        FIT_TRAIN_GROUPS,
        FIT_VALID_GROUPS,
        EVALUATION_GROUPS,
    ):
        raise ValueError("fold group counts differ from the registered 10535/1405/468 split")
    if set(fold.train) & {528}:
        raise ValueError("corrupt TRAIN row group 528 must be excluded before fit routing")
    if set(fold.valid_fit) & set(fold.evaluation):
        raise ValueError("fit VALID and evaluation VALID overlap")
    provenance_path = root / manifest["parent_export"]["source_provenance_path"]
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    parent_fit = provenance.get("fit_row_groups", {})
    if set(map(int, parent_fit.get("train", []))) != set(fold.train):
        raise ValueError("parent TRAIN fit membership differs from current fold-0 manifest")
    if set(map(int, parent_fit.get("valid", []))) != set(fold.valid_fit):
        raise ValueError("parent fit-VALID membership differs from current fold-0 manifest")
    if set(map(int, provenance.get("evaluation_row_groups", []))) != set(fold.evaluation):
        raise ValueError("parent evaluation membership differs from current fold-0 manifest")
    parquet_counts = {
        split: pq.ParquetFile(root / "datasets" / f"{split}.parquet").num_row_groups
        for split in ("train", "valid")
    }
    if parquet_counts["valid"] != ALL_VALID_GROUPS:
        raise ValueError(
            f"expected 1873 complete VALID row groups, found {parquet_counts['valid']}"
        )
    with np.load(root / "cache/valid_targets.npz", allow_pickle=False) as target_index:
        valid_target_groups = tuple(map(int, target_index["row_group"]))
    if valid_target_groups != tuple(range(ALL_VALID_GROUPS)):
        raise ValueError("VALID prediction output order must match the typed 1873-row target cache")
    if any(group < 0 or group >= parquet_counts["train"] for group in fold.train):
        raise ValueError("fold TRAIN group is outside train.parquet")
    if any(
        group < 0 or group >= parquet_counts["valid"] for group in fold.valid_fit + fold.evaluation
    ):
        raise ValueError("fold VALID group is outside valid.parquet")
    parent = manifest["parent_export"]
    initial_params = load_npz_state(root / parent["npz_path"], 211, 256, 2, 23)
    return RuntimeContext(
        root=root.resolve(strict=True),
        input_manifest_path=manifest_path.resolve(strict=True),
        input_manifest_sha256=sha256_file(manifest_path),
        input_file_hashes=input_hashes,
        fold=fold,
        initial_params=initial_params,
        source_hashes=_code_hashes(stage_root),
        store=RawSequenceStore(root),
        teachers={
            split: ComboTeacherMean(root, split, fold.manifest_sha256)
            for split in ("train", "valid")
        },
        propensity={
            split: PropensityCache(root, split, PROPENSITY_RUN) for split in ("train", "valid")
        },
        device_inventory=[
            {"device": str(device), "kind": str(getattr(device, "device_kind", "unknown"))}
            for device in devices
        ],
        mapped_sharding=_mapped_sharding(devices),
    )


def _runtime_setup() -> list[jax.Device]:
    """Select and validate the exact eight-core TPU backend before work starts."""
    devices = jax.devices("tpu")
    if len(devices) != TPU_DEVICE_COUNT:
        raise RuntimeError(f"E465 requires exactly eight TPU devices, found {devices}")
    if any("TPU v5 lite" not in str(getattr(device, "device_kind", "")) for device in devices):
        raise RuntimeError(f"E465 requires TPU v5 lite devices, found {devices}")
    jax.config.update("jax_default_matmul_precision", "highest")
    return devices


def _new_training_state(
    params: dict[str, jax.Array],
    precision: str,
    devices: list[jax.Device],
    sharding: NamedSharding,
) -> tuple[Any, AdamWState, Any, jax.Array, jax.Array]:
    """Create identical parent student/EMA params and fresh zeroed AdamW state."""
    student = _replicate(params, devices, sharding)
    ema = _replicate(params, devices, sharding)
    optimizer = _replicate(adamw_init(params), devices, sharding)
    student_state, ema_state = _zero_recurrent_states(precision, sharding)
    return student, optimizer, ema, student_state, ema_state


def _zero_recurrent_states(precision: str, sharding: NamedSharding) -> tuple[jax.Array, jax.Array]:
    """Reset student and EMA carries at the boundary between independent sequences."""
    dtype = jnp.float32 if precision == "fp32" else jnp.bfloat16
    shape = (2, LOCAL_BATCH_SIZE, 256)
    mapped_shape = (TPU_DEVICE_COUNT, *shape)
    student = jax.device_put(jnp.zeros(mapped_shape, dtype=dtype), sharding)
    ema = jax.device_put(jnp.zeros(mapped_shape, dtype=dtype), sharding)
    return student, ema


def _step_update(
    train_step: Any,
    state: tuple[Any, AdamWState, Any, jax.Array, jax.Array],
    host_batch: dict[str, np.ndarray],
    *,
    chunk_start: int,
    scheduled_step: int,
    devices: list[jax.Device],
    sharding: NamedSharding,
) -> tuple[tuple[Any, AdamWState, Any, jax.Array, jax.Array], dict[str, float]]:
    """Run one 1000-row distributed globally pooled update and EMA step."""
    params, optimizer, ema, student_state, ema_state = state
    point = onecycle_point(scheduled_step, UPDATES_PER_ARM)
    batch = _shard_chunk(host_batch, chunk_start, chunk_start + CHUNK_ROWS, sharding)
    next_state = train_step(
        params,
        optimizer,
        ema,
        student_state,
        ema_state,
        batch,
        jnp.asarray(point.learning_rate, dtype=jnp.float32),
        jnp.asarray(point.beta1, dtype=jnp.float32),
    )
    jax.block_until_ready(next_state)
    (
        next_params,
        next_optimizer,
        next_ema,
        next_student_state,
        next_ema_state,
        losses,
        norms,
        counts,
    ) = next_state
    _guard_resources(devices)
    loss_value = float(np.asarray(jax.device_get(losses))[0])
    norm_value = float(np.asarray(jax.device_get(norms))[0])
    count_values = np.asarray(jax.device_get(counts))
    if not np.all(count_values == scheduled_step + 1):
        raise AssertionError(f"AdamW count differs at attempt {scheduled_step}: {count_values}")
    if not math.isfinite(loss_value) or not math.isfinite(norm_value):
        raise FloatingPointError(f"non-finite loss or gradient norm at update {scheduled_step}")
    return (
        (next_params, next_optimizer, next_ema, next_student_state, next_ema_state),
        {
            "loss": loss_value,
            "gradient_norm_before_clip": norm_value,
            "learning_rate": point.learning_rate,
            "beta1": point.beta1,
            "beta2": point.beta2,
            "adamw_count": int(count_values[0]),
        },
    )


def _state_numpy(state: Any) -> dict[str, np.ndarray]:
    """Materialize a replicated state from one device for host export."""
    one = _first_replica(state)
    return {
        name: np.asarray(jax.device_get(value), dtype=np.float32) for name, value in one.items()
    }


def _save_arm_checkpoint(
    output: Path,
    precision: str,
    state: tuple[Any, AdamWState, Any, jax.Array, jax.Array],
    context: RuntimeContext,
    *,
    batch_group_sha: str,
    batch_data_hashes: list[str],
    history: list[dict[str, object]],
) -> dict[str, Any]:
    """Save exact FP32 student/EMA tensors, AdamW moments, and provenance."""
    arm_dir = output / precision
    arm_dir.mkdir(parents=True, exist_ok=False)
    params, optimizer, ema, _, _ = state
    student_arrays = _state_numpy(params)
    ema_arrays = _state_numpy(ema)
    optimizer_one = _first_replica(optimizer)
    first = {
        name: np.asarray(jax.device_get(value), dtype=np.float32)
        for name, value in optimizer_one.first.items()
    }
    second = {
        name: np.asarray(jax.device_get(value), dtype=np.float32)
        for name, value in optimizer_one.second.items()
    }
    if int(np.asarray(jax.device_get(optimizer_one.count))) != UPDATES_PER_ARM:
        raise AssertionError("saved AdamW state does not contain all 7460 updates")
    export_npz_state(arm_dir / "student_fp32.npz", student_arrays)
    export_npz_state(arm_dir / "ema_fp32.npz", ema_arrays)
    optimizer_path = arm_dir / "adamw_state.npz"
    optimizer_arrays = {"count": np.asarray(UPDATES_PER_ARM, dtype=np.int32)}
    optimizer_arrays.update({f"first::{name}": value for name, value in first.items()})
    optimizer_arrays.update({f"second::{name}": value for name, value in second.items()})
    np.savez(optimizer_path, **optimizer_arrays)
    history_path = arm_dir / "updates.jsonl"
    with history_path.open("x", encoding="utf-8") as stream:
        for item in history:
            stream.write(json.dumps(item, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    post_last = onecycle_point(UPDATES_PER_ARM, UPDATES_PER_ARM)
    result = {
        "precision": precision,
        "optimizer": "custom FP32 AdamW; fresh zero moments; not Muon-equivalent",
        "attempted_updates": UPDATES_PER_ARM,
        "completed_updates": UPDATES_PER_ARM,
        "optimizer_state_count": UPDATES_PER_ARM,
        "group_batch_sha256": batch_group_sha,
        "batch_data_sha256": batch_data_hashes,
        "parent_npz_sha256": context.input_file_hashes[
            str(json.loads(context.input_manifest_path.read_text())["parent_export"]["npz_path"])
        ],
        "source_hashes": context.source_hashes,
        "input_manifest_sha256": context.input_manifest_sha256,
        "student_npz_sha256": sha256_file(arm_dir / "student_fp32.npz"),
        "ema_npz_sha256": sha256_file(arm_dir / "ema_fp32.npz"),
        "adamw_state_sha256": sha256_file(optimizer_path),
        "update_log_sha256": sha256_file(history_path),
        "post_last_unused_scheduler_point": {
            "index": UPDATES_PER_ARM,
            "learning_rate": post_last.learning_rate,
            "beta1": post_last.beta1,
            "beta2": post_last.beta2,
            "used_for_optimizer_update": False,
        },
        "checkpoint_selection": "fixed final epoch-0 EMA; no validation selection",
    }
    _write_json(arm_dir / "checkpoint.json", result)
    return result


def _make_predict_step(devices: list[jax.Device]):
    """Build replicated FP32 inference for the all-VALID prediction cache."""

    def predict_replica(params: dict[str, jax.Array], state: jax.Array, inputs: jax.Array):
        prediction, next_state = forward_chunk(params, inputs, state, precision="fp32")
        return prediction, next_state

    return jax.pmap(predict_replica, devices=devices, in_axes=(0, 0, 0))


def _evaluate_all_valid(
    output: Path,
    precision: str,
    state: tuple[Any, AdamWState, Any, jax.Array, jax.Array],
    context: RuntimeContext,
    devices: list[jax.Device],
    sample_resources: Callable[[str], dict[str, Any]],
) -> dict[str, Any]:
    """Export predictions for all 1873 VALID sequences with FP32 EMA weights."""
    if precision not in {"fp32", "bf16"}:
        raise ValueError(precision)
    _, _, ema_replicated, _, _ = state
    ema = _first_replica(ema_replicated)
    sharding = context.mapped_sharding
    ema = _replicate(ema, devices, sharding)
    predict_step = _make_predict_step(devices)
    parquet = pq.ParquetFile(context.root / "datasets/valid.parquet")
    if parquet.num_row_groups != ALL_VALID_GROUPS:
        raise ValueError("VALID row-group count changed before prediction export")
    groups = tuple(range(ALL_VALID_GROUPS))
    output_path = output / precision / "predictions_valid.npy"
    if output_path.exists():
        raise FileExistsError(output_path)
    prediction_memmap = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(ALL_VALID_GROUPS, SEQUENCE_ROWS, 2),
    )
    mask_digest = hashlib.sha256()
    evaluation_set = set(context.fold.evaluation)
    scored_count = 0
    batch_groups: list[int] = []
    samples: list[Any] = []
    eval_store = context.store

    def write_batch() -> None:
        nonlocal prediction_memmap, scored_count, batch_groups, samples
        if not samples:
            return
        collated = collate_evaluation_sequences(samples)
        batch_size = len(samples)
        padded_inputs = np.zeros((GLOBAL_BATCH_SIZE, SEQUENCE_ROWS, 211), dtype=np.float32)
        padded_inputs[:batch_size] = collated["inputs"]
        mask = collated["valid"] & collated["scored"] & collated["need_prediction"]
        for offset, group in enumerate(batch_groups):
            if group in evaluation_set:
                selected = np.ascontiguousarray(mask[offset].astype(np.uint8))
                mask_digest.update(group.to_bytes(4, "little", signed=False))
                mask_digest.update(memoryview(selected))
                scored_count += int(selected.sum())
        hidden_state = jax.device_put(
            jnp.zeros((TPU_DEVICE_COUNT, 2, LOCAL_BATCH_SIZE, 256), jnp.float32),
            sharding,
        )
        for start in range(0, SEQUENCE_ROWS, CHUNK_ROWS):
            stop = start + CHUNK_ROWS
            chunk = padded_inputs[:, start:stop].reshape(
                TPU_DEVICE_COUNT, LOCAL_BATCH_SIZE, CHUNK_ROWS, 211
            )
            sharded_inputs = jax.device_put(chunk, sharding)
            predictions, hidden_state = predict_step(ema, hidden_state, sharded_inputs)
            jax.block_until_ready((predictions, hidden_state))
            local = np.asarray(jax.device_get(predictions)).reshape(
                GLOBAL_BATCH_SIZE, CHUNK_ROWS, 23
            )[:batch_size, :, :2]
            for row, group in enumerate(batch_groups):
                prediction_memmap[group, start:stop] = local[row]
            sample_resources(f"{precision}_prediction_chunk_{start}")
        batch_groups = []
        samples = []

    for row_group in groups:
        sample = load_evaluation_sequence(context.root, row_group, eval_store)
        samples.append(sample)
        batch_groups.append(row_group)
        if len(samples) == GLOBAL_BATCH_SIZE:
            write_batch()
    write_batch()
    prediction_memmap.flush()
    del prediction_memmap
    prediction_sha = sha256_file(output_path)
    split_sha = context.fold.manifest_sha256
    metadata = {
        "format": "wnn.jax-tpu-prediction-cache",
        "version": 1,
        "run": f"c20261004_e465_jax_{precision}_s3_f0",
        "split": "valid",
        "epoch": 0,
        "weight_source": "final evaluation EMA, exported and evaluated in FP32",
        "checkpoint_sha256": sha256_file(output / precision / "ema_fp32.npz"),
        "prediction_sha256": prediction_sha,
        "split_manifest_sha256": split_sha,
        "row_groups": list(groups),
        "evaluation_row_groups": sorted(evaluation_set),
        "evaluation_group_count": len(evaluation_set),
        "valid_group_count": ALL_VALID_GROUPS,
        "official_mask_policy": "need_prediction AND is_scored AND row >= 99",
        "evaluation_mask_sha256": mask_digest.hexdigest(),
        "evaluation_scored_rows": scored_count,
        "inference_precision": "fp32",
    }
    sidecar_path = output_path.with_suffix(".json")
    _write_json(sidecar_path, metadata)
    return {
        "predictions_path": str(output_path),
        "prediction_sha256": prediction_sha,
        "sidecar_path": str(sidecar_path),
        "sidecar_sha256": sha256_file(sidecar_path),
        "evaluation_mask_sha256": metadata["evaluation_mask_sha256"],
        "evaluation_scored_rows": scored_count,
    }


def _new_output(path: Path) -> Path:
    """Create an empty no-overwrite run directory."""
    path.mkdir(parents=True, exist_ok=False)
    return path


def _check_wall_budget(worker_started: float, phase: str) -> None:
    """Apply the hard wall cap from the CLI timestamp before JAX import."""
    if time.monotonic() - worker_started >= MAX_WALL_SECONDS:
        raise TimeoutError(f"E465 worker exceeded its eight-hour whole-job limit during {phase}")


def run_worker(
    *,
    phase: str,
    project_root: Path,
    input_manifest_path: Path,
    preflight_bundle_path: Path,
    output_dir: Path,
    stage_root: Path,
    worker_started_monotonic: float,
    profile_report_path: Path | None = None,
    fit_authorization_path: Path | None = None,
    profile_batches: int = 2,
) -> dict[str, Any]:
    """Run a bounded profile or one complete two-arm fixed-epoch continuation."""
    started = worker_started_monotonic
    _check_wall_budget(started, "worker entry")
    if phase not in {"profile", "fit"}:
        raise ValueError("phase must be profile or fit")
    if profile_batches != 2:
        raise ValueError("fit eligibility requires exactly two measured profile batches")
    os.environ["JAX_PLATFORMS"] = "tpu"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(ALLOCATOR_FRACTION)
    context_manifest, file_hashes = _verify_input_manifest(project_root, input_manifest_path)
    _check_wall_budget(started, "input hashing")
    input_manifest_sha256 = sha256_file(input_manifest_path)
    source_hashes = _code_hashes(stage_root)
    _check_wall_budget(started, "source hashing")
    preflight = _validate_preflight(
        preflight_bundle_path,
        input_manifest_path,
        project_root,
    )
    _check_wall_budget(started, "V6 terminal receipt validation")
    devices = _runtime_setup()
    _check_wall_budget(started, "TPU initialization")
    if phase == "fit":
        if profile_report_path is None or fit_authorization_path is None:
            raise ValueError("full fit requires a reviewed profile and root fit authorization")
        profile = _validate_profile(
            profile_report_path,
            input_manifest_sha256=input_manifest_sha256,
            source_hashes=source_hashes,
            preflight_identity=preflight,
        )
        e465_gate.validate_fit_authorization(
            fit_authorization_path,
            profile_path=profile_report_path,
            profile_report=profile,
            input_manifest_sha256=input_manifest_sha256,
            source_identity=source_hashes,
            preflight_identity=preflight,
            max_worker_seconds=MAX_WALL_SECONDS,
        )
    context = _prepare_context(
        root=project_root,
        manifest_path=input_manifest_path,
        manifest=context_manifest,
        input_hashes=file_hashes,
        stage_root=stage_root,
        devices=devices,
    )
    _check_wall_budget(started, "cache initialization")
    _validate_preflight_inputs(preflight, context)
    if context.source_hashes != source_hashes:
        raise RuntimeError("staged source changed while the run was initializing")
    group_batches = seeded_group_batches(
        context.fold.fit_union, seed=3, epoch=0, batch_size=GLOBAL_BATCH_SIZE
    )
    if len(context.fold.fit_union) != FIT_GROUPS or len(group_batches) != FULL_BATCHES:
        raise ValueError(
            "fit batch count differs from 11940 groups, 373 full batches, and drop_last"
        )
    epoch_contract = validate_epoch_batches(group_batches, context.fold.fit_union)
    if (
        epoch_contract.full_batches != FULL_BATCHES
        or epoch_contract.dropped_groups != DROPPED_GROUPS
        or epoch_contract.attempts_per_arm != UPDATES_PER_ARM
    ):
        raise AssertionError("host-checked epoch counters differ from the registered schedule")
    validate_sequence_domain(
        sequence_rows=SEQUENCE_ROWS,
        chunk_rows=CHUNK_ROWS,
        official_first_scored_row=99,
        all_valid_groups=ALL_VALID_GROUPS,
        evaluation_groups=len(context.fold.evaluation),
    )
    batch_order_sha256 = group_batch_sha256(group_batches)
    # `started` is passed by the CLI before importing this JAX module. The
    # timeout therefore includes import, hash verification, compilation,
    # training, inference, and output writes.
    all_started = started
    _check_wall_budget(all_started, "startup")
    output = _new_output(output_dir)
    telemetry_path = output / "resource_telemetry.jsonl"
    telemetry_path.touch(exist_ok=False)
    telemetry_lock = threading.Lock()

    def append_telemetry(record: dict[str, Any]) -> None:
        with telemetry_lock, telemetry_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def sample_resources(phase_name: str) -> dict[str, Any]:
        record = {"phase": phase_name, "elapsed_seconds": time.monotonic() - started}
        record.update(_guard_resources(devices))
        append_telemetry(record)
        _check_wall_budget(all_started, phase_name)
        return record

    monitor_stop = threading.Event()

    def monitor_resources() -> None:
        while not monitor_stop.wait(5.0):
            try:
                record = {
                    "phase": "periodic_resource_guard",
                    "elapsed_seconds": time.monotonic() - started,
                }
                record.update(_guard_resources(devices))
                append_telemetry(record)
                _check_wall_budget(all_started, "resource monitoring")
            except Exception as error:
                append_telemetry(
                    {
                        "phase": "resource_guard_failed",
                        "error": repr(error),
                        "elapsed_seconds": time.monotonic() - started,
                    }
                )
                os._exit(86)

    monitor = threading.Thread(target=monitor_resources, name="e465-resource-guard", daemon=True)
    monitor.start()

    def stop_monitor() -> None:
        monitor_stop.set()
        monitor.join(timeout=10.0)

    sample_resources("initialized")
    arm_reports: dict[str, Any] = {}
    profile_histories: dict[str, list[dict[str, object]]] = {}
    reference_data_hashes: list[str] | None = None
    report_profile_batches = group_batches[:profile_batches]
    for precision in ("fp32", "bf16"):
        state = _new_training_state(
            context.initial_params, precision, devices, context.mapped_sharding
        )
        train_step = _make_train_step(precision, devices)
        per_batch_seconds: list[float] = []
        arm_started = time.monotonic()
        data_hashes: list[str] = []
        history: list[dict[str, object]] = []
        batches_to_run = report_profile_batches if phase == "profile" else group_batches
        for batch_index, (yielded_index, row_groups, host_batch) in enumerate(
            _iter_prefetched_batches(batches_to_run, context)
        ):
            if yielded_index != batch_index:
                raise AssertionError("prefetched data batch order changed")
            data_digest = _batch_sha256(host_batch)
            data_hashes.append(data_digest)
            if precision == "fp32":
                if reference_data_hashes is None:
                    reference_data_hashes = []
                reference_data_hashes.append(data_digest)
            elif reference_data_hashes is None or data_digest != reference_data_hashes[batch_index]:
                raise AssertionError(f"precision arms received different host batch {batch_index}")
            batch_group_sha = group_batch_sha256((row_groups,))
            sample_resources(f"{precision}_batch_{batch_index}_loaded")
            state = (
                state[0],
                state[1],
                state[2],
                *_zero_recurrent_states(precision, context.mapped_sharding),
            )
            batch_started = time.monotonic()
            for chunk_index in range(CHUNKS_PER_BATCH):
                scheduled_step = batch_index * CHUNKS_PER_BATCH + chunk_index
                if phase == "fit" and scheduled_step >= UPDATES_PER_ARM:
                    raise AssertionError("scheduler attempted more than 7460 optimizer updates")
                chunk_start = chunk_index * CHUNK_ROWS
                state, metrics = _step_update(
                    train_step,
                    state,
                    host_batch,
                    chunk_start=chunk_start,
                    scheduled_step=scheduled_step,
                    devices=devices,
                    sharding=context.mapped_sharding,
                )
                _check_wall_budget(all_started, "training update")
                update_record = {
                    "precision": precision,
                    "epoch": 0,
                    "batch_index": batch_index,
                    "batch_group_sha256": batch_group_sha,
                    "chunk_index": chunk_index,
                    "chunk_start": chunk_start,
                    "scheduled_step": scheduled_step,
                    "attempted_updates": scheduled_step + 1,
                    "completed_updates": metrics["adamw_count"],
                    **metrics,
                }
                history.append(update_record)
                if phase == "profile" or scheduled_step % 20 == 19:
                    sample_resources(f"{precision}_update_{scheduled_step + 1}")
            per_batch_seconds.append(time.monotonic() - batch_started)
            del host_batch
        arm_seconds = time.monotonic() - arm_started
        if phase == "profile":
            max_device_fraction = max(
                float(row["observed_bytes"]) / float(row["memory_stats"]["bytes_limit"])
                for row in _guard_resources(devices)["devices"]
            )
            _, peak_rss = _host_rss_bytes()
            final_state_finite = _all_finite(state)
            if not final_state_finite:
                raise FloatingPointError(
                    f"non-finite parameters, optimizer, EMA, or recurrent state in {precision} profile"
                )
            profile_histories[precision] = history
            arm_reports[precision] = {
                "batch_count": len(per_batch_seconds),
                "attempt_count": len(history),
                "completed_update_count": int(history[-1]["completed_updates"]),
                "finite_loss_gradient_checks": len(history),
                "final_state_finite": final_state_finite,
                "first_batch_compute_seconds": per_batch_seconds[0],
                "second_batch_compute_seconds": per_batch_seconds[1],
                "profile_seconds": arm_seconds,
                "mean_fullshape_batch_wall_seconds": arm_seconds / len(per_batch_seconds),
                "peak_host_rss_bytes": peak_rss,
                "peak_device_fraction": max_device_fraction,
                "batch_data_sha256": data_hashes,
                "group_batch_sha256": [
                    group_batch_sha256((groups,)) for groups in report_profile_batches
                ],
                "compilation_included": True,
            }
            continue

        if (
            len(history) != UPDATES_PER_ARM
            or int(history[-1]["completed_updates"]) != UPDATES_PER_ARM
        ):
            raise AssertionError("full arm did not complete exactly 7460 AdamW updates")
        student_params, optimizer, ema_params, _, _ = state
        if not _all_finite((student_params, optimizer, ema_params)):
            raise FloatingPointError(f"non-finite trained state in {precision} arm")
        arm_dir = output / precision
        result = _save_arm_checkpoint(
            output,
            precision,
            state,
            context,
            batch_group_sha=batch_order_sha256,
            batch_data_hashes=data_hashes,
            history=history,
        )
        sample_resources(f"{precision}_checkpoint_saved")
        prediction_report = _evaluate_all_valid(
            output, precision, state, context, devices, sample_resources
        )
        result["prediction_export"] = prediction_report
        result["arm_wall_seconds"] = arm_seconds
        _write_json(arm_dir / "complete.json", result)
        arm_reports[precision] = result
        del state

    if phase == "profile":
        sample_resources("profile_complete_before_report")
        durations = [arm["mean_fullshape_batch_wall_seconds"] for arm in arm_reports.values()]
        projected_training = 1.20 * FULL_BATCHES * sum(durations)
        actual_precision_gate = _profile_precision_gate(arm_reports, profile_histories)
        report = {
            "format": "wnn.e465-tpu-fullshape-profile",
            "status": "profile_complete",
            "input_manifest_sha256": context.input_manifest_sha256,
            "input_file_hashes": context.input_file_hashes,
            "source_hashes": context.source_hashes,
            "device_inventory": context.device_inventory,
            "preflight_delivery_receipt_sha256": preflight["v6_delivery_receipt_sha256"],
            "preflight_terminal_receipt_sha256": preflight["v6_terminal_receipt_sha256"],
            "preflight_realdata_update_sha256": preflight["v6_realdata_update_sha256"],
            "preflight_identity": preflight,
            "parent_export": context_manifest["parent_export"],
            "fold_manifest_sha256": context.fold.manifest_sha256,
            "fit_group_counts": {
                "train": len(context.fold.train),
                "valid_fit": len(context.fold.valid_fit),
                "total": FIT_GROUPS,
                "batches": FULL_BATCHES,
                "dropped_last": DROPPED_GROUPS,
            },
            "global_batch_size": GLOBAL_BATCH_SIZE,
            "local_batch_size": LOCAL_BATCH_SIZE,
            "sequence_rows": SEQUENCE_ROWS,
            "chunk_rows": CHUNK_ROWS,
            "profile_batches": profile_batches,
            "attempts_per_arm": profile_batches * CHUNKS_PER_BATCH,
            "arms": arm_reports,
            "actual_precision_gate": actual_precision_gate,
            "projected_training_seconds": projected_training,
            "projected_training_is_a_lower_bound": True,
            "whole_job_limit_seconds": MAX_WALL_SECONDS,
            "actual_full_splitclock_fit_performed": False,
            "profile_is_not_predictive_evidence": True,
            "limitations": [
                "full 7460-update fit and all-VALID inference remain unmeasured",
                "training-time projection uses two full-shape batches per precision and a 20% margin; it excludes all-VALID inference and output I/O",
                "TPU memory values report allocator telemetry, not total physical HBM reservation",
            ],
        }
        if projected_training > MAX_WALL_SECONDS:
            report["status"] = "profile_failed_runtime_projection"
        _write_json(output / "profile_report.json", report)
        stop_monitor()
        return report

    if reference_data_hashes is None or len(reference_data_hashes) != FULL_BATCHES:
        raise AssertionError("full fit host batch hash history is incomplete")
    sample_resources("fit_complete_before_report")
    result = {
        "format": "wnn.e465-tpu-fit",
        "status": "fits_and_exports_complete",
        "input_manifest_sha256": context.input_manifest_sha256,
        "input_file_hashes": context.input_file_hashes,
        "source_hashes": context.source_hashes,
        "preflight_identity": preflight,
        "fit_authorization_sha256": sha256_file(fit_authorization_path) if fit_authorization_path else None,
        "profile_report_sha256": sha256_file(profile_report_path) if profile_report_path else None,
        "parent_export": context_manifest["parent_export"],
        "fold_manifest_sha256": context.fold.manifest_sha256,
        "fit_group_batch_sha256": batch_order_sha256,
        "fit_batches": FULL_BATCHES,
        "dropped_fit_groups": DROPPED_GROUPS,
        "updates_per_arm": UPDATES_PER_ARM,
        "attempts_per_arm": UPDATES_PER_ARM,
        "completed_optimizer_updates_per_arm": UPDATES_PER_ARM,
        "training_objective": {
            "primary": "original weighted combo wmse against fixed alpha=.5 stopped horizon-8 SBD target",
            "teachers": list(TEACHERS),
            "consistency": "coefficient 1.0 against incoming EMA, excluded from metric weights",
            "aux_struct": 0.3,
            "aux_next": 0.3,
            "propensity": "prop_v1 squared plus 0.03; warmup 99; valid true-mask 0.03; since scale 32 floor .25",
            "normalization": "frozen mean/std in exact parent FP32 state dict",
        },
        "optimizer": "fresh custom JAX FP32 AdamW, not Muon-equivalent",
        "scheduler": "exact PyTorch OneCycleLR scalar port, indices 0..7459 used",
        "arms": arm_reports,
        "batch_data_sha256_by_batch": reference_data_hashes,
        "inference": "epoch-0 final EMA parameters; full VALID order (1873); FP32; no epoch selection",
        "evaluation_row_groups": list(context.fold.evaluation),
        "evaluation_mask": "need_prediction AND is_scored AND row >= 99",
        "elapsed_seconds": time.monotonic() - all_started,
        "whole_job_limit_seconds": MAX_WALL_SECONDS,
        "scientific_limitations": [
            "AdamW is an optimizer change; this is not an exact continuation of the parent's Muon trajectory",
            "FP32 versus BF16 comparison isolates recurrent compute precision within this JAX AdamW port",
            "full-split performance and all-VALID prediction artifacts require independent scoring",
        ],
    }
    _write_json(output / "fit_report.json", result)
    stop_monitor()
    return result
