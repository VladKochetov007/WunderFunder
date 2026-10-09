"""Fail-closed, serial two-seed E520 scratch-fit and streamed-export worker.

This worker only fits the frozen raw-only recipe after independent review and
root admission receipts are present. It never scores, selects, blends, or
submits predictions.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import signal
import sys
import tarfile
import threading
import time
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any

from fit_guards import (
    EXPERIMENT_ID,
    MAX_WORKER_WALL_SECONDS,
    FitGateError,
    artifact_hashes,
    canonical_sha256,
    sha256_bytes,
    sha256_file,
    trainer_argv,
    validate_controller_admission,
    validate_prediction_chunk,
    validate_review,
    validate_training_target_semantics,
)
from support import export_guards, input_identity
from support.gpu_inventory import require_physical_gpu_inventory
from support.gpu_observer import enforce_active_gpu_observation

CONFIG_NAME = "FIT_STAGE_CONFIG.json"
REVIEW_NAME = "external_review.json"
ADMISSION_NAME = "controller_admission.json"
PACKET_META_NAME = "packet_meta.json"
MAX_HASH_BLOCK_BYTES = 1 << 20
HASH_PROGRESS_BYTES = 1 << 28
OUTPUT_SUBDIR = Path("e520_fit")
SEQUENCE_CHUNK_ROWS = 4_000
TRAIN_BATCH_ROWS = 16
PREDICTION_BYTES_PER_VALUE = 4


class FitAbort(RuntimeError):
    """Raised when live input, export, runtime, or resource evidence fails."""


class FitResourceGuard:
    """Poll worker caps continuously and recheck allocator/RSS limits per batch."""

    def __init__(
        self,
        torch_module: Any | None,
        working: Path,
        receipts: Path,
        wall_cap: int,
        *,
        started_at: float,
        runtime_identity: dict[str, Any] | None = None,
    ) -> None:
        self.torch = torch_module
        self.working = working
        self.receipts = receipts
        self.wall_cap = wall_cap
        self.started = started_at
        self.runtime_identity = runtime_identity
        self.stop_event = threading.Event()
        self.failure: str | None = None
        self.latest: dict[str, Any] = {}
        self.latest_full_time = 0.0
        self.thread = threading.Thread(target=self._poll, name="e520-resource-guard", daemon=True)

    def _enforce(self, snapshot: dict[str, Any]) -> None:
        caps = input_identity.CAPS
        elapsed = time.monotonic() - self.started
        if elapsed > self.wall_cap:
            raise FitAbort("fit worker exceeded the separately admitted wall cap")
        if snapshot["rss_peak_bytes"] > caps["rss_bytes"]:
            raise FitAbort("process RSS high-water exceeded 8 GB")
        for key, cap in (
            ("cuda_allocated_bytes", caps["cuda_allocated_bytes"]),
            ("cuda_reserved_bytes", caps["cuda_reserved_bytes"]),
            ("cuda_allocated_peak_bytes", caps["cuda_allocated_bytes"]),
            ("cuda_reserved_peak_bytes", caps["cuda_reserved_bytes"]),
            ("gpu_process_bytes", caps["gpu_process_bytes"]),
        ):
            value = snapshot.get(key)
            if value is not None and value > cap:
                raise FitAbort(f"{key} exceeded its hard cap")
        for prefix in ("host", "cgroup"):
            available = snapshot[f"{prefix}_available_bytes"]
            fraction = snapshot[f"{prefix}_available_fraction"]
            if available < caps["headroom_bytes"] or fraction < caps["headroom_fraction"]:
                raise FitAbort(f"{prefix} headroom fell below a hard cap")
        if snapshot["disk_free_bytes"] < caps["disk_free_bytes"]:
            raise FitAbort("working disk fell below the 8 GB floor")
        if self.torch is None:
            if (
                snapshot.get("gpu_process_query_status") != "pre_cuda_idle"
                or snapshot.get("gpu_process_bytes") is not None
                or snapshot.get("cuda_initialized") is not False
                or snapshot.get("cuda_allocated_bytes") is not None
            ):
                raise FitAbort("pre-CUDA resource phase is not explicitly idle")
        else:
            if snapshot.get("cuda_initialized") is not True:
                raise FitAbort("active CUDA initialization state is unavailable")
            if (
                snapshot.get("cuda_allocated_peak_bytes") is None
                or snapshot.get("cuda_reserved_peak_bytes") is None
            ):
                raise FitAbort("CUDA allocator peak APIs are unavailable")
            try:
                enforce_active_gpu_observation(
                    snapshot,
                    process_cap_bytes=caps["gpu_process_bytes"],
                    min_free_bytes=caps["headroom_bytes"] + 2 * 1024**3,
                )
                if self.runtime_identity is not None:
                    require_physical_gpu_inventory(self.runtime_identity)
            except ValueError as error:
                raise FitAbort(str(error)) from error

    def _persist_abort(self, error: Exception, snapshot: dict[str, Any]) -> None:
        """Write the exact synchronous/background failure snapshot before propagation."""
        runtime_path = self.receipts / "environment.json"
        runtime_sha = sha256_file(runtime_path) if runtime_path.is_file() else None
        _atomic_json(
            self.receipts / "resource_abort.json",
            {
                "status": "aborted",
                "phase": "live_resource_guard",
                "reason": f"{type(error).__name__}: {error}",
                "elapsed_seconds": round(time.monotonic() - self.started, 3),
                "last_snapshot": snapshot,
                "runtime_identity_path": str(runtime_path) if runtime_sha is not None else None,
                "runtime_identity_sha256": runtime_sha,
            },
        )

    def check_full(self) -> dict[str, Any]:
        try:
            snapshot = input_identity.resource_snapshot(self.torch, self.working)
        except Exception as error:
            self._persist_abort(error, self.latest)
            raise
        self.latest = snapshot
        self.latest_full_time = time.monotonic()
        _atomic_json(
            self.receipts / "resource.json",
            {
                "phase": "fit_or_export_active",
                "elapsed_seconds": round(time.monotonic() - self.started, 3),
                "snapshot": snapshot,
            },
        )
        try:
            self._enforce(snapshot)
        except Exception as error:
            self._persist_abort(error, snapshot)
            raise
        return snapshot

    def check_batch_boundary(self) -> dict[str, Any]:
        """Check allocator peaks after each trainer batch without spawning nvidia-smi each time."""
        snapshot = input_identity.resource_snapshot(None, self.working)
        snapshot.update(
            {
                "cuda_initialized": True,
                "cuda_allocated_bytes": int(self.torch.cuda.memory_allocated(0)),
                "cuda_reserved_bytes": int(self.torch.cuda.memory_reserved(0)),
                "cuda_allocated_peak_bytes": int(self.torch.cuda.max_memory_allocated(0)),
                "cuda_reserved_peak_bytes": int(self.torch.cuda.max_memory_reserved(0)),
                "gpu_free_bytes": int(self.torch.cuda.mem_get_info(0)[0]),
            }
        )
        if time.monotonic() - self.latest_full_time > 3.0:
            return self.check_full()
        for key in (
            "gpu_process_bytes",
            "gpu_process_query_status",
            "gpu_process_query_raw",
            "gpu_process_query_raw_byte_length",
            "gpu_process_query_raw_sha256",
            "gpu_process_query_raw_truncated",
            "gpu_process_query_stderr",
            "gpu_process_query_stderr_byte_length",
            "gpu_process_query_stderr_sha256",
            "gpu_process_query_stderr_truncated",
            "gpu_process_query_return_code",
            "gpu_process_query_diagnostic",
            "gpu_total_bytes",
            "gpu_memory_info_total_bytes",
            "gpu_memory_totals_match",
            "gpu_device_used_upper_bound_bytes",
            "gpu_allocation_measurement_basis",
            "gpu_resource_query_status",
            "gpu_resource_query_diagnostic",
        ):
            snapshot[key] = self.latest.get(key)
        properties = self.torch.cuda.get_device_properties(0)
        free_bytes, memory_info_total = self.torch.cuda.mem_get_info(0)
        total_bytes = int(properties.total_memory)
        free_bytes = int(free_bytes)
        memory_info_total = int(memory_info_total)
        totals_match = total_bytes == memory_info_total
        snapshot.update(
            {
                "gpu_total_bytes": total_bytes,
                "gpu_memory_info_total_bytes": memory_info_total,
                "gpu_memory_totals_match": totals_match,
                "gpu_device_used_upper_bound_bytes": (
                    total_bytes - free_bytes if totals_match else None
                ),
                "gpu_allocation_measurement_basis": "process_table_pid"
                if snapshot["gpu_process_bytes"] is not None
                else "device_total_minus_free_upper_bound"
                if totals_match
                else "unavailable",
                "gpu_resource_query_status": "ok" if totals_match else "total_mismatch",
                "gpu_free_bytes": free_bytes,
            }
        )
        try:
            self._enforce(snapshot)
        except Exception as error:
            self.latest = snapshot
            self._persist_abort(error, snapshot)
            raise
        self.latest = snapshot
        return snapshot

    def _poll(self) -> None:
        while not self.stop_event.wait(1.0):
            try:
                self.check_full()
            except Exception as error:  # persist before forcing the active trainer to stop
                self.failure = f"{type(error).__name__}: {error}"
                self._persist_abort(error, self.latest)
                os.kill(os.getpid(), signal.SIGTERM)
                return

    def start(self) -> None:
        self.check_full()
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread.ident is not None:
            self.thread.join(timeout=3)
            if self.thread.is_alive():
                raise FitAbort("resource monitor did not stop at the CUDA handoff")

    def raise_if_failed(self) -> None:
        if self.failure:
            raise FitAbort(self.failure)


def stop_pre_cuda_guard(
    pre_cuda_guard: FitResourceGuard, receipts: Path, started_at: float
) -> None:
    """Stop explicit-idle polling and record this before CUDA initialization."""
    handoff_path = receipts / "resource_monitor_handoff.json"
    pre_cuda_guard.stop()
    _atomic_json(
        handoff_path,
        {
            "phase": "pre_cuda_monitor_stopped",
            "status": "cuda_initialization_not_started",
            "elapsed_seconds": round(time.monotonic() - started_at, 3),
            "last_pre_cuda_snapshot": pre_cuda_guard.latest,
        },
    )


def start_active_guard(
    torch_module: Any,
    working: Path,
    receipts: Path,
    wall_cap: int,
    started_at: float,
    runtime: dict[str, Any],
) -> FitResourceGuard:
    """Persist runtime identity, then start active monitoring after CUDA init."""
    handoff_path = receipts / "resource_monitor_handoff.json"
    try:
        handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FitAbort("pre-CUDA monitor handoff receipt is unavailable") from error
    if (
        handoff.get("phase") != "pre_cuda_monitor_stopped"
        or handoff.get("status") != "cuda_initialization_not_started"
    ):
        raise FitAbort("CUDA cannot initialize before the pre-CUDA monitor is stopped")
    environment_path = receipts / "environment.json"
    if not environment_path.is_file():
        raise FitAbort("runtime identity receipt must be durable before active guard startup")
    try:
        runtime_on_disk = json.loads(environment_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FitAbort(
            "runtime identity receipt cannot be decoded before active startup"
        ) from error
    if runtime_on_disk != runtime:
        raise FitAbort("runtime identity receipt changed before active guard startup")
    active_guard = FitResourceGuard(
        torch_module,
        working,
        receipts,
        wall_cap,
        started_at=started_at,
        runtime_identity=runtime,
    )
    _atomic_json(
        handoff_path,
        {
            "phase": "active_monitor_starting",
            "status": "runtime_identity_persisted",
            "elapsed_seconds": round(time.monotonic() - started_at, 3),
            "runtime_identity_path": str(environment_path),
            "runtime_identity_sha256": sha256_file(environment_path),
            "pre_cuda_snapshot": handoff["last_pre_cuda_snapshot"],
        },
    )
    active_guard.start()
    _atomic_json(
        handoff_path,
        {
            "phase": "active_monitor_running",
            "status": "active_resource_gate_passed",
            "elapsed_seconds": round(time.monotonic() - started_at, 3),
            "runtime_identity_path": str(environment_path),
            "runtime_identity_sha256": sha256_file(environment_path),
            "initial_active_snapshot": active_guard.latest,
        },
    )
    return active_guard


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    """Persist small JSON receipts through a same-directory atomic replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    if temporary.exists():
        temporary.unlink()
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _safe_extract_archive(archive_path: Path, target: Path) -> None:
    """Extract only regular files/directories beneath a fresh source root."""
    if target.exists():
        raise FitAbort("refusing to reuse an existing trainer source directory")
    target.mkdir(parents=True)
    with tarfile.open(archive_path, "r") as archive:
        for member in archive.getmembers():
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts or not (member.isfile() or member.isdir()):
                raise FitAbort(f"unsafe pinned source archive member: {member.name}")
        archive.extractall(target)
    trainer = target / "scripts" / "train_rnn.py"
    if sha256_file(trainer) != "8a29459e998f7f5d6a03b7a4b87c6f4b569002e65b8c87127c37138360909cbf":
        raise FitAbort("extracted trainer bytes differ from the frozen E508-v5 parser/trainer")


def _required_receipts(packet_root: Path) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    """Validate the exact reviewed source set and two external approval receipts."""
    config_path = packet_root / CONFIG_NAME
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise FitGateError("fit configuration names another experiment")
    validate_training_target_semantics(config, packet_root)
    meta_path = packet_root / PACKET_META_NAME
    if not meta_path.is_file():
        raise FitGateError("review-gated packet metadata is missing")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    hashes = artifact_hashes(packet_root)
    if (
        meta.get("experiment_id") != EXPERIMENT_ID
        or meta.get("packet_id") != config.get("packet_id")
        or meta.get("artifact_hashes") != hashes
    ):
        raise FitGateError("packet metadata does not bind the exact fit source hash map")
    review_bytes = (packet_root / REVIEW_NAME).read_bytes()
    review_sha = sha256_file(packet_root / REVIEW_NAME)
    validate_review(review_bytes, meta.get("external_review_sha256", ""), hashes)
    admission_bytes = (packet_root / ADMISSION_NAME).read_bytes()
    admission_sha = sha256_file(packet_root / ADMISSION_NAME)
    admission = validate_controller_admission(
        admission_bytes,
        meta.get("controller_admission_sha256", ""),
        review_sha256=review_sha,
        hashes=hashes,
        config=config,
    )
    if meta.get("controller_admission_sha256") != admission_sha:
        raise FitGateError("packet metadata does not pin the controller admission receipt")
    if (
        meta.get("grader_script_sha256") != config["grader_artifacts"]["script_sha256"]
        or meta.get("profile_receipt_sha256") != admission["profile_gate"]["receipt_sha256"]
        or meta.get("worker_wall_cap_seconds")
        != admission["live_admission"]["worker_wall_cap_seconds"]
        or meta.get("anchor_report_sha256")
        != config["evaluation"]["anchor_evaluation_report_sha256"]
        or meta.get("score_enabled") is not False
    ):
        raise FitGateError(
            "packet metadata does not bind the admitted profile/grader/no-score contract"
        )
    return config, meta, review_sha, admission_sha


def _input_order_receipts(
    source_root: Path, identity: dict[str, Any], config: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Load only small target indices and bind full row-group/sequence ordering."""
    import numpy as np

    orders: dict[str, dict[str, Any]] = {}
    for split_name in ("train", "valid"):
        cache_path = source_root / "cache" / f"{split_name}_targets.npz"
        with np.load(cache_path, allow_pickle=False) as cache:
            groups = np.asarray(cache["row_group"], dtype="<i8")
            sequence_ids = np.asarray(cache["seq_ix"], dtype="<i8")
        if groups.ndim != 1 or sequence_ids.shape != groups.shape or not len(groups):
            raise FitAbort(f"malformed {split_name} target index")
        group_hash = hashlib.sha256(groups.tobytes()).hexdigest()
        index_digest = hashlib.sha256(groups.tobytes() + sequence_ids.tobytes()).hexdigest()
        if index_digest != identity["target_index_hashes"].get(split_name):
            raise FitAbort(f"{split_name} target cache index changed after input preflight")
        expected_count = config["prediction_export"]["splits"]
        if split_name not in expected_count:
            raise FitAbort("prediction export split contract changed")
        if (
            split_name == "valid"
            and group_hash != config["evaluation"]["shared_target_row_groups_sha256"]
        ):
            raise FitAbort("valid export row-group digest differs from the fixed E511 anchor order")
        orders[split_name] = {
            "row_groups": groups,
            "seq_ix": sequence_ids,
            "row_group_count": int(len(groups)),
            "row_groups_sha256": group_hash,
            "target_index_sha256": index_digest,
            "target_cache_sha256": identity["observed"][f"cache/{split_name}_targets.npz"][
                "sha256"
            ],
        }
    return orders


def _write_digest_handoff_runtime(
    staged: dict[str, Any], receipts: Path, *, phase: str
) -> dict[str, Any]:
    """Persist cumulative digest handoff counters without re-reading raw data."""
    path = receipts / "verified_digest_handoff_runtime.json"
    report = input_identity.digest_handoff_runtime_report(
        staged["identity"],
        staged["digest_handoff"],
        staged["raw_paths"],
        staged["digest_receipts"],
        phase=phase,
    )
    _atomic_json(path, report)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "phase": phase,
        "sources": report["sources"],
    }


def _expected_fit_group_order(split: Any, seed: int) -> dict[str, Any]:
    """Match the trainer's deterministic full-fold membership order for one seed."""
    import numpy as np

    train = np.random.default_rng(seed).permutation(np.asarray(split.train_groups, dtype=np.int64))
    valid = np.asarray(split.valid_groups, dtype=np.int64)
    return {
        "train": {
            "count": int(len(train)),
            "ordered_int64_le_sha256": hashlib.sha256(
                train.astype("<i8", copy=False).tobytes()
            ).hexdigest(),
        },
        "valid": {
            "count": int(len(valid)),
            "ordered_int64_le_sha256": hashlib.sha256(
                valid.astype("<i8", copy=False).tobytes()
            ).hexdigest(),
        },
        "train_groups": train,
        "valid_groups": valid,
    }


def _validate_fit_provenance(
    provenance: dict[str, Any], checkpoint: dict[str, Any], seed: int, split: Any
) -> dict[str, Any]:
    """Assert fixed epoch/recipe and exact fold membership before exporting."""
    expected = _expected_fit_group_order(split, seed)
    fit_groups = provenance.get("fit_row_groups", {})
    import numpy as np

    observed_membership = {
        name: {
            "count": len(fit_groups.get(name, [])),
            "ordered_int64_le_sha256": hashlib.sha256(
                np.asarray(fit_groups.get(name, []), dtype="<i8").tobytes()
            ).hexdigest(),
        }
        for name in ("train", "valid")
    }
    if observed_membership != {key: expected[key] for key in ("train", "valid")}:
        raise FitAbort("trainer fit row groups differ from the seed-specific fold-0 contract")
    recorded_identity = provenance.get("fit_membership_identity", {})
    if any(recorded_identity.get(name) != observed_membership[name] for name in ("train", "valid")):
        raise FitAbort("trainer fit membership digest differs from its explicit row-group list")
    ck_args = checkpoint.get("args", {})
    fixed = {
        "seed": seed,
        "arch": "gru",
        "cell": "gru",
        "hidden": 128,
        "layers": 2,
        "dropout": 0.0,
        "raw_only": True,
        "loss": "wmse",
        "loss_norm": "chunk",
        "train_source": "train+valid",
        "train_seqs": 10_607,
        "valid_fold": 0,
        "epochs": 4,
        "checkpoint_policy": "fixed",
        "checkpoint_epoch": 3,
        "no_eval_during_fit": True,
        "batch": 16,
        "chunk": 2_000,
        "lr": 0.001,
        "wd": 0.01,
        "optim": "adamw",
        "sched": "onecycle",
        "precision": "fp16",
        "init": "",
        "distill": "",
        "propensity": "",
        "prop_input": "",
        "prop_power": 1.0,
        "prop_eps": 0.0,
        "valid_true_mask": -1,
        "distill_boot_h": 0,
        "ema_teacher": 0.0,
        "aux_struct": 0.0,
        "aux_next": 0.0,
        "aux_tier": 0.0,
        "aux_mask": 0.0,
        "aux_sign": 0.0,
        "aux_quote": 0.0,
        "aux_weight": 0.0,
        "swa_frac": 0.0,
        "fast": True,
    }
    mismatched = {
        key: (ck_args.get(key), value) for key, value in fixed.items() if ck_args.get(key) != value
    }
    if checkpoint.get("epoch") != 3 or provenance.get("selected_epoch") != 3 or mismatched:
        raise FitAbort(f"checkpoint is not the fixed epoch-3 raw baseline: {mismatched}")
    if provenance.get("split_manifest_sha256") != input_identity.SPLIT_SHA256:
        raise FitAbort("trainer provenance names a different split manifest")
    return {
        "fit_membership": observed_membership,
        "fit_membership_identity": provenance.get("fit_membership_identity"),
        "evaluation_row_groups_count": len(provenance.get("evaluation_row_groups", [])),
        "selected_epoch": provenance.get("selected_epoch"),
        "trainer_args": {key: ck_args.get(key) for key in fixed},
    }


def _trainer_module(source_root: Path, suffix: str) -> Any:
    """Import the pinned trainer under a unique module name without editing its archive."""
    path = source_root / "scripts" / "train_rnn.py"
    spec = importlib.util.spec_from_file_location(f"e520_train_rnn_{suffix}", path)
    if spec is None or spec.loader is None:
        raise FitAbort("cannot load the pinned trainer source")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(source_root / "src"))
    spec.loader.exec_module(module)
    return module


def _atomic_torch_save(torch_module: Any) -> Callable[[], None]:
    """Patch only path-based checkpoint saves to staged fsync/rename writes."""
    original_save = torch_module.save

    def save(obj: Any, destination: Any, *args: Any, **kwargs: Any) -> Any:
        if not isinstance(destination, (str, os.PathLike)):
            return original_save(obj, destination, *args, **kwargs)
        final = Path(destination)
        partial = final.with_name(final.name + ".partial")
        if partial.exists():
            raise FitAbort(f"stale staged checkpoint exists: {partial.name}")
        original_save(obj, partial, *args, **kwargs)
        with partial.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(partial, final)
        _fsync_directory(final.parent)
        return None

    torch_module.save = save

    def restore() -> None:
        torch_module.save = original_save

    return restore


def _load_checkpoint(module: Any, checkpoint_path: Path, device: Any) -> tuple[Any, dict[str, Any]]:
    """Recreate the raw-only GRU from the fixed checkpoint for streamed export."""
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = checkpoint["args"]
    module.RAW_ONLY = bool(args["raw_only"])
    module.RESEARCH_FEATURES = args["research_features"]
    module.PRIVILEGED_MODE = args["privileged_mode"]
    module.PRIVILEGED_HORIZON = int(args["privileged_horizon"])
    module.INPUT_PROPENSITY = None
    module.ONLINE_NORM = bool(args["online_norm"])
    module.HEURISTICS = bool(args["heuristics"])
    module.READOUT = checkpoint.get("readout")
    module.READOUT_TILTED = bool(checkpoint.get("readout_tilted", False))
    model = module.make_model(args, int(checkpoint["n_in"]), 2)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    model.eval()
    return model, checkpoint


def _aligned_flush_and_advise(
    mapped: Any,
    descriptor: int,
    *,
    first_row: int,
    last_row: int,
    sequence_rows: int,
) -> dict[str, Any]:
    """Flush a small written memmap span and release completed file-cache pages."""
    fadvise = getattr(os, "posix_fadvise", None)
    dontneed = getattr(os, "POSIX_FADV_DONTNEED", None)
    if not callable(fadvise) or dontneed is None:
        raise FitAbort("POSIX_FADV_DONTNEED is required for bounded prediction-file page cache")
    page_size = int(os.sysconf("SC_PAGE_SIZE"))
    first_byte = int(mapped.offset) + first_row * sequence_rows * 2 * PREDICTION_BYTES_PER_VALUE
    last_byte = int(mapped.offset) + last_row * sequence_rows * 2 * PREDICTION_BYTES_PER_VALUE
    aligned_start = first_byte - first_byte % page_size
    aligned_end = min(
        int(mapped.offset) + int(mapped.nbytes),
        ((last_byte + page_size - 1) // page_size) * page_size,
    )
    span = max(page_size, aligned_end - aligned_start)
    mapped._mmap.flush(aligned_start, span)
    fadvise(descriptor, aligned_start, span, dontneed)
    return {
        "status": "hinted",
        "advice": "POSIX_FADV_DONTNEED",
        "page_size_bytes": page_size,
        "advised_start_bytes": aligned_start,
        "advised_length_bytes": span,
        "semantics": "successful kernel hint; page eviction is not asserted",
    }


def _hash_file_releasing_cache(path: Path, receipt_path: Path, label: str) -> str:
    """Stream SHA-256 and best-effort release every completed page-aligned range."""
    fadvise = getattr(os, "posix_fadvise", None)
    dontneed = getattr(os, "POSIX_FADV_DONTNEED", None)
    page_size = int(os.sysconf("SC_PAGE_SIZE"))
    if not callable(fadvise) or dontneed is None or MAX_HASH_BLOCK_BYTES % page_size:
        record = {
            "status": "unsupported",
            "path": str(path),
            "file_bytes": path.stat().st_size,
            "posix_fadvise_available": callable(fadvise),
            "dontneed_constant_available": dontneed is not None,
        }
        _record_hash_io_advice(receipt_path, label, record)
        raise FitAbort("output cache-safe hash support is unavailable")
    digest = hashlib.sha256()
    hashed = advised = 0
    calls = 0
    failures = 0
    record: dict[str, Any] = {
        "status": "hashing",
        "path": str(path),
        "file_bytes": path.stat().st_size,
        "block_bytes": MAX_HASH_BLOCK_BYTES,
        "page_size_bytes": page_size,
        "advice": "POSIX_FADV_DONTNEED",
        "semantics": "kernel hint only; success does not prove eviction",
        "bytes_hashed": 0,
        "bytes_advised": 0,
        "advice_calls": 0,
        "advice_failures": 0,
    }
    with path.open("rb", buffering=0) as stream:
        fd = stream.fileno()
        while block := stream.read(MAX_HASH_BLOCK_BYTES):
            digest.update(block)
            hashed += len(block)
            advised_end = hashed - hashed % page_size
            if advised_end > advised:
                try:
                    fadvise(fd, advised, advised_end - advised, dontneed)
                    calls += 1
                    advised = advised_end
                except OSError:
                    failures += 1
            if (
                hashed >= int(record["file_bytes"])
                or hashed // HASH_PROGRESS_BYTES > record["bytes_hashed"] // HASH_PROGRESS_BYTES
            ):
                record.update(
                    {
                        "bytes_hashed": hashed,
                        "bytes_advised": advised,
                        "advice_calls": calls,
                        "advice_failures": failures,
                    }
                )
                _record_hash_io_advice(receipt_path, label, record)
    if hashed != path.stat().st_size or failures:
        record.update({"status": "failed", "bytes_hashed": hashed, "advice_failures": failures})
        _record_hash_io_advice(receipt_path, label, record)
        raise FitAbort("output hash was incomplete or page-cache advice failed")
    record.update(
        {
            "status": "completed",
            "bytes_hashed": hashed,
            "bytes_advised": advised,
            "advice_calls": calls,
            "advice_failures": failures,
            "sha256": digest.hexdigest(),
        }
    )
    _record_hash_io_advice(receipt_path, label, record)
    return digest.hexdigest()


def _record_hash_io_advice(receipt_path: Path, label: str, record: dict[str, Any]) -> None:
    """Atomically merge one export hash record without erasing prior split entries."""
    files: dict[str, Any] = {}
    if receipt_path.exists():
        prior = json.loads(receipt_path.read_text(encoding="utf-8"))
        if prior.get("phase") != "output_hash_pagecache" or not isinstance(
            prior.get("files"), dict
        ):
            raise FitAbort("output hash advice receipt has an unexpected schema")
        files.update(prior["files"])
    files[label] = record.copy()
    _atomic_json(
        receipt_path,
        {"phase": "output_hash_pagecache", "files": files},
    )


def _check_disk_for_remaining_outputs(
    working: Path, remaining_payload_bytes: int, min_free_bytes: int
) -> int:
    free = shutil.disk_usage(working).free
    if remaining_payload_bytes < 0 or free < remaining_payload_bytes + min_free_bytes:
        raise FitAbort("disk cannot hold remaining staged predictions and preserve the 8 GB floor")
    return free


def _write_split_predictions(
    *,
    model: Any,
    module: Any,
    torch_module: Any,
    source_root: Path,
    order: dict[str, Any],
    split_name: str,
    run_name: str,
    seed: int,
    checkpoint_sha256: str,
    trainer_provenance_sha256: str,
    split_sha256: str,
    output_root: Path,
    working: Path,
    receipts: Path,
    guard: FitResourceGuard,
    remaining_payload_bytes: int,
    input_identity_sha256: str,
    source_archive_sha256: str,
    trainer_sha256: str,
    fit_argv_sha256: str,
) -> tuple[dict[str, Any], int]:
    """Predict one fixed target-cache order through 4k-row CUDA chunks to a memmap."""
    import numpy as np

    rows = int(order["row_group_count"])
    sequence_rows = int(module.SEQUENCE_LENGTH)
    feature_count = int(module.N_FEATURES)
    if sequence_rows != 20_000 or feature_count <= 0:
        raise FitAbort("pinned trainer export dimensions differ from the E520 tensor contract")
    expected_shape = (rows, sequence_rows, 2)
    payload_bytes = rows * sequence_rows * 2 * PREDICTION_BYTES_PER_VALUE
    free_at_start = _check_disk_for_remaining_outputs(
        working, remaining_payload_bytes, input_identity.CAPS["disk_free_bytes"]
    )
    contract_binding = {
        "split_manifest_sha256": split_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "target_order_sha256": order["row_groups_sha256"],
    }
    export_guards.validate_streaming_export_preconditions(
        rows=rows,
        free_disk_bytes=free_at_start,
        reserve_bytes=(remaining_payload_bytes - payload_bytes)
        + input_identity.CAPS["disk_free_bytes"],
        split_sha256=split_sha256,
        expected_split_sha256=input_identity.SPLIT_SHA256,
        row_groups_sha256=order["row_groups_sha256"],
        expected_row_groups_sha256=order["row_groups_sha256"],
        provenance=contract_binding,
        checkpoint_sha256=checkpoint_sha256,
        target_order_sha256=order["row_groups_sha256"],
    )
    final = output_root / f"{run_name}_{split_name}.npy"
    partial = final.with_name(final.name + ".partial")
    sidecar = final.with_suffix(".json")
    if final.exists() or partial.exists() or sidecar.exists():
        raise FitAbort(f"refusing to overwrite prediction output {final.name}")
    mapped: Any | None = None
    reader: Any | None = None
    mapped = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float32, shape=expected_shape)
    descriptor = os.open(partial, os.O_RDWR)
    groups = order["row_groups"]
    sequence_ids = order["seq_ix"]
    written_rows = 0
    written_chunks = 0
    pagecache_advice_count = 0
    progress_path = receipts / "phase.json"
    try:
        reader = module.SequenceFile(source_root / "datasets" / f"{split_name}.parquet")
        for batch_start in range(0, rows, TRAIN_BATCH_ROWS):
            batch_stop = min(batch_start + TRAIN_BATCH_ROWS, rows)
            batch_rows = batch_stop - batch_start
            raw_batch = np.empty((batch_rows, sequence_rows, feature_count), dtype=np.float32)
            for offset, (row_group, expected_seq_ix) in enumerate(
                zip(
                    groups[batch_start:batch_stop],
                    sequence_ids[batch_start:batch_stop],
                    strict=True,
                )
            ):
                sequence = reader.read(int(row_group), targets=False, mask=False)
                if sequence.seq_ix != int(expected_seq_ix):
                    raise FitAbort(
                        f"{split_name} row-group sequence identity changed during export"
                    )
                if sequence.features.shape != (sequence_rows, feature_count):
                    raise FitAbort(f"{split_name} raw feature shape changed during export")
                raw_batch[offset] = sequence.features
                del sequence
            state = None
            for time_start, time_stop in export_guards.chunk_ranges(
                sequence_rows, SEQUENCE_CHUNK_ROWS
            ):
                guard.raise_if_failed()
                guard.check_batch_boundary()
                cpu_chunk = torch_module.from_numpy(raw_batch[:, time_start:time_stop, :])
                device_chunk = cpu_chunk.to("cuda", non_blocking=False)
                model_input = module.build_inputs(device_chunk)
                if model_input.shape != (
                    batch_rows,
                    SEQUENCE_CHUNK_ROWS,
                    int(module.N_FEATURES + module.N_EVENT_FEATURES),
                ):
                    raise FitAbort(
                        "export model-input shape differs from the pinned raw-only trainer"
                    )
                with (
                    torch_module.inference_mode(),
                    torch_module.autocast("cuda", dtype=torch_module.float16, enabled=True),
                ):
                    output, state = module.forward_chunk(
                        model, model_input, slice(0, time_stop - time_start), state
                    )
                    predictions = (
                        module.readout(output, *module.READOUT, tilted=module.READOUT_TILTED)
                        if module.READOUT is not None
                        else output[..., :2].float()
                    )
                values = predictions.detach().float().cpu().numpy()
                validate_prediction_chunk(
                    values,
                    expected_shape=(batch_rows, SEQUENCE_CHUNK_ROWS, 2),
                )
                mapped[batch_start:batch_stop, time_start:time_stop, :] = values
                del cpu_chunk, device_chunk, model_input, output, predictions, values
                _aligned_flush_and_advise(
                    mapped,
                    descriptor,
                    first_row=batch_start,
                    last_row=batch_stop,
                    sequence_rows=sequence_rows,
                )
                pagecache_advice_count += 1
                written_chunks += 1
                guard.check_batch_boundary()
            written_rows = batch_stop
            remaining_payload_bytes -= batch_rows * sequence_rows * 2 * PREDICTION_BYTES_PER_VALUE
            _atomic_json(
                progress_path,
                {
                    "phase": "prediction_export",
                    "run_name": run_name,
                    "seed": seed,
                    "split": split_name,
                    "rows_written": written_rows,
                    "expected_rows": rows,
                    "chunks_written": written_chunks,
                    "expected_chunks": ((rows + TRAIN_BATCH_ROWS - 1) // TRAIN_BATCH_ROWS)
                    * (sequence_rows // SEQUENCE_CHUNK_ROWS),
                    "payload_bytes": payload_bytes,
                    "disk_free_bytes": shutil.disk_usage(working).free,
                },
            )
            del raw_batch
        if written_rows != rows:
            raise FitAbort("prediction export did not cover every target row")
        mapped.flush()
        mapped._mmap.close()
        del mapped
        mapped = None
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        reader = None
        os.replace(partial, final)
        _fsync_directory(final.parent)
        if final.stat().st_size < payload_bytes:
            raise FitAbort("final prediction file is shorter than its declared tensor payload")
        hash_advice_path = receipts / f"hash_io_advice_{run_name}_{split_name}.json"
        output_sha = _hash_file_releasing_cache(final, hash_advice_path, f"{run_name}/{split_name}")
        export_receipt: dict[str, Any] = {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "run_name": run_name,
            "seed": seed,
            "split": split_name,
            "path": str(final),
            "format": "numpy-npy-float32",
            "shape": list(expected_shape),
            "dtype": "float32",
            "row_group_order_sha256": order["row_groups_sha256"],
            "target_index_sha256": order["target_index_sha256"],
            "target_cache_sha256": order["target_cache_sha256"],
            "split_manifest_sha256": split_sha256,
            "checkpoint_sha256": checkpoint_sha256,
            "trainer_provenance_sha256": trainer_provenance_sha256,
            "trainer_argv_sha256": fit_argv_sha256,
            "trainer_source_sha256": trainer_sha256,
            "source_archive_sha256": source_archive_sha256,
            "input_identity_sha256": input_identity_sha256,
            "selected_epoch": 3,
            "prediction_sha256": output_sha,
            "prediction_payload_bytes": payload_bytes,
            "file_bytes": final.stat().st_size,
            "rows_written": written_rows,
            "chunk_rows": SEQUENCE_CHUNK_ROWS,
            "max_batch_rows": TRAIN_BATCH_ROWS,
            "finite_checked_per_chunk": True,
            "official_metric_clipping_applied": False,
            "hash_pagecache_advice_receipt": str(hash_advice_path),
            "hash_pagecache_advice_sha256": sha256_file(hash_advice_path),
            "write_pagecache_advice": "POSIX_FADV_DONTNEED completed per row-batch/chunk; kernel hint only",
            "status": "complete",
        }
        sidecar_partial = sidecar.with_name(sidecar.name + ".partial")
        _atomic_json(sidecar_partial, export_receipt)
        os.replace(sidecar_partial, sidecar)
        _fsync_directory(sidecar.parent)
        export_receipt["sidecar_path"] = str(sidecar)
        export_receipt["sidecar_sha256"] = sha256_file(sidecar)
        export_receipt["pagecache_advice_calls"] = pagecache_advice_count
        return export_receipt, remaining_payload_bytes
    except Exception:
        if mapped is not None:
            mapped._mmap.close()
            del mapped
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if reader is not None:
            del reader


def _run_one_seed(
    *,
    seed: int,
    run_name: str,
    source_root: Path,
    orders: dict[str, dict[str, Any]],
    split: Any,
    output_root: Path,
    working: Path,
    receipts: Path,
    guard: FitResourceGuard,
    torch_module: Any,
    split_sha256: str,
    input_identity_sha256: str,
    source_archive_sha256: str,
    trainer_sha256: str,
    remaining_payload_bytes: int,
) -> tuple[dict[str, Any], int]:
    """Fit one scratch model, verify fixed checkpoint, and export both target orders."""
    run_dir = source_root / "artifacts" / "runs" / run_name
    if run_dir.exists():
        raise FitAbort(f"run output already exists and will not be reused: {run_name}")
    argv = trainer_argv(seed, run_name)
    argv_sha = canonical_sha256(argv)
    _atomic_json(
        receipts / "phase.json",
        {
            "phase": "fit_starting",
            "seed": seed,
            "run_name": run_name,
            "argv": argv,
            "argv_sha256": argv_sha,
            "fit_start_unix": time.time(),
        },
    )
    trainer_module = _trainer_module(source_root, f"seed{seed}")
    original_stop = trainer_module.StageTimer.stop

    def checked_stop(timer: Any) -> None:
        original_stop(timer)
        guard.raise_if_failed()
        guard.check_batch_boundary()

    trainer_module.StageTimer.stop = checked_stop
    restore_save = _atomic_torch_save(torch_module)
    old_argv = sys.argv
    old_cwd = Path.cwd()
    torch_module.cuda.reset_peak_memory_stats(0)
    fit_start = time.monotonic()
    try:
        os.chdir(source_root)
        sys.argv = argv[1:]
        trainer_module.main()
        guard.raise_if_failed()
        fit_seconds = time.monotonic() - fit_start
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)
        trainer_module.StageTimer.stop = original_stop
        restore_save()
    guard.check_full()
    run_dir = source_root / "artifacts" / "runs" / run_name
    checkpoint_path = run_dir / "model.pt"
    epoch_checkpoint_path = run_dir / "model_ep3.pt"
    provenance_path = run_dir / "provenance.json"
    if any(
        not path.is_file() for path in (checkpoint_path, epoch_checkpoint_path, provenance_path)
    ):
        raise FitAbort(f"fixed epoch-3 outputs are incomplete for {run_name}")
    provenance_bytes = provenance_path.read_bytes()
    provenance = json.loads(provenance_bytes)
    checkpoint_sha = sha256_file(checkpoint_path)
    epoch_sha = sha256_file(epoch_checkpoint_path)
    if checkpoint_sha != provenance.get("checkpoint_sha256", {}).get("model.pt"):
        raise FitAbort("trainer provenance checkpoint hash disagrees with model.pt bytes")
    if epoch_sha != provenance.get("checkpoint_sha256", {}).get("model_ep3.pt"):
        raise FitAbort("trainer provenance epoch checkpoint hash disagrees with model_ep3.pt bytes")
    checkpoint = torch_module.load(checkpoint_path, map_location="cpu", weights_only=False)
    fit_identity = _validate_fit_provenance(provenance, checkpoint, seed, split)
    fit_provenance_sha = sha256_bytes(provenance_bytes)
    peak = {
        "cuda_allocated_peak_bytes": int(torch_module.cuda.max_memory_allocated(0)),
        "cuda_reserved_peak_bytes": int(torch_module.cuda.max_memory_reserved(0)),
    }
    if peak["cuda_allocated_peak_bytes"] > input_identity.CAPS["cuda_allocated_bytes"]:
        raise FitAbort("fit allocated-memory high-water exceeded 2 GB")
    if peak["cuda_reserved_peak_bytes"] > input_identity.CAPS["cuda_reserved_bytes"]:
        raise FitAbort("fit reserved-memory high-water exceeded 2.5 GB")
    _atomic_json(
        receipts / "phase.json",
        {
            "phase": "fit_completed",
            "seed": seed,
            "run_name": run_name,
            "fit_seconds": round(fit_seconds, 3),
            "checkpoint_sha256": checkpoint_sha,
            "model_ep3_sha256": epoch_sha,
            "trainer_provenance_sha256": fit_provenance_sha,
            "selected_epoch": 3,
            "fit_identity": fit_identity,
            "cuda_peak": peak,
        },
    )
    del checkpoint
    model, loaded_checkpoint = _load_checkpoint(
        trainer_module, checkpoint_path, torch_module.device("cuda")
    )
    if loaded_checkpoint.get("epoch") != 3:
        raise FitAbort("reloaded export checkpoint is not fixed epoch 3")
    member: dict[str, Any] = {
        "seed": seed,
        "run_name": run_name,
        "fit_argv": argv,
        "fit_argv_sha256": argv_sha,
        "fit_seconds": round(fit_seconds, 3),
        "selected_epoch": 3,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "epoch_checkpoint_path": str(epoch_checkpoint_path),
        "epoch_checkpoint_sha256": epoch_sha,
        "trainer_provenance_path": str(provenance_path),
        "trainer_provenance_sha256": fit_provenance_sha,
        "input_identity_sha256": input_identity_sha256,
        "source_archive_sha256": source_archive_sha256,
        "trainer_source_sha256": trainer_sha256,
        "fit_identity": fit_identity,
        "cuda_peak": peak,
        "exports": {},
    }
    for split_name in ("train", "valid"):
        export, remaining_payload_bytes = _write_split_predictions(
            model=model,
            module=trainer_module,
            torch_module=torch_module,
            source_root=source_root,
            order=orders[split_name],
            split_name=split_name,
            run_name=run_name,
            seed=seed,
            checkpoint_sha256=checkpoint_sha,
            trainer_provenance_sha256=fit_provenance_sha,
            split_sha256=split_sha256,
            output_root=output_root,
            working=working,
            receipts=receipts,
            guard=guard,
            remaining_payload_bytes=remaining_payload_bytes,
            input_identity_sha256=input_identity_sha256,
            source_archive_sha256=source_archive_sha256,
            trainer_sha256=trainer_sha256,
            fit_argv_sha256=argv_sha,
        )
        member["exports"][split_name] = export
    del model, loaded_checkpoint, trainer_module
    torch_module.cuda.synchronize(0)
    torch_module.cuda.empty_cache()
    guard.check_full()
    return member, remaining_payload_bytes


def run_fit(packet_root: Path) -> dict[str, Any]:
    """Run the admitted serial fit pair and return its durable top-level receipt."""
    packet_root = packet_root.resolve()
    working = Path(os.environ.get("KAGGLE_WORKING", "/kaggle/working")).resolve()
    if not working.is_dir():
        raise FitAbort(f"working directory is missing: {working}")
    output = working / OUTPUT_SUBDIR
    receipts = output / "receipts"
    output_root = output / "predictions"
    receipts.mkdir(parents=True, exist_ok=False)
    output_root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    receipt_path = receipts / "fit_pair_receipt.json"
    handoff_path = receipts / "resource_monitor_handoff.json"
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": "starting",
        "fit_started": False,
        "score_started": False,
        "retry_count": 0,
        "members": [],
        "phase_receipts": {},
        "created_unix": time.time(),
    }
    _atomic_json(receipt_path, receipt)
    guard: FitResourceGuard | None = None
    staged: dict[str, Any] | None = None
    signal_previous = signal.getsignal(signal.SIGTERM)

    def terminate_handler(_signum: int, _frame: Any) -> None:
        raise FitAbort("worker received SIGTERM; partial checkpoint/output files are not complete")

    signal.signal(signal.SIGTERM, terminate_handler)
    try:
        # Guard hashing and identity validation too; explicit pre-CUDA idle is
        # the only active-GPU measurement allowed before torch initializes CUDA.
        guard = FitResourceGuard(
            None,
            working,
            receipts,
            MAX_WORKER_WALL_SECONDS,
            started_at=started,
        )
        guard.start()
        config, packet_meta, review_sha, admission_sha = _required_receipts(packet_root)
        for receipt_name in (REVIEW_NAME, ADMISSION_NAME, PACKET_META_NAME):
            source_receipt = packet_root / receipt_name
            destination_receipt = receipts / receipt_name
            shutil.copyfile(source_receipt, destination_receipt)
            with destination_receipt.open("rb") as stream:
                os.fsync(stream.fileno())
        _fsync_directory(receipts)
        receipt.update(
            {
                "phase": "gate_receipts_validated",
                "candidate_artifact_hashes": packet_meta["artifact_hashes"],
                "external_review_sha256": review_sha,
                "controller_admission_sha256": admission_sha,
                "profile_receipt_sha256": packet_meta["profile_receipt_sha256"],
                "grader_script_sha256": packet_meta["grader_script_sha256"],
                "anchor_report_sha256": config["evaluation"]["anchor_evaluation_report_sha256"],
            }
        )
        _atomic_json(receipt_path, receipt)
        wall_cap = int(packet_meta["worker_wall_cap_seconds"])
        if not 0 < wall_cap <= config["resource_policy"]["max_worker_wall_seconds"]:
            raise FitGateError("packet wall cap exceeds the frozen E520 worker limit")
        if time.monotonic() - started > wall_cap:
            raise FitGateError(
                "pre-fit identity work exceeded the externally admitted worker wall cap"
            )
        guard.wall_cap = wall_cap
        source_root = working / "e520_source"
        archive_path = packet_root / "source_archive.tar"
        archive_hash = sha256_file(archive_path)
        if archive_hash != config["source"]["archive_sha256"]:
            raise FitAbort("pinned trainer source archive hash mismatch")
        _safe_extract_archive(archive_path, source_root)
        identity_path = receipts / "input_identity.json"
        staged = input_identity.verify_and_stage_inputs(
            Path(os.environ.get("KAGGLE_INPUT", "/kaggle/input")),
            source_root,
            identity_path,
        )
        digest_handoff_receipt = _write_digest_handoff_runtime(
            staged, receipts, phase="initial_split_load_complete"
        )
        identity_bytes = identity_path.read_bytes()
        input_orders = _input_order_receipts(source_root, staged["identity"], config)
        receipt.update(
            {
                "phase": "input_identity_verified",
                "input_identity_path": str(identity_path),
                "input_identity_sha256": sha256_bytes(identity_bytes),
                "input_identity": staged["identity"],
                "verified_digest_handoff_runtime": digest_handoff_receipt,
                "source_archive_sha256": archive_hash,
                "trainer_source_sha256": config["source"]["trainer_sha256"],
                "target_orders": {
                    name: {
                        key: value
                        for key, value in order.items()
                        if key not in ("row_groups", "seq_ix")
                    }
                    for name, order in input_orders.items()
                },
            }
        )
        _atomic_json(receipt_path, receipt)
        source_trainer = source_root / "scripts" / "train_rnn.py"
        if sha256_file(source_trainer) != config["source"]["trainer_sha256"]:
            raise FitAbort("trainer source hash changed after archive extraction")
        expected_payload = sum(
            order["row_group_count"] * 20_000 * 2 * PREDICTION_BYTES_PER_VALUE
            for order in input_orders.values()
        ) * len(config["recipe"]["seeds"])
        min_free = int(config["resource_policy"]["min_disk_free_bytes"])
        free = shutil.disk_usage(working).free
        if free < expected_payload + min_free + 512 * 1024**2:
            raise FitAbort(
                "pre-fit disk gate failed: complete streamed prediction pair plus 8 GB floor and 512 MB margin do not fit"
            )
        # CUDA is initialized only after the code, review, admission, source,
        # and inputs pass. The explicit-idle monitor is stopped before it.
        pre_cuda_guard = guard
        try:
            stop_pre_cuda_guard(pre_cuda_guard, receipts, started)
            import torch

            if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                raise FitAbort("E520 fit requires exactly one visible CUDA GPU")
            torch.cuda.init()
            runtime = input_identity.runtime_identity(torch)
            _atomic_json(receipts / "environment.json", runtime)
            guard = start_active_guard(
                torch,
                working,
                receipts,
                wall_cap,
                started,
                runtime,
            )
        except Exception as error:
            if not (receipts / "resource_abort.json").exists():
                _atomic_json(
                    receipts / "resource_abort.json",
                    {
                        "status": "aborted",
                        "phase": "cuda_startup_handoff",
                        "reason": f"{type(error).__name__}: {error}",
                        "elapsed_seconds": round(time.monotonic() - started, 3),
                        "last_snapshot": pre_cuda_guard.latest,
                        "runtime_identity_path": str(receipts / "environment.json")
                        if (receipts / "environment.json").is_file()
                        else None,
                        "runtime_identity_sha256": sha256_file(receipts / "environment.json")
                        if (receipts / "environment.json").is_file()
                        else None,
                    },
                )
            raise
        receipt.update(
            {
                "phase": "live_resource_gates_passed",
                "runtime_identity_path": str(receipts / "environment.json"),
                "runtime_identity_sha256": sha256_file(receipts / "environment.json"),
                "pre_fit_disk_free_bytes": free,
                "prediction_payload_bytes_required": expected_payload,
                "worker_wall_cap_seconds": wall_cap,
                "fit_started": True,
                "fit_order": ["seed3", "seed4"],
                "parallel_fits": False,
                "resource_monitor_handoff_path": str(handoff_path),
                "resource_monitor_handoff_sha256": sha256_file(handoff_path),
            }
        )
        _atomic_json(receipt_path, receipt)
        remaining_payload = expected_payload
        for seed, run_name in zip(
            config["recipe"]["seeds"], config["recipe"]["run_names"], strict=True
        ):
            guard.raise_if_failed()
            member, remaining_payload = _run_one_seed(
                seed=int(seed),
                run_name=str(run_name),
                source_root=source_root,
                orders=input_orders,
                split=staged["split"],
                output_root=output_root,
                working=working,
                receipts=receipts,
                guard=guard,
                torch_module=torch,
                split_sha256=staged["split"].sha256,
                input_identity_sha256=receipt["input_identity_sha256"],
                source_archive_sha256=archive_hash,
                trainer_sha256=config["source"]["trainer_sha256"],
                remaining_payload_bytes=remaining_payload,
            )
            receipt["members"].append(member)
            receipt["phase"] = f"seed{seed}_fit_and_exports_completed"
            receipt["elapsed_seconds"] = round(time.monotonic() - started, 3)
            receipt["verified_digest_handoff_runtime"] = _write_digest_handoff_runtime(
                staged, receipts, phase=f"seed{seed}_completed"
            )
            _atomic_json(receipt_path, receipt)
        guard.check_full()
        if remaining_payload != 0:
            raise FitAbort("paired export byte accounting did not close at zero")
        receipt.update(
            {
                "status": "complete",
                "phase": "fit_pair_and_exports_complete_no_score",
                "fit_started": True,
                "score_started": False,
                "retry_count": 0,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "terminal_resource_snapshot": guard.latest,
                "prediction_payload_bytes": expected_payload,
            }
        )
        _atomic_json(receipt_path, receipt)
        return receipt
    except Exception as error:
        if handoff_path.is_file():
            receipt["resource_monitor_handoff_path"] = str(handoff_path)
            receipt["resource_monitor_handoff_sha256"] = sha256_file(handoff_path)
        if staged is not None:
            try:
                receipt["verified_digest_handoff_runtime"] = _write_digest_handoff_runtime(
                    staged, receipts, phase="worker_aborted"
                )
            except Exception as telemetry_error:
                receipt["verified_digest_handoff_persist_error"] = (
                    f"{type(telemetry_error).__name__}: {telemetry_error}"
                )
        receipt.update(
            {
                "status": "aborted",
                "phase": "aborted_no_retry",
                "fit_started": receipt.get("fit_started", False),
                "score_started": False,
                "retry_count": 0,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "abort_type": type(error).__name__,
                "abort_reason": str(error),
                "resource_abort_path": str(receipts / "resource_abort.json")
                if (receipts / "resource_abort.json").exists()
                else None,
            }
        )
        _atomic_json(receipt_path, receipt)
        raise
    finally:
        if guard is not None:
            guard.stop()
        signal.signal(signal.SIGTERM, signal_previous)


def main() -> None:
    """Run exactly the admitted seed pair and print only the durable receipt pointer."""
    result = run_fit(Path(__file__).resolve().parent)
    print(
        f"E520 fit status={result['status']} elapsed_seconds={result['elapsed_seconds']:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
