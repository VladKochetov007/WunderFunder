from __future__ import annotations

import hashlib
import json
import operator
import os
import platform
import stat
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from wnn.propensity_recovery_validation import validate_resume_boundary
from wnn.run_identity import normalization_identity
from wnn.splits import CampaignSplit, file_sha256
from wnn.training_state import (
    atomic_save_training_state,
    capture_rng,
    loader_state,
    restore_loader,
    restore_rng,
)

STATE_FORMAT = "wnn.strict-propensity-training-state"
STATE_VERSION = 1
MAX_STATE_BYTES = 1_000_000_000


def _cpu_copy(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    return value


def _fit_dataset_indices(loader: DataLoader, split: CampaignSplit) -> list[int]:
    """Require the strict loader's declared row-group order to match the split."""
    indices = getattr(loader.dataset, "indices", None)
    if indices is None:
        raise ValueError("strict propensity fit dataset does not expose ordered row-group indices")
    try:
        actual = [operator.index(index) for index in indices]
    except (TypeError, ValueError) as error:
        raise ValueError("strict propensity fit dataset indices are invalid") from error
    expected = list(map(int, split.valid_groups))
    if actual != expected:
        raise ValueError("strict propensity loader indices differ from ordered split.valid_groups")
    return actual


def build_static_contract(
    *,
    args: dict[str, object],
    provenance: dict[str, object],
    split: CampaignSplit,
    data_root: Path,
    source_root: Path,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, object]:
    """Bind resume identity to exact code, split, data source, options and runtime."""
    source_manifest = provenance.get("source_sha256")
    if not isinstance(source_manifest, dict) or not source_manifest:
        raise ValueError("strict propensity source manifest is missing")
    for relative, expected in source_manifest.items():
        source_path = Path(relative)
        if source_path.is_absolute() or ".." in source_path.parts:
            raise ValueError("strict propensity source path escapes its source root")
        if file_sha256(source_root / source_path) != expected:
            raise ValueError(f"strict propensity source changed: {relative}")
    split_before = file_sha256(split.path)
    if split_before != split.sha256:
        raise ValueError("strict propensity split changed before contract construction")
    split_doc = json.loads(split.path.read_text())
    if file_sha256(split.path) != split_before:
        raise ValueError("strict propensity split changed during contract construction")
    fit_groups = np.asarray(split.valid_groups, dtype="<i8")
    dataset_fit_groups = np.asarray(_fit_dataset_indices(loader, split), dtype="<i8")
    evaluation_groups = np.asarray(split.evaluation_groups, dtype="<i8")
    component_bytes = json.dumps(split.components, separators=(",", ":")).encode()
    portable_path = os.environ.get("WNN_PORTABLE_DATA_MANIFEST")
    portable = None
    if portable_path:
        portable_file = Path(portable_path).resolve()
        portable = {"path": str(portable_file), "sha256": file_sha256(portable_file)}
    options = {key: value for key, value in args.items() if key != "resume"}
    options = json.loads(json.dumps(options, default=str, sort_keys=True))
    runtime = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "device": str(device),
        "cuda_runtime": torch.version.cuda,
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_initialized() else 0,
        "threads": torch.get_num_threads(),
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cudnn_version": torch.backends.cudnn.version(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        runtime["device_name"] = properties.name
        runtime["device_capability"] = list(torch.cuda.get_device_capability(device))
    optimizer_defaults = json.loads(json.dumps(optimizer.defaults, default=str, sort_keys=True))
    return {
        "options": options,
        "data_root": str(data_root.resolve()),
        "source_root": str(source_root.resolve()),
        "source_sha256": source_manifest,
        "split": {
            "path": str(split.path),
            "sha256": split.sha256,
            "fold": split.fold,
            "n_folds": split.n_folds,
            "sources": split_doc["sources"],
            "fit_valid_count": len(fit_groups),
            "fit_valid_ordered_int64_le_sha256": hashlib.sha256(fit_groups.tobytes()).hexdigest(),
            "loader_fit_valid_count": len(dataset_fit_groups),
            "loader_fit_valid_ordered_int64_le_sha256": hashlib.sha256(
                dataset_fit_groups.tobytes()
            ).hexdigest(),
            "evaluation_count": len(evaluation_groups),
            "evaluation_ordered_int64_le_sha256": hashlib.sha256(
                evaluation_groups.tobytes()
            ).hexdigest(),
            "component_count": len(split.components),
            "component_mapping_sha256": hashlib.sha256(component_bytes).hexdigest(),
            "portable_registry": portable,
        },
        "model": {
            "class": "PropensityNet",
            "hidden": int(options["hidden"]),
            "n_in": int(provenance["n_in"]),
        },
        "optimizer": {"class": "AdamW", "defaults": optimizer_defaults},
        "scheduler": {"kind": "none", "reason": "fixed learning rate; no scheduler is used"},
        "loader": loader_state(loader),
        "runtime": runtime,
    }


def make_resume_state(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    contract: dict[str, object],
    next_epoch: int,
    progress: dict[str, object],
    history: list[dict[str, object]],
    provenance: dict[str, object],
    elapsed_seconds: float,
    attempt_id: str,
    event_sequence: int,
) -> dict[str, object]:
    """Capture a portable epoch-boundary state for exact restart of the next epoch."""
    return _cpu_copy(
        {
            "format": STATE_FORMAT,
            "version": STATE_VERSION,
            "boundary": "completed_epoch",
            "contract": contract,
            "next_epoch": next_epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler_state": None,
            "loader": loader_state(loader),
            "rng": capture_rng(),
            "progress": progress,
            "history": history,
            "provenance": provenance,
            "elapsed_seconds": elapsed_seconds,
            "attempt_id": attempt_id,
            "event_sequence": event_sequence,
        }
    )


def load_resume_state(path: Path, static_contract: dict[str, object]) -> dict[str, object]:
    """Load and validate a bounded local strict propensity checkpoint without symlinks."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_STATE_BYTES:
            raise ValueError("strict propensity training state is not a bounded regular file")
        with os.fdopen(os.dup(descriptor), "rb") as stream:
            state = torch.load(stream, map_location="cpu", weights_only=False)
        after = os.fstat(descriptor)
        path_stat = path.lstat()
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ) or (path_stat.st_dev, path_stat.st_ino) != (after.st_dev, after.st_ino):
            raise ValueError("strict propensity training state changed during read")
    finally:
        os.close(descriptor)
    required = {
        "contract",
        "next_epoch",
        "model",
        "optimizer",
        "scheduler_state",
        "loader",
        "rng",
        "progress",
        "history",
        "provenance",
        "elapsed_seconds",
        "attempt_id",
        "event_sequence",
    }
    if (
        not isinstance(state, dict)
        or state.get("format") != STATE_FORMAT
        or state.get("version") != STATE_VERSION
        or state.get("boundary") != "completed_epoch"
        or required - state.keys()
    ):
        raise ValueError("strict propensity resume state is incomplete or incompatible")
    if (
        not isinstance(state["contract"], dict)
        or state["contract"].get("static") != static_contract
        or state["scheduler_state"] is not None
    ):
        raise ValueError("strict propensity resume contract differs")
    validate_resume_boundary(state, static_contract, path.parent)
    return state


def restore_resume_state(
    state: dict[str, object],
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
) -> None:
    """Restore model, optimizer, loader, then RNG as the last construction action."""
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    restore_loader(loader, state["loader"])
    if normalization_identity(model) != state["contract"]["normalization_identity"]:
        raise ValueError("strict propensity normalization buffers differ from the checkpoint")
    restore_rng(state["rng"])


def save_resume_state(
    state: dict[str, object], path: Path, *, on_saved: Callable[[], None] | None = None
) -> str:
    """Atomically save a CPU state and return its exact file digest."""
    atomic_save_training_state(state, path)
    if on_saved is not None:
        on_saved()
    return file_sha256(path)


def atomic_write_json(path: Path, value: dict[str, object]) -> None:
    """Atomically replace a JSON receipt in the owned strict run directory."""
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
