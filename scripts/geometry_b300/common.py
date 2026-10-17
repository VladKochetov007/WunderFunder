"""Shared command and provenance helpers for the B300 geometry handoff."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mounts import mount_for_path, read_mount_table, require_mount_mode

ROOT = Path(__file__).resolve().parents[2]
SPLIT_DIGESTS = {
    0: "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d",
    1: "31b9dc040761dcc81e99bfe69e0bc446f1522b21ab629b2a9c5a03d94bb006ae",
    2: "a9973e5798d3109362571c299b2d08eca71d01e92008da39b0d0ddde54c93aa9",
    3: "78157b3347b9e9ba6d2c075d0c53df612e67af9ecd5ee8af7fd7957e406b8343",
}
CONTROL_BY_ARM = {
    "baseline": "all",
    "moments": "moments_only",
    "dictionary": "dictionary_only",
    "combined": "none",
}
PROFILE_CONFIG = {
    "hidden": 512,
    "layers": 1,
    "batch": 8,
    "chunk_rows": 2000,
    "raw_and_event_width": 211,
    "geometry_feature_width": 60,
    "model_input_width": 271,
    "precision": "fp32",
    "epochs": 3,
    "checkpoint_epoch": 2,
    "optimizer": "AdamW",
    "scheduler": "OneCycle",
    "learning_rate": 0.001,
    "weight_decay": 0.01,
    "gradient_clip": 1.0,
    "train_sequences": 10607,
    "profile_warmup_updates": 2,
    "profile_measured_updates": 8,
    "arms": 4,
    "cells_per_arm": 30,
    "full_matrix_fits": 120,
    "waves": 15,
    "workers_per_wave": 8,
    "expected_optimizer_updates_by_fold": {"0": 44760, "1": 44790, "2": 44850, "3": 44640},
    "full_matrix_optimizer_updates": 5371320,
}


def sha256_file(path: Path) -> str:
    """Hash a file using bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Atomically replace a JSON receipt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def utc_now() -> str:
    """Return an ISO timestamp in UTC."""
    return datetime.now(UTC).isoformat()


def git_value(root: Path, *arguments: str) -> str:
    """Read one Git value from the pinned checkout."""
    return subprocess.check_output(["git", *arguments], cwd=root, text=True).strip()


def checkout_commit(root: Path) -> str:
    """Require a clean tracked checkout and return its commit SHA."""
    dirty = git_value(root, "status", "--porcelain")
    if dirty:
        raise RuntimeError("B300 runner requires a clean committed source checkout")
    return git_value(root, "rev-parse", "HEAD")


def pair_identity(pair_index: int) -> tuple[int, int]:
    """Return the frozen crossed fold and fitting-seed identity for one cell."""
    if not 0 <= pair_index < 30:
        raise ValueError("pair index must be between 0 and 29")
    return pair_index % 4, 1001 + pair_index // 4


def validate_mounts(root: Path, fold: int, manifest_override: Path | None) -> Path:
    """Check required mount targets and small source-identity manifests."""
    mounts = read_mount_table()
    input_paths = [root / "datasets" / f"{name}.parquet" for name in ("train", "valid")]
    input_paths.extend(
        root / "cache" / name
        for name in ("train_targets.npz", "valid_targets.npz", "chains.npz", "chain_folds.npz")
    )
    for path in input_paths:
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"required read-only input is missing: {path}")
        require_mount_mode(path, mounts, "ro")
    split_path = root / f"research/campaign_2026-09-29/artifacts/splits/fold{fold}.json"
    if sha256_file(split_path) != SPLIT_DIGESTS[fold]:
        raise ValueError(f"frozen split manifest changed: {split_path}")
    split = json.loads(split_path.read_text())
    portable_dir = Path(os.environ.get("WNN_PORTABLE_INPUT_DIR", "/mnt/geometry-inputs"))
    registry = manifest_override or portable_dir / f"portable_inputs_fold{fold}.json"
    if not registry.is_file():
        raise FileNotFoundError(f"portable source registry is missing: {registry}")
    require_mount_mode(registry, mounts, "ro")
    payload = json.loads(registry.read_text())
    if payload.get("version") != 1 or payload.get("split_manifest_sha256") != SPLIT_DIGESTS[fold]:
        raise ValueError("portable source registry is not bound to this frozen split")
    entries = payload.get("sources", {})
    for name, identity in split["sources"].items():
        if entries.get(name, {}).get("original_identity") != identity:
            raise ValueError(f"portable source identity does not match frozen split: {name}")
    for directory in (root / "artifacts", root / "cache"):
        directory.mkdir(parents=True, exist_ok=True)
        if not os.access(directory, os.W_OK):
            raise PermissionError(f"persistent output mount is not writable: {directory}")
        require_mount_mode(directory, mounts, "rw")
    scratch_text = os.environ.get("WNN_B300_SCRATCH")
    if not scratch_text:
        raise OSError("set WNN_B300_SCRATCH to a writable physical-disk mount")
    scratch = Path(scratch_text).resolve()
    scratch_mount = mount_for_path(scratch, mounts)
    if scratch_mount.filesystem in {"tmpfs", "ramfs"}:
        raise ValueError("WNN_B300_SCRATCH resolves to RAM-backed storage")
    require_mount_mode(scratch, mounts, "rw")
    scratch.mkdir(parents=True, exist_ok=True)
    return registry.resolve()


def execution_environment(root: Path, registry: Path) -> dict[str, str]:
    """Set project, portable-data and bounded CPU/scratch paths for child tools."""
    environment = os.environ.copy()
    scratch = str(Path(os.environ["WNN_B300_SCRATCH"]).resolve())
    environment.update(
        {
            "WNN_PROJECT_ROOT": str(root),
            "WNN_PORTABLE_DATA_MANIFEST": str(registry),
            "TMPDIR": scratch,
            "TMP": scratch,
            "TEMP": scratch,
            "WNN_TMP_ROOT": scratch,
            "TORCHINDUCTOR_CACHE_DIR": str(Path(scratch) / "torchinductor"),
            "WNN_LOADER_WORKERS": "0",
            "WNN_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMBA_NUM_THREADS": "1",
            "POLARS_MAX_THREADS": "1",
            "ARROW_NUM_THREADS": "1",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "PYTHONUNBUFFERED": "1",
        }
    )
    Path(environment["TORCHINDUCTOR_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    return environment


def gpu_identity() -> dict[str, Any]:
    """Verify a single visible CUDA card and record its immutable identity."""
    gpu_id = os.environ.get("WNN_GPU_ID", "")
    if not gpu_id or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise OSError("set WNN_GPU_ID and CUDA_VISIBLE_DEVICES for one physical card")
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("each B300 worker must see exactly one CUDA device")
    device = torch.cuda.get_device_properties(0)
    return {
        "gpu_id": gpu_id,
        "visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "name": device.name,
        "total_memory_bytes": int(device.total_memory),
        "compute_capability": [int(device.major), int(device.minor)],
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
    }


def train_command(name: str, fold: int, seed: int, control: str) -> list[str]:
    """Build the frozen fresh-H512 training command for one scientific arm."""
    return [
        sys.executable,
        "-u",
        "scripts/train_rnn.py",
        "--name",
        name,
        "--device",
        "cuda",
        "--hidden",
        "512",
        "--layers",
        "1",
        "--batch",
        "8",
        "--chunk",
        "2000",
        "--epochs",
        "3",
        "--checkpoint-policy",
        "fixed",
        "--checkpoint-epoch",
        "2",
        "--no-eval-during-fit",
        "--save-training-state",
        "--loss",
        "wmse",
        "--optim",
        "adamw",
        "--sched",
        "onecycle",
        "--lr",
        "0.001",
        "--wd",
        "0.01",
        "--clip",
        "1.0",
        "--precision",
        "fp32",
        "--grad-stats",
        "--deterministic",
        "--seed",
        str(seed),
        "--split-manifest",
        f"research/campaign_2026-09-29/artifacts/splits/fold{fold}.json",
        "--train-source",
        "train+valid",
        "--train-seqs",
        "10607",
        "--research-features",
        "causal_geometry_discrete",
        "--research-feature-control",
        control,
    ]


def prediction_command(name: str, fold: int) -> list[str]:
    """Build the exact held-out prediction command for a fixed epoch-2 model."""
    split = f"research/campaign_2026-09-29/artifacts/splits/fold{fold}.json"
    return [
        sys.executable,
        "-u",
        "scripts/predict_run.py",
        name,
        "valid",
        "--campaign-split-manifest",
        split,
        "--campaign-evaluation-only",
        "--device",
        "cuda",
        "--output-name",
        name,
        "--epoch",
        "2",
    ]


def score_command(name: str, fold: int) -> list[str]:
    """Build the unchanged official global_wp_clip2_v1 grading command."""
    split = f"research/campaign_2026-09-29/artifacts/splits/fold{fold}.json"
    return [
        sys.executable,
        "-u",
        "scripts/score_run.py",
        "--run",
        name,
        "--split",
        split,
        "--epoch",
        "2",
    ]
