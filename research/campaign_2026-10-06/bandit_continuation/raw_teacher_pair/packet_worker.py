from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import TextIO

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

PACKET_ID = "E508_RAW_BASELINE_V3"
SPLIT_SHA256 = "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d"
ASSETS_SHA256 = "19fe0eb6445553c2adff00e821a4fa16ca5969473ca8be420c32cb8e4b4b72c7"
ASSETS_BYTES = 6268
PORTABLE_SHA256 = "90d26c7063050092eeda774c7c6298897a8fb133e4804ac13d0a281bd328ff8c"
SOURCE_MANIFEST_SHA256 = "851898475dfa701f2089ec76cfe9592538aa2e7f9b70ebcb68c0a614e332d6fb"
RAW_DATASET_REF = "vladkochetov07/wunder-funder"
RAW_DATASET_VERSION_AT_PREREGISTRATION = 1
RAW_DATASET_VERSION_OBSERVED_AT_UTC = "2026-10-07T00:11:43+00:00"
AUDIT_CHECKPOINT_RELATIVE = "artifacts/runs/c20261003_combo_s3_f0/model_ep3.pt"
AUDIT_CHECKPOINT_SHA256 = "d4a1e50b1892b4d9edd7c255908b68073a760c059ca80223e98a63c7e023371e"
AUDIT_CHECKPOINT_BYTES = 3_977_183
TRAIN_BYTES = 29_240_219_011
VALID_BYTES = 5_615_509_128
TARGET_ORDERS = {"train": 10_606, "valid": 1_873}
FIT_COUNTS = {"train": 10_535, "valid": 1_405}
EVAL_VALID_COUNT = 468
CORRUPT_TRAIN_GROUP = 528
SEEDS = (3, 4)
EPOCHS = 4
CHECKPOINT_EPOCH = 3
BATCH = 16
CHUNK = 2_000
MAX_PRODUCTION_WALL_SECONDS = 6_300
PROFILE_ONLY = True
OUTPUT_MIN_FREE_BYTES = 1_500_000_000
GPU_ALLOCATOR_BYTES = 4_300_000_000

ROOT = Path(__file__).resolve().parent
PHASE_FILE = Path("/kaggle/working/e508_raw_teacher_pair/phase.json")
WORKER_STATUS = Path("/kaggle/working/e508_raw_teacher_pair/worker_status.json")
INPUT_IDENTITY_REPORT = Path("/kaggle/working/e508_raw_teacher_pair/input_identity.json")
LOG_DIR = Path("/kaggle/working/e508_raw_teacher_pair/logs")
SPLIT_RELATIVE = Path("research/campaign_2026-09-29/artifacts/splits/fold0.json")
PORTABLE_RELATIVE = Path("research/campaign_2026-10-04/portable_inputs.json")
ASSET_PATHS = (
    "cache/chain_folds.npz",
    "cache/chains.npz",
    "cache/train_targets.npz",
    "cache/valid_targets.npz",
    "research/campaign_2026-09-29/artifacts/splits/fold0.json",
    "research/campaign_2026-10-04/portable_inputs.json",
)


def sha256_path(path: Path) -> str:
    """Hash a bounded artifact in chunks, never allocating its full contents."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, object]) -> None:
    """Atomically persist phase evidence so provider termination leaves a record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def set_phase(phase: str, **details: object) -> None:
    """Update the externally polled stage file and emit a flushed event."""
    now = time.time()
    state = {"packet": PACKET_ID, "phase": phase, "updated_unix": now, **details}
    write_json(PHASE_FILE, state)
    print(json.dumps({"event": "phase", **state}), flush=True)


def find_unique_file(
    root: Path, name: str, *, size: int | None = None, digest: str | None = None
) -> Path:
    """Resolve an attached Kaggle file by content identity, not guessed mount path."""
    matches = [path for path in root.rglob(name) if path.is_file()]
    matches = [path for path in matches if size is None or path.stat().st_size == size]
    if digest is not None:
        matches = [path for path in matches if sha256_path(path) == digest]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one {name} matching size={size} sha256={digest}; found {len(matches)}"
        )
    return matches[0]


def parquet_footer_identity(path: Path) -> dict[str, object]:
    """Hash the exact Parquet metadata footer and its eight-byte trailer."""
    stat = path.stat()
    if stat.st_size < 8:
        raise ValueError(f"parquet file is shorter than its trailer: {path}")
    with path.open("rb") as stream:
        stream.seek(-8, os.SEEK_END)
        trailer = stream.read(8)
        footer_bytes = int.from_bytes(trailer[:4], "little")
        if trailer[4:] != b"PAR1" or footer_bytes > stat.st_size - 8:
            raise ValueError(
                f"invalid parquet trailer for {path}: "
                f"footer_bytes={footer_bytes}, trailer_magic={trailer[4:].hex()}"
            )
        range_bytes = footer_bytes + 8
        stream.seek(-range_bytes, os.SEEK_END)
        digest = hashlib.sha256()
        remaining = range_bytes
        while remaining:
            block = stream.read(min(remaining, 1 << 20))
            if not block:
                raise ValueError(f"truncated parquet footer for {path}")
            digest.update(block)
            remaining -= len(block)
    return {
        "path": str(path.resolve()),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "footer_bytes": footer_bytes,
        "footer_range_bytes_including_trailer": range_bytes,
        "footer_range_start_offset": stat.st_size - range_bytes,
        "trailer_magic": trailer[4:].decode("ascii"),
        "footer_sha256": digest.hexdigest(),
    }


def audit_checkpoint_manifest_entry(
    source_manifest: dict[str, object],
) -> dict[str, object]:
    """Require the pinned source manifest to bind the audit-only checkpoint."""
    files = source_manifest.get("files")
    expected = (
        files.get(AUDIT_CHECKPOINT_RELATIVE)
        if isinstance(files, dict)
        else None
    )
    if (
        not isinstance(expected, dict)
        or expected.get("sha256") != AUDIT_CHECKPOINT_SHA256
        or expected.get("size_bytes") != AUDIT_CHECKPOINT_BYTES
        or expected.get("source_path") != AUDIT_CHECKPOINT_RELATIVE
    ):
        raise ValueError("audit-only E463 checkpoint is not bound by its source manifest")
    return expected


def asset_file(
    root: Path, manifest: dict[str, object], relative: str
) -> tuple[Path, dict[str, object]]:
    """Find a hashed fold asset by its manifest logical path and exact stored size."""
    entries = [entry for entry in manifest["files"] if entry["path"] == relative]
    if len(entries) != 1:
        raise ValueError(f"assets manifest must contain one {relative}")
    entry = entries[0]
    path = find_unique_file(root, str(entry["filename"]), size=int(entry["size"]))
    return path, entry


def target_index_digest(row_group: np.ndarray, seq_ix: np.ndarray) -> str:
    """Reproduce the split's compact index identity without opening target labels."""
    digest = hashlib.sha256()
    digest.update(np.asarray(row_group, dtype="<i8").tobytes())
    digest.update(np.asarray(seq_ix, dtype="<i8").tobytes())
    return digest.hexdigest()


def exact_group_sets(split: object) -> dict[str, set[int]]:
    """Require exact fold-0 fit/evaluation membership, including linked exclusions."""
    train = set(map(int, split.train_groups))
    valid = set(map(int, split.valid_groups))
    evaluation = set(map(int, split.evaluation_groups))
    if (len(train), len(valid), len(evaluation)) != (
        FIT_COUNTS["train"],
        FIT_COUNTS["valid"],
        EVAL_VALID_COUNT,
    ):
        raise ValueError("fold0 split membership counts changed")
    # Row-group identifiers are scoped to their source Parquet file. Group 528
    # is corrupt only in train.parquet; valid.parquet has its own group 528.
    if CORRUPT_TRAIN_GROUP in train:
        raise ValueError("corrupt train row group entered fit membership")
    return {"train": train, "valid": valid, "evaluation": evaluation}


def training_argv(name: str, seed: int, *, profile: bool = False) -> list[str]:
    """Freeze the raw-only baseline command and make disabled losses explicit."""
    if seed not in SEEDS:
        raise ValueError("only paired seeds 3 and 4 are registered")
    args = [
        "--name",
        name,
        "--device",
        "cuda:0",
        "--arch",
        "gru",
        "--cell",
        "gru",
        "--hidden",
        "128",
        "--layers",
        "2",
        "--dropout",
        "0",
        "--seed",
        str(seed),
        "--raw-only",
        "--loss",
        "wmse",
        "--loss-norm",
        "chunk",
        "--train-source",
        "train+valid",
        "--train-seqs",
        "10607",
        "--split-manifest",
        SPLIT_RELATIVE.as_posix(),
        "--valid-fold",
        "0",
        "--epochs",
        str(EPOCHS),
        "--checkpoint-policy",
        "fixed",
        "--checkpoint-epoch",
        str(CHECKPOINT_EPOCH),
        "--no-eval-during-fit",
        "--batch",
        str(BATCH),
        "--chunk",
        str(CHUNK),
        "--lr",
        "0.001",
        "--wd",
        "0.01",
        "--optim",
        "adamw",
        "--sched",
        "onecycle",
        "--precision",
        "fp16",
        "--init",
        "",
        "--distill",
        "",
        "--distill-boot-h",
        "0",
        "--propensity",
        "",
        "--prop-input",
        "",
        "--prop-power",
        "1",
        "--prop-eps",
        "0",
        "--valid-true-mask",
        "-1",
        "--ema-teacher",
        "0",
        "--aux-struct",
        "0",
        "--aux-next",
        "0",
        "--aux-tier",
        "0",
        "--aux-mask",
        "0",
        "--aux-sign",
        "0",
        "--aux-quote",
        "0",
        "--aux",
        "",
        "--aux-weight",
        "0",
        "--swa-frac",
        "0",
        "--fast",
    ]
    if profile:
        args.extend(("--profile-warmup", "1", "--profile-steps", "3"))
    return args


def profile_estimate_seconds(step_wall_ms: float) -> float:
    """Conservatively estimate two fits and two full prediction exports."""
    if not math.isfinite(step_wall_ms) or step_wall_ms <= 0:
        raise ValueError("profile must report a finite positive step_wall")
    batches_per_epoch = (FIT_COUNTS["train"] + FIT_COUNTS["valid"]) // BATCH
    fit_batches = len(SEEDS) * batches_per_epoch * EPOCHS
    export_batches = len(SEEDS) * sum(math.ceil(rows / BATCH) for rows in TARGET_ORDERS.values())
    # Count every export batch at 75% of a measured forward/backward training step.
    seconds = (fit_batches + 0.75 * export_batches) * step_wall_ms / 1000.0
    return seconds + 900.0


def input_map(input_root: Path) -> dict[str, object]:
    """Audit attached assets and raw inputs, then create only symlinked data paths."""
    if not input_root.is_dir():
        raise FileNotFoundError("/kaggle/input is absent")
    assets_manifest_path = find_unique_file(
        input_root, "assets-manifest.json", size=ASSETS_BYTES, digest=ASSETS_SHA256
    )
    assets = json.loads(assets_manifest_path.read_text())
    if assets.get("dataset") != "vladkochetov07/wnn-c20261004-fold0-assets":
        raise ValueError("fold0 assets manifest names a different private dataset")
    source_manifest_path = find_unique_file(
        input_root, "source-manifest.json", digest=SOURCE_MANIFEST_SHA256
    )
    source_manifest = json.loads(source_manifest_path.read_text())
    checkpoint_path = find_unique_file(
        input_root,
        "model_ep3.pt",
        size=AUDIT_CHECKPOINT_BYTES,
        digest=AUDIT_CHECKPOINT_SHA256,
    )
    audit_checkpoint_manifest_entry(source_manifest)

    required_assets: dict[str, tuple[Path, dict[str, object]]] = {
        relative: asset_file(input_root, assets, relative) for relative in ASSET_PATHS
    }
    portable_path, _ = required_assets[PORTABLE_RELATIVE.as_posix()]
    if sha256_path(portable_path) != PORTABLE_SHA256:
        raise ValueError("attached portable input registry changed")
    split_path, _ = required_assets[SPLIT_RELATIVE.as_posix()]
    if sha256_path(split_path) != SPLIT_SHA256:
        raise ValueError("attached fold0 manifest changed")
    for relative in ("cache/chains.npz", "cache/chain_folds.npz"):
        path, entry = required_assets[relative]
        if sha256_path(path) != entry["sha256"]:
            raise ValueError(f"small chain asset hash changed: {relative}")

    raw_files = {
        "datasets/train.parquet": find_unique_file(input_root, "train.parquet"),
        "datasets/valid.parquet": find_unique_file(input_root, "valid.parquet"),
    }
    portable = json.loads(portable_path.read_text())
    parquet_identity: dict[str, dict[str, object]] = {}
    for logical_name, source in raw_files.items():
        registry_entry = portable["sources"][logical_name]
        identity = registry_entry["original_identity"]
        mount_relative = source.resolve().relative_to(input_root.resolve())
        mount_root = input_root / mount_relative.parts[0]
        expected = {
            "dataset_ref": RAW_DATASET_REF,
            "dataset_version_at_preregistration": RAW_DATASET_VERSION_AT_PREREGISTRATION,
            "dataset_version_observed_at_utc": RAW_DATASET_VERSION_OBSERVED_AT_UTC,
            "logical_path": logical_name,
            "bytes": int(identity["bytes"]),
            "footer_bytes": int(identity["footer_bytes"]),
            "footer_sha256": str(identity["footer_sha256"]),
            "sha256": str(registry_entry["sha256"]),
        }
        observed: dict[str, object] = {
            "dataset_ref": RAW_DATASET_REF,
            "dataset_version_from_runtime_mount": None,
            "mount_path": str(mount_root.resolve()),
            "relative_path_under_kaggle_input": mount_relative.as_posix(),
            "dataset_version_note": (
                "Kaggle does not expose the attached dataset version in the mounted "
                "filesystem; version 1 was current at preregistration and exact file "
                "hashes bind mounted bytes to that frozen source registry"
            ),
        }
        hash_started_unix: float | None = None
        try:
            observed.update(parquet_footer_identity(source))
            hash_started_unix = time.time()
            parquet_identity[logical_name] = {
                "expected": expected,
                "observed": observed,
                "status": "full_file_sha256_in_progress",
                "sha256_started_unix": hash_started_unix,
            }
            write_json(
                INPUT_IDENTITY_REPORT,
                {
                    "packet": PACKET_ID,
                    "dataset_version_at_preregistration": RAW_DATASET_VERSION_AT_PREREGISTRATION,
                    "dataset_version_observed_at_utc": RAW_DATASET_VERSION_OBSERVED_AT_UTC,
                    "status": "full_file_sha256_in_progress",
                    "raw_parquet": parquet_identity,
                },
            )
            print(
                json.dumps(
                    {
                        "event": "raw_source_sha256",
                        "logical_path": logical_name,
                        "stage": "started",
                        "source_path": str(source.resolve()),
                        "expected_sha256": expected["sha256"],
                        "started_unix": hash_started_unix,
                    }
                ),
                flush=True,
            )
            observed["sha256"] = sha256_path(source)
            hash_finished_unix = time.time()
            observed["sha256_started_unix"] = hash_started_unix
            observed["sha256_finished_unix"] = hash_finished_unix
            observed["sha256_elapsed_seconds"] = hash_finished_unix - hash_started_unix
            print(
                json.dumps(
                    {
                        "event": "raw_source_sha256",
                        "logical_path": logical_name,
                        "stage": "completed",
                        "elapsed_seconds": hash_finished_unix - hash_started_unix,
                        "actual_sha256": observed["sha256"],
                    }
                ),
                flush=True,
            )
        except Exception as error:
            report = {
                "status": "inspection_failed",
                "logical_path": logical_name,
                "expected": expected,
                "observed": {**observed, "inspection_error": repr(error)},
                "sha256_started_unix": hash_started_unix,
                "already_inspected": parquet_identity,
            }
            write_json(INPUT_IDENTITY_REPORT, report)
            raise
        footer_matches = all(
            observed.get(key) == expected[key]
            for key in ("bytes", "footer_bytes", "footer_sha256")
        )
        full_file_matches = observed["sha256"] == expected["sha256"]
        entry = {
            "expected": expected,
            "observed": observed,
            "footer_identity_matches": footer_matches,
            "full_file_sha256_matches": full_file_matches,
        }
        parquet_identity[logical_name] = entry
        write_json(
            INPUT_IDENTITY_REPORT,
            {
                "packet": PACKET_ID,
                "dataset_version_at_preregistration": RAW_DATASET_VERSION_AT_PREREGISTRATION,
                "dataset_version_observed_at_utc": RAW_DATASET_VERSION_OBSERVED_AT_UTC,
                "status": "matched" if footer_matches and full_file_matches else "mismatch",
                "raw_parquet": parquet_identity,
            },
        )
        if not footer_matches or not full_file_matches:
            raise ValueError(
                f"raw parquet source identity changed: {logical_name}; "
                f"footer_matches={footer_matches}, full_file_sha256_matches={full_file_matches}"
            )

    for relative, (source, _) in required_assets.items():
        destination = ROOT / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"refusing to replace existing staged input: {destination}")
        destination.symlink_to(source)
    for relative, source in raw_files.items():
        destination = ROOT / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(source)

    split_manifest = json.loads(split_path.read_text())
    portable_required = False
    for logical_name, identity in split_manifest["sources"].items():
        source = ROOT / logical_name
        stat = source.stat()
        if stat.st_size != identity["bytes"] or stat.st_mtime_ns != identity["mtime_ns"]:
            portable_required = True
    return {
        "assets_manifest": {"path": str(assets_manifest_path), "sha256": ASSETS_SHA256},
        "source_manifest": {"path": str(source_manifest_path), "sha256": SOURCE_MANIFEST_SHA256},
        "audit_only_checkpoint": {
            "path": str(checkpoint_path),
            "sha256": AUDIT_CHECKPOINT_SHA256,
            "used_for_fit": False,
        },
        "fold0_manifest": {"path": str(split_path), "sha256": SPLIT_SHA256},
        "portable_registry": {"path": str(portable_path), "sha256": PORTABLE_SHA256},
        "asset_files": {
            relative: {
                "path": str(path),
                "sha256": entry["sha256"],
                "size": entry["size"],
            }
            for relative, (path, entry) in required_assets.items()
        },
        "raw_files": parquet_identity,
        "portable_registry_required_by_mtime": portable_required,
        "staged_paths_are_symlinks": True,
    }


def validate_target_indices(split: object) -> dict[str, list[int]]:
    """Check cache row orders without loading cached target values."""
    expected = exact_group_sets(split)
    orders: dict[str, list[int]] = {}
    for name, count in TARGET_ORDERS.items():
        with np.load(ROOT / "cache" / f"{name}_targets.npz", allow_pickle=False) as values:
            groups = np.asarray(values["row_group"], dtype=np.int64)
            seq_ix = np.asarray(values["seq_ix"], dtype=np.int64)
        if groups.shape != (count,) or len(np.unique(groups)) != count:
            raise ValueError(f"{name} target row order changed")
        digest = target_index_digest(groups, seq_ix)
        portable = json.loads((ROOT / PORTABLE_RELATIVE).read_text())
        source_identity = portable["sources"][f"cache/{name}_targets.npz"]["original_identity"]
        if digest != source_identity["index_sha256"]:
            raise ValueError(f"{name} target index checksum changed")
        if name == "train" and CORRUPT_TRAIN_GROUP in set(map(int, groups)):
            raise ValueError("corrupt row group appears in train target order")
        if not expected[name].issubset(set(map(int, groups))):
            raise ValueError(f"{name} target order omits eligible fitting groups")
        orders[name] = list(map(int, groups))
    return orders


def hardware_snapshot() -> dict[str, object]:
    """Capture actual Kaggle CPU, RAM and GPU facts before any worker allocation."""
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    meminfo: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, _, value = line.partition(":")
        fields = value.strip().split()
        if fields and fields[0].isdigit():
            meminfo[key] = int(fields[0]) * (1024 if len(fields) > 1 and fields[1] == "kB" else 1)
    devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            devices.append({"index": index, "name": props.name, "total_memory": props.total_memory})
    return {
        "cpu_affinity_count": len(affinity),
        "cpu_affinity": affinity,
        "meminfo_bytes": meminfo,
        "cuda_available": torch.cuda.is_available(),
        "cuda_devices": devices,
        "nvidia_smi": subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
        ).stdout.strip(),
    }


def gate_hardware(snapshot: dict[str, object]) -> None:
    """Require one usable GPU and enough host headroom for a serial worker."""
    if not snapshot["cuda_available"] or not snapshot["cuda_devices"]:
        raise RuntimeError("actual Kaggle GPU inventory is empty")
    if snapshot["cpu_affinity_count"] < 2:
        raise RuntimeError("fewer than two host CPUs are available")
    mem = snapshot["meminfo_bytes"]
    if int(mem.get("MemAvailable", 0)) < 4 * 1024**3:
        raise RuntimeError("less than 4 GiB host memory is available for the profile gate")
    if int(snapshot["cuda_devices"][0]["total_memory"]) < 6 * 1024**3:
        raise RuntimeError("selected GPU has less than 6 GiB total memory")
    if not re.search(r"Tesla T4", str(snapshot["cuda_devices"][0]["name"])):
        raise RuntimeError("actual device is not a Tesla T4 as declared in kernel metadata")


def set_worker_environment() -> None:
    """Use a single bounded loader and the contract's 4.3 GB allocator admission."""
    os.environ.update(
        {
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "WNN_THREADS": "2",
            "WNN_LOADER_WORKERS": "0",
            "WNN_LOADER_RAM_GB": "2",
            "WNN_GPU_MEMORY_BYTES": str(GPU_ALLOCATOR_BYTES),
            "OMP_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "NUMEXPR_NUM_THREADS": "2",
        }
    )


class FeatureOnlySequences(Dataset[torch.Tensor]):
    """Read model inputs from raw parquet without requesting any label or mask column."""

    def __init__(self, split: str, groups: list[int]) -> None:
        from wnn.data import SequenceFile

        self.path = ROOT / "datasets" / f"{split}.parquet"
        self.groups = groups
        self.reader = SequenceFile(self.path)

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index: int) -> torch.Tensor:
        sequence = self.reader.read(self.groups[index], targets=False, mask=False)
        return torch.from_numpy(sequence.features)


def export_predictions(
    run: str, seed: int, split_manifest_sha256: str, orders: dict[str, list[int]]
) -> list[dict[str, object]]:
    """Export full cache-order float32 predictions from feature-only parquet reads."""
    import train_rnn

    from wnn.splits import file_sha256, load_campaign_split

    run_root = ROOT / "artifacts" / "runs" / run
    provenance_path = run_root / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    checkpoint_path = run_root / f"model_ep{CHECKPOINT_EPOCH}.pt"
    checkpoint_sha = file_sha256(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = checkpoint["args"]
    expected_fit = exact_group_sets(load_campaign_split(ROOT / SPLIT_RELATIVE, ROOT))
    actual_train = set(map(int, provenance["fit_row_groups"]["train"]))
    actual_valid = set(map(int, provenance["fit_row_groups"]["valid"]))
    if actual_train != expected_fit["train"] or actual_valid != expected_fit["valid"]:
        raise ValueError(f"{run} provenance does not match exact fold0 fit groups")
    if checkpoint["epoch"] != CHECKPOINT_EPOCH or args["checkpoint_epoch"] != CHECKPOINT_EPOCH:
        raise ValueError(f"{run} did not select fixed zero-based checkpoint epoch 3")
    if (
        args.get("seed") != seed
        or not args.get("raw_only")
        or args.get("init")
        or args.get("distill")
    ):
        raise ValueError(f"{run} checkpoint differs from the frozen raw-only scratch recipe")
    if args.get("split_manifest_sha256") != split_manifest_sha256:
        raise ValueError(f"{run} checkpoint uses another split manifest")
    if (
        args.get("loss") != "wmse"
        or args.get("optim") != "adamw"
        or args.get("sched") != "onecycle"
    ):
        raise ValueError(f"{run} checkpoint has unexpected loss or optimizer")
    checkpoint_provenance = checkpoint.get("provenance", {})
    if checkpoint_provenance.get("fit_row_groups") != provenance.get("fit_row_groups"):
        raise ValueError(f"{run} checkpoint and run fit memberships disagree")

    train_rnn.RAW_ONLY = bool(args.get("raw_only"))
    train_rnn.ONLINE_NORM = bool(args.get("online_norm"))
    train_rnn.HEURISTICS = bool(args.get("heuristics"))
    train_rnn.RESEARCH_FEATURES = args.get("research_features", "none")
    train_rnn.PRIVILEGED_MODE = args.get("privileged_mode", "none")
    train_rnn.PRIVILEGED_HORIZON = args.get("privileged_horizon", 8)
    train_rnn.SENTINEL_PREPROCESSOR = None
    train_rnn.INPUT_PROPENSITY = None
    train_rnn.READOUT = None
    train_rnn.READOUT_TILTED = False
    model_state = checkpoint["model"]
    n_out = int(model_state["head.2.weight"].shape[0])
    if n_out != 2:
        raise ValueError("serving head must expose exactly two outputs")
    model = train_rnn.make_model(args, int(checkpoint["n_in"]), n_out)
    model.load_state_dict(model_state)
    model = model.to(torch.device("cuda:0"))
    model.eval()

    records: list[dict[str, object]] = []
    for split_name in ("train", "valid"):
        groups = orders[split_name]
        final_path = ROOT / "cache" / f"pred_{run}_{split_name}.npy"
        partial_path = final_path.with_name(final_path.name + ".partial")
        if final_path.exists() or partial_path.exists():
            raise FileExistsError(f"refusing to overwrite prediction export: {final_path}")
        output = np.lib.format.open_memmap(
            partial_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(groups), 20_000, 2),
        )
        loader = DataLoader(
            FeatureOnlySequences(split_name, groups),
            batch_size=BATCH,
            shuffle=False,
            num_workers=0,
        )
        offset = 0
        for features in loader:
            prediction = train_rnn.predict(model, features, 4_000, torch.device("cuda:0"))
            if prediction.shape[-1] != 2 or not np.isfinite(prediction).all():
                raise ValueError(f"{run} produced invalid {split_name} predictions")
            output[offset : offset + len(prediction)] = prediction
            offset += len(prediction)
        output.flush()
        del output
        if offset != len(groups):
            raise RuntimeError(f"{run} wrote {offset} of {len(groups)} {split_name} groups")
        partial_path.replace(final_path)
        metadata = {
            "run": run,
            "split": split_name,
            "epoch": CHECKPOINT_EPOCH,
            "checkpoint_sha256": checkpoint_sha,
            "prediction_sha256": file_sha256(final_path),
            "split_manifest_sha256": split_manifest_sha256,
            "row_groups": groups,
            "row_groups_sha256": hashlib.sha256(
                np.asarray(groups, dtype="<i8").tobytes()
            ).hexdigest(),
            "feature_only_export": True,
            "label_columns_read": False,
            "mask_columns_read": False,
            "seed": seed,
            "serving_head_width": n_out,
        }
        sidecar_path = final_path.with_suffix(".json")
        write_json(sidecar_path, metadata)
        records.append(
            {
                "prediction_path": str(final_path),
                "prediction_sha256": metadata["prediction_sha256"],
                "prediction_bytes": final_path.stat().st_size,
                "sidecar_path": str(sidecar_path),
                "sidecar_sha256": file_sha256(sidecar_path),
                "row_count": len(groups),
            }
        )
    del model
    torch.cuda.empty_cache()
    return records


class Tee:
    """Write child trainer output to a durable phase log and the live kernel stream."""

    def __init__(self, file: TextIO, console: TextIO) -> None:
        self.file = file
        self.console = console

    def write(self, text: str) -> int:
        self.file.write(text)
        self.file.flush()
        self.console.write(text)
        self.console.flush()
        return len(text)

    def flush(self) -> None:
        self.file.flush()
        self.console.flush()


def run_trainer(argv: list[str], log_path: Path) -> str:
    """Call the pinned trainer in-process so immutable-input digests can be reused."""
    import train_rnn

    previous_argv = sys.argv
    old_directory = Path.cwd()
    sys.argv = [str(ROOT / "scripts" / "train_rnn.py"), *argv]
    try:
        os.chdir(ROOT)
        with log_path.open("x", encoding="utf-8") as log:
            tee = Tee(log, sys.stdout)
            with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
                train_rnn.main()
        return log_path.read_text(encoding="utf-8")
    finally:
        sys.argv = previous_argv
        os.chdir(old_directory)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def heartbeat_loop(stop: threading.Event, state: dict[str, object]) -> None:
    """Persist lightweight host heartbeats while native training code is running."""
    try:
        import resource

        peak_rss = 0
        while not stop.wait(30):
            usage = resource.getrusage(resource.RUSAGE_SELF)
            # Linux reports ru_maxrss in KiB.
            peak_rss = max(peak_rss, int(usage.ru_maxrss) * 1024)
            write_json(
                WORKER_STATUS,
                {
                    "packet": PACKET_ID,
                    "phase": state.get("phase", "unknown"),
                    "heartbeat_unix": time.time(),
                    "peak_rss_bytes": peak_rss,
                    "pid": os.getpid(),
                },
            )
            print(
                json.dumps(
                    {
                        "event": "heartbeat",
                        "phase": state.get("phase"),
                        "peak_rss_bytes": peak_rss,
                    }
                ),
                flush=True,
            )
    except OSError as error:
        print(json.dumps({"event": "heartbeat_error", "error": str(error)}), flush=True)


def call_trainer_logged(
    state: dict[str, object], phase: str, argv: list[str], log_path: Path
) -> str:
    """Mark a trainer stage durably before entering Torch's native kernels."""
    state["phase"] = phase
    state["phase_started_unix"] = time.time()
    write_json(PHASE_FILE, {"packet": PACKET_ID, **state})
    print(json.dumps({"event": "trainer_start", "phase": phase, "argv": argv}), flush=True)
    text = run_trainer(argv, log_path)
    print(json.dumps({"event": "trainer_complete", "phase": phase}), flush=True)
    return text


def parse_profile(log_text: str) -> dict[str, float]:
    """Require a finite multi-step trainer profile before any full fit begins."""
    matches = re.findall(r"^PROFILE\s+(\{.*\})$", log_text, flags=re.MULTILINE)
    if len(matches) != 1:
        raise ValueError("profile log does not contain exactly one PROFILE record")
    profile = json.loads(matches[0])
    step_wall_ms = float(profile["step_wall"])
    if not math.isfinite(step_wall_ms) or step_wall_ms <= 0:
        raise ValueError("profile step_wall is not finite")
    return {str(key): float(value) for key, value in profile.items()}


def verify_run(run: str, seed: int, groups: dict[str, set[int]]) -> dict[str, object]:
    """Verify immutable training provenance against exact fold member identities."""
    run_root = ROOT / "artifacts" / "runs" / run
    provenance = json.loads((run_root / "provenance.json").read_text())
    actual_train = set(map(int, provenance["fit_row_groups"]["train"]))
    actual_valid = set(map(int, provenance["fit_row_groups"]["valid"]))
    if actual_train != groups["train"] or actual_valid != groups["valid"]:
        raise ValueError(f"{run} recorded fit rows differ from the frozen exact sets")
    if provenance.get("split_manifest_sha256") != SPLIT_SHA256:
        raise ValueError(f"{run} split manifest hash differs")
    args = provenance.get("args", {})
    frozen = {
        "seed": seed,
        "hidden": 128,
        "layers": 2,
        "arch": "gru",
        "cell": "gru",
        "dropout": 0.0,
        "raw_only": True,
        "loss": "wmse",
        "loss_norm": "chunk",
        "optim": "adamw",
        "sched": "onecycle",
        "lr": 0.001,
        "wd": 0.01,
        "precision": "fp16",
        "epochs": EPOCHS,
        "checkpoint_policy": "fixed",
        "checkpoint_epoch": CHECKPOINT_EPOCH,
        "no_eval_during_fit": True,
        "train_source": "train+valid",
        "train_seqs": 10_607,
        "valid_fold": 0,
        "valid_train_fraction": 1.0,
        "batch": BATCH,
        "chunk": CHUNK,
        "fast": True,
        "init": "",
        "distill": "",
        "distill_boot_h": 0,
        "propensity": "",
        "prop_input": "",
        "prop_power": 1.0,
        "prop_eps": 0.0,
        "valid_true_mask": -1.0,
        "mask_weight": False,
        "ema_teacher": 0.0,
        "aux_struct": 0.0,
        "aux_next": 0.0,
        "aux_tier": 0.0,
        "aux_mask": 0.0,
        "aux_sign": 0.0,
        "aux_quote": 0.0,
        "aux": "",
        "aux_weight": 0.0,
        "swa_frac": 0.0,
        "class_balance": False,
    }
    for key, expected in frozen.items():
        if args.get(key) != expected:
            raise ValueError(f"{run} changed frozen argument {key}: {args.get(key)!r}")
    checkpoint = run_root / f"model_ep{CHECKPOINT_EPOCH}.pt"
    checkpoint_sha = sha256_path(checkpoint)
    if provenance.get("checkpoint_sha256", {}).get(checkpoint.name) != checkpoint_sha:
        raise ValueError(f"{run} checkpoint hash differs from its trainer provenance")
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model_state = checkpoint_payload.get("model", {})
    input_width = int(checkpoint_payload.get("n_in", 0))
    if (
        checkpoint_payload.get("epoch") != CHECKPOINT_EPOCH
        or input_width <= 0
        or model_state.get("gru.weight_ih_l0", torch.empty(0)).shape[0] != 3 * 128
        or model_state.get("gru.weight_ih_l1", torch.empty(0)).shape[0] != 3 * 128
        or model_state.get("head.2.weight", torch.empty(0)).shape != (2, 128)
        or model_state.get("inp.weight", torch.empty(0)).shape != (128, input_width)
    ):
        raise ValueError(f"{run} checkpoint tensors differ from the frozen GRU128x2 serving model")
    if set(map(int, provenance.get("evaluation_row_groups", []))) != groups["evaluation"]:
        raise ValueError(f"{run} evaluation exclusion metadata differs from fold0")
    return {
        "run": run,
        "seed": seed,
        "fit_counts": {"train": len(actual_train), "valid": len(actual_valid)},
        "fit_row_group_sha256": {
            "train": hashlib.sha256(
                np.asarray(sorted(actual_train), dtype="<i8").tobytes()
            ).hexdigest(),
            "valid": hashlib.sha256(
                np.asarray(sorted(actual_valid), dtype="<i8").tobytes()
            ).hexdigest(),
        },
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "epoch": CHECKPOINT_EPOCH,
        "model_head_width": 2,
        "provenance_sha256": sha256_path(run_root / "provenance.json"),
    }


def main() -> None:
    """Run input preflight, profile gate, paired fits and feature-only exports."""
    LOG_DIR.mkdir(parents=True, exist_ok=False)
    start_time = time.time()
    state: dict[str, object] = {"phase": "startup", "status": "active", "start_unix": start_time}

    def terminate_worker(signum: int, frame: object) -> None:
        """Save a termination marker before unwinding the worker's finally block."""
        state.update({"phase": "termination_requested", "signal": signum})
        write_json(PHASE_FILE, {"packet": PACKET_ID, **state, "updated_unix": time.time()})
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate_worker)
    heartbeat_stop = threading.Event()
    heartbeat = threading.Thread(target=heartbeat_loop, args=(heartbeat_stop, state), daemon=True)
    heartbeat.start()
    result: dict[str, object] = {"packet": PACKET_ID, "status": "active", "start_unix": start_time}

    def mark_phase(phase: str, **details: object) -> None:
        state.update({"phase": phase, "updated_unix": time.time(), **details})
        set_phase(phase, **details)

    try:
        mark_phase("input_inventory")
        discovered = input_map(Path("/kaggle/input"))
        print(json.dumps({"event": "input_map", "value": discovered}), flush=True)
        set_worker_environment()
        if discovered["portable_registry_required_by_mtime"]:
            os.environ["WNN_PORTABLE_DATA_MANIFEST"] = str(ROOT / PORTABLE_RELATIVE)
        else:
            os.environ.pop("WNN_PORTABLE_DATA_MANIFEST", None)

        mark_phase("split_and_target_preflight")
        from wnn.splits import load_campaign_split

        split = load_campaign_split(ROOT / SPLIT_RELATIVE, ROOT)
        groups = exact_group_sets(split)
        orders = validate_target_indices(split)
        if set(orders["train"]) != set(range(10_607)) - {CORRUPT_TRAIN_GROUP}:
            raise ValueError("train target index is not the full corrupt-row-excluded order")
        if set(orders["valid"]) != set(range(1_873)):
            raise ValueError("valid target index is not the full cache row order")
        result["input_map"] = discovered
        result["split_sha256"] = split.sha256
        result["fit_group_counts"] = {name: len(groups[name]) for name in ("train", "valid")}
        result["evaluation_valid_groups"] = len(groups["evaluation"])
        result["target_row_counts"] = {name: len(value) for name, value in orders.items()}
        result["exact_fit_sets_validated"] = True

        mark_phase("hardware_gate")
        snapshot = hardware_snapshot()
        gate_hardware(snapshot)
        result["hardware"] = snapshot
        device_free, _ = torch.cuda.mem_get_info(0)
        if device_free < 6 * 1024**3:
            raise RuntimeError("less than 6 GiB is free on selected GPU before profile")
        disk = shutil.disk_usage(Path("/kaggle/working"))
        planned_output_bytes = (
            sum(len(orders[name]) for name in orders) * 20_000 * 2 * 4 * len(SEEDS)
        )
        if disk.free - planned_output_bytes < OUTPUT_MIN_FREE_BYTES:
            raise OSError("insufficient Kaggle working-disk space for both full prediction pairs")
        planned_runs = ["e508_raw_profile_s3_f0"] + [f"e508_raw_s{seed}_f0" for seed in SEEDS]
        for run in planned_runs:
            if (ROOT / "artifacts" / "runs" / run).exists():
                raise FileExistsError(f"refusing to reuse existing training run: {run}")
        for run in planned_runs[1:]:
            for split_name in ("train", "valid"):
                if (ROOT / "cache" / f"pred_{run}_{split_name}.npy").exists():
                    raise FileExistsError(
                        f"refusing to replace existing prediction cache for {run}"
                    )
        result["disk_preflight"] = {
            "free_bytes": disk.free,
            "planned_prediction_bytes": planned_output_bytes,
            "minimum_free_after_exports": OUTPUT_MIN_FREE_BYTES,
            "staged_inputs_bytes_copied": 0,
        }
        mark_phase("profile_s3", input_map_verified=True, hardware=snapshot)
        torch.cuda.reset_peak_memory_stats(0)
        profile_name = "e508_raw_profile_s3_f0"
        profile_text = call_trainer_logged(
            state,
            "profile_s3",
            training_argv(profile_name, 3, profile=True),
            LOG_DIR / "profile_s3.log",
        )
        profile = parse_profile(profile_text)
        peak_allocated = torch.cuda.max_memory_allocated(0)
        peak_reserved = torch.cuda.max_memory_reserved(0)
        if peak_reserved > GPU_ALLOCATOR_BYTES:
            raise RuntimeError("profile exceeded the frozen 4.3 GB Torch allocator admission")
        estimated_seconds = profile_estimate_seconds(profile["step_wall"])
        if estimated_seconds > MAX_PRODUCTION_WALL_SECONDS:
            raise TimeoutError(
                f"profile projects {estimated_seconds:.0f}s, beyond the 6,300s production cap"
            )
        result["profile"] = {
            "metrics": profile,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "estimated_pair_fit_and_export_seconds": estimated_seconds,
            "profile_log_sha256": sha256_path(LOG_DIR / "profile_s3.log"),
        }
        mark_phase("profile_gate_passed", profile=result["profile"])

        if PROFILE_ONLY:
            result["status"] = "profile_ready_awaiting_controller_review"
            result["fit_admitted"] = False
            result["fit_stages_started"] = []
            result["profile_review_required"] = [
                "actual device and host headroom",
                "finite profile timing and estimated pair/export duration",
                "Torch allocated/reserved memory peak",
                "exact fold0 membership and target-cache row order",
                "attached manifests and raw Parquet footer identities",
            ]
            result["finish_unix"] = time.time()
            result["wall_seconds"] = round(result["finish_unix"] - start_time, 2)
            profile_result = ROOT / "e508_raw_teacher_pair_profile_result.json"
            write_json(profile_result, result)
            mark_phase("profile_ready_awaiting_controller_review", result_path=str(profile_result))
            return

        run_records: list[dict[str, object]] = []
        for seed in SEEDS:
            run = f"e508_raw_s{seed}_f0"
            fit_phase = f"fit_seed_{seed}"
            mark_phase(fit_phase, run=run)
            fit_started = time.time()
            call_trainer_logged(
                state,
                fit_phase,
                training_argv(run, seed),
                LOG_DIR / f"{fit_phase}.log",
            )
            fit_record = verify_run(run, seed, groups)
            fit_record["fit_wall_seconds"] = round(time.time() - fit_started, 2)
            export_phase = f"export_seed_{seed}"
            mark_phase(export_phase, run=run)
            fit_record["predictions"] = export_predictions(run, seed, split.sha256, orders)
            run_records.append(fit_record)
            result["runs"] = run_records
            write_json(
                WORKER_STATUS, {"packet": PACKET_ID, "phase": export_phase, "runs": run_records}
            )

        result["runs"] = run_records
        result["status"] = "complete"
        result["finish_unix"] = time.time()
        result["wall_seconds"] = round(result["finish_unix"] - start_time, 2)
        write_json(ROOT / "e508_raw_teacher_pair_result.json", result)
        mark_phase("complete", result_path=str(ROOT / "e508_raw_teacher_pair_result.json"))
    except BaseException as error:
        result.update(
            {
                "status": "failed",
                "finish_unix": time.time(),
                "error": repr(error),
                "traceback": traceback.format_exc(),
            }
        )
        write_json(ROOT / "e508_raw_teacher_pair_result.json", result)
        mark_phase("failed", error=repr(error))
        raise
    finally:
        heartbeat_stop.set()
        heartbeat.join(timeout=2)
        result["final_diagnostics"] = {
            "pid": os.getpid(),
            "python": sys.version,
            "cuda_devices": torch.cuda.device_count(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(0)
            if torch.cuda.is_available()
            else 0,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(0)
            if torch.cuda.is_available()
            else 0,
        }
        result["finish_unix"] = time.time()
        write_json(ROOT / "e508_raw_teacher_pair_finally.json", result)
        write_json(WORKER_STATUS, {"packet": PACKET_ID, **result})


if __name__ == "__main__":
    main()
