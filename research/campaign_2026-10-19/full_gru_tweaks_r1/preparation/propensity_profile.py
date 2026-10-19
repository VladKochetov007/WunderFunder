"""Run a bounded, actual-shape strict-FP32 propensity engineering profile."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import stat
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import numpy as np
    from propensity_profile_core import ProfileOutcome

PROFILE_SOURCE_COMMIT = "235db75ab91e24c69f21e637a83a48512dc674fe"
PROFILE_GPU_UUID = "GPU-a50d0c3b-d28f-a2a7-8b43-3e02b543fc0c"
PROFILE_GPU_MEMORY_BYTES = 3_000_000_000
PROFILE_BATCH_SIZE = 16
PROFILE_SEQUENCE_LENGTH = 20_000
PROFILE_CHUNK_ROWS = 2_000
PROFILE_REPEATS = 2
PROFILE_SEED = 0


@dataclass(frozen=True)
class ProfilePlan:
    """Deterministic real-row membership and chunk schedule for this profile."""

    first_four_groups: tuple[int, ...]
    batch_groups: tuple[int, ...]
    repeat_batch_groups: tuple[tuple[int, ...], ...]
    chunk_starts: tuple[int, ...]


class ProfileRows(Protocol):
    plan: ProfilePlan
    sequence_ids: tuple[int, ...]
    feature_batches: tuple[np.ndarray, ...]
    masks: tuple[np.ndarray, ...]
    split_sha256: str
    valid_path: Path
    valid_stat: dict[str, int]
    valid_registered_identity: dict[str, object]
    load_seconds: float


class _Journal(Protocol):
    def append(self, attempt_id: str, event: dict[str, object], *, durable: bool = False) -> int:
        """Append a durable profile progress event."""


def build_profile_plan(
    valid_groups: tuple[int, ...], evaluation_groups: tuple[int, ...]
) -> ProfilePlan:
    """Choose the first 16 ordered fit-valid groups and two exact repeats."""
    if len(valid_groups) < PROFILE_BATCH_SIZE:
        raise ValueError(f"profile requires {PROFILE_BATCH_SIZE} fit-valid groups")
    if len(set(valid_groups)) != len(valid_groups):
        raise ValueError("fit-valid groups are not unique")
    groups = tuple(map(int, valid_groups[:PROFILE_BATCH_SIZE]))
    if set(groups) & set(map(int, evaluation_groups)):
        raise ValueError("profile groups overlap outer evaluation groups")
    if PROFILE_SEQUENCE_LENGTH % PROFILE_CHUNK_ROWS:
        raise ValueError("profile sequence length must divide evenly into chunks")
    starts = tuple(range(0, PROFILE_SEQUENCE_LENGTH, PROFILE_CHUNK_ROWS))
    return ProfilePlan(groups[:4], groups, (groups,) * PROFILE_REPEATS, starts)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _phase(journal: _Journal, attempt_id: str, phase: str) -> None:
    journal.append(attempt_id, {"event": "profile_phase", "phase": phase}, durable=True)
    print(json.dumps({"event": "PROFILE_PHASE", "phase": phase}), flush=True)


def _load_rows(
    data_root: Path, split_path: Path, journal: _Journal, attempt_id: str
) -> ProfileRows:
    """Read first fit-valid feature/mask groups after the durable START event."""
    import numpy as np
    from propensity_profile_core import ProfileRowsData

    from wnn.data import N_FEATURES, SequenceFile
    from wnn.splits import load_campaign_split

    started = time.monotonic()
    _phase(journal, attempt_id, "validating_split_and_membership")
    split = load_campaign_split(split_path, data_root)
    plan = build_profile_plan(split.valid_groups, split.evaluation_groups)
    split.assert_fit_groups((), plan.batch_groups)
    manifest = json.loads(split_path.read_bytes())
    relative = "datasets/valid.parquet"
    if relative not in manifest["sources"]:
        raise ValueError("fold manifest does not identify datasets/valid.parquet")
    valid_path = data_root / relative
    before = valid_path.stat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("valid parquet is not a regular file")
    reader = SequenceFile(valid_path)
    if max(plan.batch_groups) >= len(reader):
        raise ValueError("fit-valid group is outside the canonical parquet")
    _phase(journal, attempt_id, "reading_fit_valid_features_and_masks")
    features: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    sequence_ids: list[int] = []
    for group in plan.batch_groups:
        sequence = reader.read(group, targets=False, mask=True)
        if sequence.targets is not None:
            raise RuntimeError("target columns were unexpectedly read")
        if sequence.scored is None or sequence.scored.shape != (PROFILE_SEQUENCE_LENGTH,):
            raise ValueError("fit-valid is_scored mask is missing or has the wrong shape")
        if sequence.features.shape != (PROFILE_SEQUENCE_LENGTH, N_FEATURES):
            raise ValueError("raw feature sequence has the wrong shape")
        if sequence.features.dtype != np.float32 or sequence.scored.dtype != np.bool_:
            raise ValueError("raw features/mask have unexpected dtypes")
        features.append(sequence.features)
        masks.append(sequence.scored)
        sequence_ids.append(sequence.seq_ix)
    after = valid_path.stat()
    before_id = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_id = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_id != after_id:
        raise RuntimeError("valid parquet changed while the profile read fit groups")
    return ProfileRowsData(
        plan=plan,
        sequence_ids=tuple(sequence_ids),
        feature_batches=tuple(features),
        masks=tuple(masks),
        split_sha256=split.sha256,
        valid_path=valid_path,
        valid_stat={
            "device": after.st_dev,
            "inode": after.st_ino,
            "bytes": after.st_size,
            "mtime_ns": after.st_mtime_ns,
        },
        valid_registered_identity=manifest["sources"][relative],
        load_seconds=time.monotonic() - started,
    )


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_create_only(path: Path, payload: dict[str, object]) -> None:
    """Persist the profile result without replacing an earlier attempt."""
    import json
    import os

    raw = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                raise OSError("short write to profile result")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_result(args: argparse.Namespace, outcome: ProfileOutcome, journal: _Journal) -> None:
    """Bind and persist completed profile evidence without replacing prior output."""
    from wnn.propensity import STRICT_FP32_PROFILE, ordered_groups_sha256

    rows = outcome.rows
    profile_path = Path(__file__).resolve()
    core_path = profile_path.with_name("propensity_profile_core.py")
    config_path = profile_path.with_name("PROPENSITY_PREREQUISITE.json")
    result = {
        "schema": "strict_propensity_actual_shape_profile.v1",
        "status": "completed_engineering_profile_only",
        "attempt_id": outcome.attempt_id,
        "source_commit": PROFILE_SOURCE_COMMIT,
        "profile_sha256": _sha256(profile_path),
        "profile_core_sha256": _sha256(core_path),
        "configuration_sha256": _sha256(config_path),
        "propensity_fit_source_sha256": outcome.source_hashes,
        "split_manifest_sha256": rows.split_sha256,
        "valid_source": {
            "path": str(rows.valid_path),
            **rows.valid_stat,
            "registered_identity": rows.valid_registered_identity,
        },
        "membership": {
            "policy": "first_16_ordered_campaign_fit_valid_groups; same B16 repeated twice",
            "first_four_groups": list(rows.plan.first_four_groups),
            "batch_groups": list(rows.plan.batch_groups),
            "batch_groups_sha256": ordered_groups_sha256(rows.plan.batch_groups),
            "sequence_ids": list(rows.sequence_ids),
            "evaluation_rows_read": False,
            "target_columns_read": False,
            "training_label": "is_scored mask only; no t0/t1 values",
        },
        "profile": {
            "model": "PropensityNet(211, hidden=64)",
            "precision_profile": STRICT_FP32_PROFILE,
            "seed": PROFILE_SEED,
            "optimizer": "AdamW lr=0.002 wd=0.01; no gradient clipping",
            "warmup_rows": 99,
            "chunk_rows": PROFILE_CHUNK_ROWS,
            "chunk_starts": list(rows.plan.chunk_starts),
            "repeat_count": PROFILE_REPEATS,
            "repeat_records": outcome.repeat_records,
            "normalization": {
                **outcome.normalization,
                "groups": list(rows.plan.batch_groups),
                "scope": "profile-only; discarded before full-fit normalization",
            },
            "optimizer_progress": outcome.optimizer_progress,
            "checkpoint_written": False,
            "quality_metric_computed": False,
        },
        "runtime": outcome.runtime,
    }
    output = args.output.resolve(strict=True)
    _write_create_only(output / "PROFILE_RESULT.json", result)
    journal.append(
        outcome.attempt_id,
        {"event": "profile_completed", "optimizer_progress": outcome.optimizer_progress},
        durable=True,
    )
    print(
        json.dumps({"event": "PROFILE_COMPLETE", "result": str(output / "PROFILE_RESULT.json")}),
        flush=True,
    )


def main() -> None:
    """Run the admitted profile command through the cold-import-safe entry."""
    args = _parser().parse_args()
    source_root = args.source_root.resolve(strict=True)
    data_root, split_path = args.data_root.resolve(strict=True), args.split.resolve(strict=True)
    output = args.output.resolve(strict=True)
    if not stat.S_ISDIR(output.lstat().st_mode) or output.is_symlink():
        raise ValueError("profile output must be an existing non-symlink directory")
    progress_path = output / "UPDATE_PROGRESS.jsonl"
    if os.path.lexists(progress_path) or os.path.lexists(output / "PROFILE_RESULT.json"):
        raise FileExistsError("profile result/progress path already exists")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != PROFILE_GPU_UUID:
        raise ValueError("profile must be pinned to the admitted A4000 UUID")
    if os.environ.get("WNN_GPU_MEMORY_BYTES") != str(PROFILE_GPU_MEMORY_BYTES):
        raise ValueError("profile must use the admitted 3,000,000,000-byte allocator cap")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("strict profile requires the declared cuBLAS workspace setting")
    if os.environ.get("WNN_PROJECT_ROOT") != str(data_root):
        raise ValueError("profile project root must match the admitted data root")
    if os.environ.get("WNN_LOADER_WORKERS") != "0" or os.environ.get("WNN_THREADS") != "1":
        raise ValueError("profile must use the admitted single-thread, zero-worker settings")
    sys.path.insert(0, str(source_root / "scripts"))
    sys.path.insert(0, str(source_root / "src"))
    from wnn.propensity_progress import ProgressJournal

    attempt_id = str(uuid.uuid4())
    journal = ProgressJournal(progress_path)
    started = time.monotonic()
    started_utc = dt.datetime.now(dt.UTC).isoformat()
    _phase(journal, attempt_id, "started")
    rows = _load_rows(data_root, split_path, journal, attempt_id)
    from propensity_profile_core import run_profile

    outcome = run_profile(args, rows, journal, attempt_id, started_utc, started)
    _write_result(args, outcome, journal)


if __name__ == "__main__":
    main()
