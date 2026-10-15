#!/usr/bin/env python3
"""Write a small, source-qualified, label-free complete-sequence CPU fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow

from wnn.data import N_FEATURES, SEQUENCE_LENGTH, WARMUP, SequenceFile


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_FOLD0_SHA256 = (
    "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d"
)
SLOT_COUNT = 8
SEED = 3
SPLIT_PATH = "research/campaign_2026-09-29/artifacts/splits/fold0.json"
TRAIN_PATH = "datasets/train.parquet"
MAX_SPLIT_BYTES = 2_000_000
HASH_CHUNK_BYTES = 1 << 20


def canonical_sha256(value: Any) -> str:
    """Hash canonical JSON with stable UTF-8 encoding and key ordering."""
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a file in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: memoryview | bytes) -> str:
    """Hash a contiguous in-memory view without making a second array copy."""
    return hashlib.sha256(value).hexdigest()


def parquet_footer_identity(path: Path) -> dict[str, int | str]:
    """Return the same bounded Parquet footer identity used by campaign splits."""
    stat = path.stat()
    if stat.st_size < 12:
        raise ValueError(f"Parquet source is too small: {path}")
    with path.open("rb") as stream:
        stream.seek(-8, os.SEEK_END)
        trailer = stream.read(8)
        footer_bytes = int.from_bytes(trailer[:4], "little")
        if trailer[4:] != b"PAR1" or footer_bytes > stat.st_size - 8:
            raise ValueError(f"invalid Parquet footer: {path}")
        stream.seek(-8 - footer_bytes, os.SEEK_END)
        digest = hashlib.sha256()
        remaining = footer_bytes + 8
        while remaining:
            chunk = stream.read(min(remaining, HASH_CHUNK_BYTES))
            if not chunk:
                raise ValueError(f"truncated Parquet footer: {path}")
            digest.update(chunk)
            remaining -= len(chunk)
    return {
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "footer_bytes": footer_bytes,
        "footer_sha256": digest.hexdigest(),
    }


def read_frozen_split(path: Path) -> tuple[dict[str, Any], str]:
    """Read and pin the small F0 manifest before any payload is created."""
    stat = path.stat()
    if stat.st_size <= 0 or stat.st_size > MAX_SPLIT_BYTES:
        raise ValueError("fold-0 split manifest exceeds its bounded read size")
    raw = path.read_bytes()
    if len(raw) != stat.st_size:
        raise ValueError("fold-0 split changed while being read")
    digest = hashlib.sha256(raw).hexdigest()
    if digest != EXPECTED_FOLD0_SHA256:
        raise ValueError(f"unexpected fold-0 split SHA-256: {digest}")
    manifest = json.loads(raw)
    if int(manifest.get("fold", -1)) != 0 or int(manifest.get("n_folds", -1)) != 4:
        raise ValueError("unsupported fold-0 campaign split")
    return manifest, digest


def select_sequences(manifest: dict[str, Any]) -> tuple[list[dict[str, int]], dict[str, Any]]:
    """Select first seed-3 fit-train groups in distinct non-evaluation components."""
    fit_groups = [int(value) for value in manifest["fit"]["train"]]
    if len(fit_groups) != 10_535 or len(set(fit_groups)) != len(fit_groups):
        raise ValueError("unexpected F0 fit-train membership")

    train_nodes = {
        int(node["row_group"]): {
            "sequence_id": int(node["seq_ix"]),
            "component_id": int(node["component"]),
        }
        for node in manifest["sequences"]["train"]
    }
    valid_nodes = {
        int(node["row_group"]): int(node["component"])
        for node in manifest["sequences"]["valid"]
    }
    evaluation_components = {
        valid_nodes[int(group)] for group in manifest["evaluation"]["valid"]
    }
    if not set(fit_groups).issubset(train_nodes):
        raise ValueError("fit-train list contains an unknown source-qualified row group")

    membership = np.asarray(fit_groups, dtype="<i8")
    permutation = np.random.default_rng(SEED).permutation(membership)
    selected: list[dict[str, int]] = []
    selected_components: set[int] = set()
    for source_position, raw_group in enumerate(permutation):
        group = int(raw_group)
        identity = train_nodes[group]
        component = identity["component_id"]
        if component in evaluation_components:
            raise ValueError("F0 fitting membership overlaps an evaluation component")
        if component in selected_components:
            continue
        selected.append(
            {
                "source_position": source_position,
                "train_row_group": group,
                **identity,
            }
        )
        selected_components.add(component)
        if len(selected) == SLOT_COUNT:
            break
    if len(selected) != SLOT_COUNT:
        raise ValueError("could not select eight distinct fitting components")

    receipt = {
        "algorithm": (
            "numpy.default_rng(seed).permutation(fold0.fit.train); scan in order "
            "and retain the first train row group from each distinct component"
        ),
        "seed": SEED,
        "numpy_version": np.__version__,
        "input_source_order": "fold0.json fit.train list order",
        "candidate_count": len(fit_groups),
        "candidate_membership_sha256": sha256_bytes(memoryview(membership).cast("B")),
        "permuted_candidate_order_sha256": sha256_bytes(
            memoryview(np.asarray(permutation, dtype="<i8")).cast("B")
        ),
        "selected_order_scope": (
            "fixture-selection order only; not a DataLoader minibatch-order claim"
        ),
        "selected": [
            {
                "slot": slot,
                "source": "train",
                **identity,
            }
            for slot, identity in enumerate(selected)
        ],
    }
    receipt["sampler_manifest_sha256"] = canonical_sha256(receipt)
    return selected, receipt


def fsync_file(path: Path) -> None:
    """Flush file contents and metadata before publishing payload paths."""
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=SCRIPT_ROOT
        / "research/campaign_2026-10-15/vast_a_cpu_probe_r1/CPU_TIMING_FIXTURE_R1.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = args.data_root.resolve()
    artifact_dir = args.artifact_dir.resolve()
    manifest_out = args.manifest_out.resolve()
    split_path = data_root / SPLIT_PATH
    train_path = data_root / TRAIN_PATH

    split, split_sha256 = read_frozen_split(split_path)
    source_record = split["sources"][TRAIN_PATH]
    source_stat = parquet_footer_identity(train_path)
    for key in ("bytes", "mtime_ns", "footer_bytes", "footer_sha256"):
        if source_stat[key] != source_record.get(key):
            raise ValueError(f"train source {key} differs from frozen split")
    source_identity = {
        "dataset_path": TRAIN_PATH,
        **source_stat,
    }
    source_identity_sha256 = canonical_sha256(source_identity)
    selected, sampler_manifest = select_sequences(split)

    if artifact_dir.exists():
        if any(artifact_dir.iterdir()):
            raise FileExistsError(f"refusing to overwrite fixture payloads: {artifact_dir}")
    else:
        artifact_dir.mkdir(parents=True)
    try:
        payload_root = artifact_dir.relative_to(data_root).as_posix()
    except ValueError as error:
        raise ValueError("payload directory must remain under the source repository root") from error
    if payload_root.startswith(("/tmp/", "/dev/shm/")):
        raise ValueError("payload directory must be on persistent physical storage")
    if manifest_out.exists():
        raise FileExistsError(f"refusing to overwrite fixture manifest: {manifest_out}")

    feature_shape = (SLOT_COUNT, SEQUENCE_LENGTH, N_FEATURES)
    request_shape = (SLOT_COUNT, SEQUENCE_LENGTH)
    feature_bytes = int(np.prod(feature_shape)) * np.dtype("<f4").itemsize
    request_bytes = int(np.prod(request_shape)) * np.dtype("u1").itemsize
    feature_partial = artifact_dir / ".features.bin.partial"
    request_partial = artifact_dir / ".need_prediction.bin.partial"
    feature_path = artifact_dir / "features.bin"
    request_path = artifact_dir / "need_prediction.bin"
    if any(path.exists() for path in (feature_partial, request_partial, feature_path, request_path)):
        raise FileExistsError("fixture payload path already exists")

    feature_map = np.memmap(
        feature_partial, mode="w+", dtype="<f4", shape=feature_shape, order="C"
    )
    request_map = np.memmap(
        request_partial, mode="w+", dtype="u1", shape=request_shape, order="C"
    )
    request_row = np.zeros(SEQUENCE_LENGTH, dtype=np.uint8)
    request_row[WARMUP:] = 1
    per_sequence: list[dict[str, Any]] = []
    sequence_file = SequenceFile(train_path)
    try:
        for slot, identity in enumerate(selected):
            sequence = sequence_file.read(
                identity["train_row_group"], targets=False, mask=False
            )
            if sequence.seq_ix != identity["sequence_id"]:
                raise ValueError(
                    "source sequence ID disagrees with source-qualified F0 split"
                )
            if sequence.targets is not None or sequence.scored is not None:
                raise ValueError("target or scoring-mask payload entered the fixture reader")
            if sequence.features.shape != (SEQUENCE_LENGTH, N_FEATURES):
                raise ValueError("source row group is not one complete 20k sequence")
            features = np.ascontiguousarray(sequence.features, dtype="<f4")
            if not np.isfinite(features).all():
                raise ValueError(
                    f"non-finite feature value in train row group {identity['train_row_group']}"
                )
            feature_view = memoryview(features).cast("B")
            feature_digest = sha256_bytes(feature_view)
            feature_map[slot] = features
            request_map[slot] = request_row
            per_sequence.append(
                {
                    "slot": slot,
                    "source": "train",
                    "train_row_group": identity["train_row_group"],
                    "source_position": identity["source_position"],
                    "sequence_id": identity["sequence_id"],
                    "component_id": identity["component_id"],
                    "feature_sha256": feature_digest,
                    "need_prediction_sha256": sha256_bytes(
                        memoryview(request_row).cast("B")
                    ),
                }
            )
            del feature_view, features, sequence
    finally:
        feature_map.flush()
        request_map.flush()
        del feature_map
        del request_map
        sequence_file._pf.close()

    if feature_partial.stat().st_size != feature_bytes:
        raise ValueError("feature memmap has an unexpected byte length")
    if request_partial.stat().st_size != request_bytes:
        raise ValueError("request memmap has an unexpected byte length")
    fsync_file(feature_partial)
    fsync_file(request_partial)
    os.replace(feature_partial, feature_path)
    os.replace(request_partial, request_path)

    feature_readback = np.memmap(
        feature_path, mode="r", dtype="<f4", shape=feature_shape, order="C"
    )
    request_readback = np.memmap(
        request_path, mode="r", dtype="u1", shape=request_shape, order="C"
    )
    try:
        if not np.array_equal(request_readback[0], request_row):
            raise ValueError("request schedule failed readback validation")
        for slot, record in enumerate(per_sequence):
            if sha256_bytes(memoryview(feature_readback[slot]).cast("B")) != record[
                "feature_sha256"
            ]:
                raise ValueError(f"feature readback hash mismatch at slot {slot}")
    finally:
        del feature_readback
        del request_readback

    if parquet_footer_identity(train_path) != source_stat:
        raise ValueError("train source identity changed during fixture preparation")
    source_map = {
        **source_identity,
        "source_identity_sha256": source_identity_sha256,
        "source_identity_scope": (
            "frozen fold-0 Parquet footer SHA-256, file byte length and mtime_ns; "
            "the 29GB parquet body was not rescanned"
        ),
        "sampler_manifest_sha256": sampler_manifest["sampler_manifest_sha256"],
        "split_path": SPLIT_PATH,
        "split_sha256": split_sha256,
        "fold": 0,
        "seed": SEED,
    }
    import wnn.data as data_module

    manifest = {
        "format": "wnn.vast-cpu-sequence-panel",
        "version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "runtime-only CPU timing fixture; no quality evaluation",
        "payload_root": payload_root,
        "source": source_map,
        "sampler_manifest": sampler_manifest,
        "arrays": {
            "features": {
                "file": "features.bin",
                "dtype": "<f4",
                "shape": list(feature_shape),
                "bytes": feature_bytes,
                "sha256": sha256_file(feature_path),
            },
            "need_prediction": {
                "file": "need_prediction.bin",
                "dtype": "|u1",
                "shape": list(request_shape),
                "bytes": request_bytes,
                "sha256": sha256_file(request_path),
            },
        },
        "sequences": per_sequence,
        "request_schedule": {
            "kind": "fixed runtime request vector, not an official scoring mask",
            "false_rows": [0, WARMUP - 1],
            "true_rows": [WARMUP, SEQUENCE_LENGTH - 1],
            "need_prediction_sha256_per_sequence": per_sequence[0][
                "need_prediction_sha256"
            ],
        },
        "excluded_payloads": [
            "targets",
            "official_scoring_mask",
            "teacher_outputs",
            "distillation_cache_values",
            "propensity_values",
        ],
        "reader": {
            "api": "wnn.data.SequenceFile.read",
            "targets": False,
            "mask": False,
            "complete_sequence_rows": SEQUENCE_LENGTH,
            "features": N_FEATURES,
        },
        "producer": {
            "generator_path": "scripts/prepare_vast_a_cpu_probe_fixture.py",
            "generator_sha256": sha256_file(Path(__file__).resolve()),
            "sequence_reader_path": "src/wnn/data.py",
            "sequence_reader_sha256": sha256_file(Path(data_module.__file__).resolve()),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pyarrow": pyarrow.__version__,
        },
    }
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    if manifest_out.exists():
        raise FileExistsError(f"refusing to overwrite fixture manifest: {manifest_out}")
    temporary_manifest = manifest_out.with_name(f".{manifest_out.name}.partial")
    if temporary_manifest.exists():
        raise FileExistsError(f"refusing to overwrite partial manifest: {temporary_manifest}")
    encoded = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with temporary_manifest.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary_manifest, manifest_out)
    print(
        json.dumps(
            {
                "manifest_path": str(manifest_out),
                "manifest_sha256": sha256_file(manifest_out),
                "payload_root": str(artifact_dir),
                "feature_bytes": feature_bytes,
                "need_prediction_bytes": request_bytes,
                "selected": per_sequence,
            },
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
