"""Run a single bounded NumPy timing pass on the pinned raw CPU fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from wnn.research_features import research_features_numpy

MANIFEST_SHA256 = "941b656d8408be6f5c8f5816c1e32f3c8b24e59040bb4ed84bb895078a9f9f05"
FEATURES_SHA256 = "e8e8e1268cea54be12c3cf93e1be79b18cb097a1584851bcc4f5dc65885dd337"
FEATURE_SHAPE = (8, 20_000, 112)
FEATURE_BYTES = math.prod(FEATURE_SHAPE) * np.dtype("<f4").itemsize
MAX_MANIFEST_BYTES = 1_000_000
HASH_BLOCK_BYTES = 1 << 20


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(HASH_BLOCK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def _sha256_buffer(array: np.ndarray) -> str:
    if not array.flags.c_contiguous:
        raise ValueError("fixture slot must be contiguous for byte hashing")
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def _load_manifest(path: Path) -> tuple[dict[str, object], str]:
    size = path.stat().st_size
    if size < 1 or size > MAX_MANIFEST_BYTES:
        raise ValueError(f"fixture manifest size is outside the bound: {size}")
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != MANIFEST_SHA256:
        raise ValueError("fixture manifest SHA-256 differs from the frozen identity")
    manifest = json.loads(payload)
    if manifest.get("format") != "wnn.vast-cpu-sequence-panel":
        raise ValueError("unexpected fixture manifest format")
    if manifest.get("arrays", {}).get("features", {}).get("sha256") != FEATURES_SHA256:
        raise ValueError("fixture feature identity differs from the frozen payload")
    if manifest["arrays"]["features"].get("shape") != list(FEATURE_SHAPE):
        raise ValueError("fixture feature geometry differs from the frozen payload")
    if manifest["arrays"]["features"].get("bytes") != FEATURE_BYTES:
        raise ValueError("fixture feature byte count differs from its geometry")
    return manifest, digest


def _slot_hashes(manifest: dict[str, object], features: np.memmap) -> list[dict[str, object]]:
    sequences = manifest.get("sequences")
    if not isinstance(sequences, list) or len(sequences) != FEATURE_SHAPE[0]:
        raise ValueError("fixture must bind exactly eight source-qualified sequences")
    actual: list[dict[str, object]] = []
    for slot, metadata in enumerate(sequences):
        if metadata.get("slot") != slot:
            raise ValueError("fixture sequence slots must be ordered and complete")
        row = features[slot]
        if row.shape != FEATURE_SHAPE[1:] or not np.isfinite(row).all():
            raise ValueError(f"fixture slot {slot} has invalid shape or non-finite values")
        digest = _sha256_buffer(row)
        if digest != metadata.get("feature_sha256"):
            raise ValueError(f"fixture slot {slot} SHA-256 differs from its manifest")
        actual.append(
            {
                key: metadata[key]
                for key in (
                    "slot",
                    "source",
                    "source_position",
                    "train_row_group",
                    "sequence_id",
                    "component_id",
                )
            }
            | {"feature_sha256": digest}
        )
    return actual


def run_profile(manifest_path: Path, payload_root: Path) -> dict[str, object]:
    manifest, manifest_sha = _load_manifest(manifest_path)
    features_info = manifest["arrays"]["features"]
    filename = features_info.get("file")
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ValueError("feature payload name must be a single safe path component")
    features_path = payload_root / filename
    if features_path.stat().st_size != FEATURE_BYTES:
        raise ValueError("feature payload byte count differs from the frozen geometry")
    features_sha = _sha256_file(features_path)
    if features_sha != FEATURES_SHA256:
        raise ValueError("feature payload SHA-256 differs from the frozen identity")
    features = np.memmap(features_path, mode="r", dtype="<f4", shape=FEATURE_SHAPE)
    slots = _slot_hashes(manifest, features)

    timings: dict[str, list[dict[str, object]]] = {"group_age": [], "field_age": []}
    for metadata in slots:
        slot = int(metadata["slot"])
        raw = features[slot]
        for mode in ("group_age", "field_age"):
            started = time.perf_counter()
            encoded = research_features_numpy(raw, mode)
            elapsed = time.perf_counter() - started
            output_hash = _sha256_buffer(encoded)
            timings[mode].append(
                {
                    "slot": slot,
                    "elapsed_seconds": elapsed,
                    "rows_per_second": FEATURE_SHAPE[1] / elapsed,
                    "output_sha256": output_hash,
                }
            )
            del encoded

    summary: dict[str, object] = {
        "format": "wnn.field-age-cpu-profile",
        "version": 1,
        "created_utc": datetime.now(UTC).isoformat(),
        "scope": "single-pass NumPy feature-construction timing; no fit or quality evaluation",
        "runtime": {
            "python": __import__("sys").version.split()[0],
            "numpy": np.__version__,
            "cpu_count_visible": __import__("os").cpu_count(),
        },
        "source": {
            "fixture_manifest_sha256": manifest_sha,
            "fixture_feature_payload_sha256": features_sha,
            "fixture_source_identity_sha256": manifest["source"]["source_identity_sha256"],
            "sampler_manifest_sha256": manifest["sampler_manifest"]["sampler_manifest_sha256"],
            "split_sha256": manifest["source"]["split_sha256"],
            "sequences": slots,
        },
        "payload_scope": {
            "read": ["features.bin"],
            "not_read": [
                "need_prediction.bin",
                "labels",
                "official_scoring_mask",
                "teacher_outputs",
                "distillation_cache_values",
                "propensity_values",
                "train.parquet body",
            ],
        },
        "transform": {
            "definition": "row 0 age 0; exact raw float32 inequality resets age to 0; otherwise increment by 1; cap at 10000; encode log1p(age)",
            "group_ranges_half_open": [
                [start, end]
                for start, end in (
                    (0, 22),
                    (22, 44),
                    (44, 52),
                    (52, 74),
                    (74, 96),
                    (96, 104),
                    (104, 112),
                )
            ],
            "group_age": "one any-field-change clock per range, broadcast to its raw columns",
            "field_age": "one clock per raw column",
        },
        "single_pass_timings": timings,
        "interpretation_limit": "runtime-only exposed fitting features; no predictive-quality or generalization claim",
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--payload-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite a single-pass profile: {args.output}")
    report = run_profile(args.manifest, args.payload_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "status": "complete"}, sort_keys=True))


if __name__ == "__main__":
    main()
