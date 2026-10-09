#!/usr/bin/env python3
"""Verify a mounted parsed cache once, then profile bounded mmap replay."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, cast

import numpy as np

from wnn.parsed_sequence_cache_reader import (
    ArrayRole,
    ParsedSequenceCache,
    SourceName,
    _cgroup_snapshot,
    _phase_io,
    verify_parsed_cache,
)


def _pin(value: str) -> str:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise argparse.ArgumentTypeError("expected a lowercase SHA-256 pin")
    return value


def _role_samples(cache: ParsedSequenceCache, limit: int, seed: int) -> list[tuple[str, str, int]]:
    rng = np.random.default_rng(seed)
    samples: list[tuple[str, str, int]] = []
    for role, split in (
        ("fit_train", "train"),
        ("fit_valid", "valid"),
        ("evaluation_valid", "valid"),
    ):
        groups = cache.groups(cast(ArrayRole, role))
        if not groups:
            raise ValueError(f"empty cache role: {role}")
        count = min(limit, len(groups))
        indices = np.sort(rng.choice(len(groups), size=count, replace=False))
        samples.extend((role, split, groups[int(index)]) for index in indices)
    return samples


def profile_cache(args: argparse.Namespace) -> dict[str, Any]:
    receipt, receipt_sha256 = verify_parsed_cache(
        args.manifest, args.expected_manifest_sha256, args.validation_receipt
    )
    cache = ParsedSequenceCache.open(
        args.manifest,
        expected_manifest_sha256=args.expected_manifest_sha256,
        validation_receipt_path=args.validation_receipt,
        expected_validation_receipt_sha256=receipt_sha256,
        expected_cache_key=receipt["cache_key"],
        expected_split_manifest_sha256=receipt["split_manifest_sha256"],
        expected_fold=receipt["fold"],
    )
    before_io, before_memory = _phase_io(), _cgroup_snapshot()
    started = time.perf_counter()
    samples = _role_samples(cache, args.per_role, args.seed)
    checksum = 0.0
    bytes_copied = 0
    for role, split, group in samples:
        sequence = cache.read(cast(SourceName, split), group, cast(ArrayRole, role))
        checksum += float(sequence.features[0, 0]) + float(sequence.targets[-1, 0])
        bytes_copied += sequence.features.nbytes + sequence.targets.nbytes + sequence.scored.nbytes
    elapsed = time.perf_counter() - started
    after_io, after_memory = _phase_io(), _cgroup_snapshot()
    cache.close()
    return {
        "schema": "wnn.parsed-sequence-cache-reader-profile",
        "format_version": 1,
        "manifest_sha256": args.expected_manifest_sha256,
        "cache_key": receipt["cache_key"],
        "validation_receipt_path": str(args.validation_receipt),
        "validation_receipt_sha256": receipt_sha256,
        "full_array_hash_verification": receipt["verification"],
        "mmap_sample": {
            "per_role_limit": args.per_role,
            "sequences_read": len(samples),
            "bytes_copied_to_owned_arrays": bytes_copied,
            "elapsed_seconds": elapsed,
            "process_io_before": before_io,
            "process_io_after": after_io,
            "memory_before": before_memory,
            "memory_after": after_memory,
            "checksum": checksum,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", type=_pin, required=True)
    parser.add_argument("--validation-receipt", type=Path, required=True)
    parser.add_argument("--profile-output", type=Path, required=True)
    parser.add_argument("--per-role", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261009)
    args = parser.parse_args()
    if not 1 <= args.per_role <= 8:
        parser.error("--per-role must be between 1 and 8")
    if args.profile_output.exists() or args.profile_output.is_symlink():
        parser.error("profile output already exists; refusing to overwrite")
    result = profile_cache(args)
    raw = json.dumps(result, sort_keys=True, indent=2) + "\n"
    args.profile_output.parent.mkdir(parents=True, exist_ok=True)
    with args.profile_output.open("x", encoding="utf-8") as stream:
        stream.write(raw)
        stream.flush()
    print(
        json.dumps(
            {
                "manifest_sha256": result["manifest_sha256"],
                "cache_key": result["cache_key"],
                "validation_receipt_sha256": result["validation_receipt_sha256"],
                "sample_sequences": result["mmap_sample"]["sequences_read"],
                "sample_seconds": result["mmap_sample"]["elapsed_seconds"],
                "profile_path": str(args.profile_output),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
