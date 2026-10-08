#!/usr/bin/env python3
"""Prepare source-qualified cohort membership and build parsed sequence caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from wnn.parsed_sequence_cache import (
    CacheAdmissionError,
    build_parsed_cache,
    canonical_json_bytes,
    prepare_membership,
    sha256_bytes,
)


def _positive_or_zero_bytes(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer byte count") from exc
    if result < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return result


def _positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if result < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return result


def _sha256_pin(value: str) -> str:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise argparse.ArgumentTypeError("must be a lowercase SHA-256 hex digest")
    return value


def _prepare(args: argparse.Namespace) -> dict[str, Any]:
    membership = prepare_membership(
        split_manifest_path=args.split_manifest,
        portable_registry_path=args.portable_registry,
        expected_split_sha256=args.expected_split_sha256,
        expected_registry_sha256=args.expected_registry_sha256,
        fold=args.fold,
        fraction=args.fraction,
        seed=args.seed,
    )
    if args.output.exists() or args.output.is_symlink():
        raise CacheAdmissionError("membership output already exists; refusing to overwrite")
    if not args.output.parent.is_dir():
        raise CacheAdmissionError("membership output parent must already exist")
    payload = canonical_json_bytes(membership) + b"\n"
    with args.output.open("xb") as stream:
        stream.write(payload)
        stream.flush()
    return {
        "membership_path": str(args.output),
        "membership_sha256": sha256_bytes(payload),
        "fold": membership["fold"],
        "selection": membership["selection"],
        "member_counts": membership["member_counts"],
        "array_shapes": membership["array_shapes"],
    }


def _build(args: argparse.Namespace) -> dict[str, Any]:
    manifest = build_parsed_cache(
        membership_path=args.membership,
        expected_membership_sha256=args.expected_membership_sha256,
        source_root=args.source_root,
        output_dir=args.output_dir,
        output_reserve_bytes=args.output_reserve_bytes,
        checkpoint_reserve_bytes=args.checkpoint_reserve_bytes,
        free_reserve_bytes=args.free_reserve_bytes,
        source_verification_path=args.source_verification,
        expected_source_verification_sha256=args.expected_source_verification_sha256,
    )
    return {
        "manifest_path": str(args.output_dir / "manifest.json"),
        "cache_key": manifest["cache_key"],
        "fold": manifest["fold"],
        "member_counts": manifest["member_counts"],
        "array_bytes": manifest["estimated_array_bytes_before_admission"],
        "observed_parse_once": manifest["telemetry"]["observed_parse_once"],
        "sequence_read_invocations": manifest["telemetry"]["parsing"]["sequence_read_invocations"],
        "mapped_virtual_bytes": manifest["telemetry"]["mmap_validation"]["virtual_mapping_bytes"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare-membership", help="freeze component-preserving membership"
    )
    prepare.add_argument("--split-manifest", type=Path, required=True)
    prepare.add_argument("--portable-registry", type=Path, required=True)
    prepare.add_argument("--expected-split-sha256", type=_sha256_pin, required=True)
    prepare.add_argument("--expected-registry-sha256", type=_sha256_pin, required=True)
    prepare.add_argument("--fold", type=_positive_int, required=True)
    prepare.add_argument("--fraction", type=float, required=True)
    prepare.add_argument("--seed", type=int, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.set_defaults(run=_prepare)

    build = subparsers.add_parser("build", help="build arrays from a pinned membership artifact")
    build.add_argument("--membership", type=Path, required=True)
    build.add_argument("--expected-membership-sha256", type=_sha256_pin, required=True)
    build.add_argument("--source-root", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--output-reserve-bytes", type=_positive_or_zero_bytes, required=True)
    build.add_argument("--checkpoint-reserve-bytes", type=_positive_or_zero_bytes, required=True)
    build.add_argument("--free-reserve-bytes", type=_positive_or_zero_bytes, required=True)
    build.add_argument("--source-verification", type=Path)
    build.add_argument("--expected-source-verification-sha256", type=_sha256_pin)
    build.set_defaults(run=_build)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        result = args.run(args)
    except (CacheAdmissionError, OSError, json.JSONDecodeError) as exc:
        parser.exit(2, f"parsed sequence cache rejected: {exc}\n")
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
