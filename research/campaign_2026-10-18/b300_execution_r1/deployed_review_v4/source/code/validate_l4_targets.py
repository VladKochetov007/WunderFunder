#!/usr/bin/env python3
"""Validate target cache indexes and values against the supplied Parquets."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from canonical_manifest_checks import load_canonical_manifests, validate_manifest_sequence_ids

TARGETS = {
    "train": (
        1_909_335_790,
        "ead1a05e7aa5ac3d1c10cde65986be9f48fd342cadb95330611e58c3a8f817e9",
        "4e52088dc9fd01c19fa0d2558e87f60559ce84d1ab83cc3952c0ab9d9512c597",
    ),
    "valid": (
        337_186_198,
        "dbb7a1a48dd0af9fa524a80c2694008a56cb91d5dae80e8d0d4984c0c5622955",
        "ed030f767a6c650db666844ecffef21f0e96cc9fa2d7e77a0872bdf26c5bed38",
    ),
}
PARQUETS = {
    "train": (29_240_219_011, "4a7756449a4bcf272cf715bb405a2152987041605f6c1cce650243ab2abf4e81"),
    "valid": (5_615_509_128, "2861ee30914065702626a3ac033d80d0563cdacaea50c667fc828c2eeb418782"),
}


def sha256_file(path: Path) -> str:
    """Hash a file incrementally."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def index_digest(row_groups: np.ndarray, sequence_ids: np.ndarray) -> str:
    """Match the campaign manifest's source-qualified target-index digest."""
    digest = hashlib.sha256()
    for values in (row_groups, sequence_ids):
        digest.update(np.asarray(values, dtype="<i8").tobytes())
    return digest.hexdigest()


def validate_group_order(split: str, row_groups: np.ndarray, n_row_groups: int) -> list[int]:
    """Require target row groups to follow source Parquet order."""
    if split not in {"train", "valid"}:
        raise ValueError(f"unsupported source split: {split}")
    expected = [group for group in range(n_row_groups) if split != "train" or group != 528]
    if np.asarray(row_groups).tolist() != expected:
        raise ValueError(f"{split} target row_group array differs from canonical parquet order")
    return expected


def validate_target_index(
    split: str,
    target_path: Path,
    parquet_path: Path,
    mask_path: Path | None,
    canonical_manifests: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Check cached row-group/sequence IDs against every supplied parquet group."""
    import pyarrow.parquet as pq

    expected_bytes, expected_sha, expected_index_sha = TARGETS[split]
    with np.load(target_path, allow_pickle=False) as cache:
        raw_groups = np.asarray(cache["row_group"])
        raw_sequence_ids = np.asarray(cache["seq_ix"])
        cached_mask = np.asarray(cache["mask"]) if split == "valid" else None
    if (
        raw_groups.ndim != 1
        or raw_sequence_ids.ndim != 1
        or raw_groups.dtype.kind not in "iu"
        or raw_sequence_ids.dtype.kind not in "iu"
        or len(raw_groups) != len(raw_sequence_ids)
    ):
        raise ValueError(f"{split} target index arrays have incompatible shapes")
    row_groups = raw_groups.astype(np.int64, copy=False)
    sequence_ids = raw_sequence_ids.astype(np.int64, copy=False)
    if len(set(row_groups.tolist())) != len(row_groups) or len(set(sequence_ids.tolist())) != len(
        sequence_ids
    ):
        raise ValueError(f"{split} target index contains duplicate identities")
    parquet = pq.ParquetFile(parquet_path)
    expected_groups = validate_group_order(split, row_groups, parquet.num_row_groups)
    validate_manifest_sequence_ids(canonical_manifests, split, row_groups, sequence_ids)
    if split == "valid" and (
        cached_mask is None
        or cached_mask.shape != (len(row_groups), 20_000)
        or cached_mask.dtype.kind != "b"
    ):
        raise ValueError("valid target mask must be boolean [N,20000]")
    external_masks: dict[int, np.ndarray] = {}
    mask_parquet = pq.ParquetFile(mask_path) if split == "valid" and mask_path is not None else None
    if split == "valid" and mask_parquet is None:
        raise ValueError("valid-mask parquet is required for independent mask alignment")
    if mask_parquet is not None:
        if mask_parquet.num_row_groups != parquet.num_row_groups:
            raise ValueError("valid-mask parquet row-group count differs from valid parquet")
        for group in range(mask_parquet.num_row_groups):
            table = mask_parquet.read_row_group(
                group, columns=["seq_ix", "step_in_seq", "is_scored"], use_threads=False
            )
            seq = table.column("seq_ix").to_numpy()
            step = table.column("step_in_seq").to_numpy()
            scored = table.column("is_scored").to_numpy().astype(bool, copy=False)
            if (
                table.num_rows != 20_000
                or np.any(seq != seq[0])
                or not np.array_equal(step, np.arange(20_000))
            ):
                raise ValueError(
                    f"valid-mask row group {group} has invalid sequence identity/order"
                )
            seq_ix = int(seq[0])
            if seq_ix in external_masks:
                raise ValueError("valid-mask parquet repeats a sequence id")
            external_masks[seq_ix] = scored
    mask_checked = 0
    targets_checked = 0
    with zipfile.ZipFile(target_path) as archive, archive.open("y.npy") as target_stream:
        version = np.lib.format.read_magic(target_stream)
        shape, fortran_order, dtype = np.lib.format._read_array_header(target_stream, version)
        if shape != (len(row_groups), 20_000, 2) or fortran_order or dtype != np.dtype(np.float32):
            raise ValueError(f"{split} target values must be C-order float32 [N,20000,2]")
        target_row_bytes = 20_000 * 2 * np.dtype(np.float32).itemsize
        for index, group in enumerate(expected_groups):
            columns = ["seq_ix", "t0", "t1"] + (
                ["step_in_seq", "is_scored"] if split == "valid" else []
            )
            table = parquet.read_row_group(group, columns=columns, use_threads=False)
            if table.num_rows != 20_000:
                raise ValueError(f"{split} row group {group} has unexpected row count")
            values = table.column("seq_ix").to_numpy()
            if np.any(values != values[0]) or int(values[0]) != int(sequence_ids[index]):
                raise ValueError(f"{split} seq_ix mismatch at row group {group}")
            actual_targets = np.column_stack(
                (table.column("t0").to_numpy(), table.column("t1").to_numpy())
            ).astype(np.float32, copy=False)
            raw_targets = target_stream.read(target_row_bytes)
            if len(raw_targets) != target_row_bytes or not np.array_equal(
                np.frombuffer(raw_targets, dtype=np.float32).reshape(20_000, 2), actual_targets
            ):
                raise ValueError(f"{split} target values differ at row group {group}")
            targets_checked += 1
            if split == "valid":
                step = table.column("step_in_seq").to_numpy()
                scored = table.column("is_scored").to_numpy().astype(bool, copy=False)
                if not np.array_equal(step, np.arange(20_000)) or not np.array_equal(
                    cached_mask[index], scored
                ):
                    raise ValueError(f"valid target mask/order mismatch at row group {group}")
                if not np.array_equal(cached_mask[index], external_masks[int(sequence_ids[index])]):
                    raise ValueError(
                        f"valid-mask parquet differs at seq_ix {int(sequence_ids[index])}"
                    )
                mask_checked += 1
    actual_sha = sha256_file(target_path)
    expected_parquet_bytes, expected_parquet_sha = PARQUETS[split]
    parquet_sha = sha256_file(parquet_path)
    actual_index_sha = index_digest(row_groups, sequence_ids)
    mismatch_reasons = []
    if target_path.stat().st_size != expected_bytes or actual_sha != expected_sha:
        mismatch_reasons.append("target cache archive differs from the frozen R4 encoding")
    if actual_index_sha != expected_index_sha:
        mismatch_reasons.append("target cache index digest differs from frozen R4")
    if parquet_path.stat().st_size != expected_parquet_bytes or parquet_sha != expected_parquet_sha:
        mismatch_reasons.append("raw parquet differs from the frozen R4 source identity")
    return {
        "split": split,
        "path": str(target_path.resolve()),
        "bytes": target_path.stat().st_size,
        "sha256": actual_sha,
        "expected_r4_bytes": expected_bytes,
        "expected_r4_sha256": expected_sha,
        "index_sha256": actual_index_sha,
        "expected_r4_index_sha256": expected_index_sha,
        "parquet": {
            "path": str(parquet_path.resolve()),
            "bytes": parquet_path.stat().st_size,
            "sha256": parquet_sha,
            "expected_r4_bytes": expected_parquet_bytes,
            "expected_r4_sha256": expected_parquet_sha,
        },
        "valid_mask_parquet": {
            "path": str(mask_path.resolve()),
            "bytes": mask_path.stat().st_size,
            "sha256": sha256_file(mask_path),
        }
        if mask_path is not None
        else None,
        "identity_class": "r4_exact"
        if target_path.stat().st_size == expected_bytes
        and actual_sha == expected_sha
        and actual_index_sha == expected_index_sha
        and parquet_path.stat().st_size == expected_parquet_bytes
        and parquet_sha == expected_parquet_sha
        else "new_attempt_local_reencoded",
        "identity_mismatch_reasons": mismatch_reasons,
        "index_content_verified_against_parquet": True,
        "target_values_checked_against_parquet_row_groups": targets_checked,
        "valid_mask_parquet_mask_rows_checked": mask_checked,
        "fresh_source_bound_manifest_required_if_reencoded": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Validate both target caches and write their verified identities."""
    parser = argparse.ArgumentParser(description=__doc__)
    for split in ("train", "valid"):
        parser.add_argument(f"--{split}-targets", type=Path, required=True)
        parser.add_argument(f"--{split}-parquet", type=Path, required=True)
    parser.add_argument("--valid-mask-parquet", type=Path, required=True)
    parser.add_argument("--canonical-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.receipt.exists():
        raise FileExistsError(args.receipt)
    canonical_manifests = load_canonical_manifests(args.canonical_dir)
    result = {
        split: validate_target_index(
            split,
            getattr(args, f"{split}_targets"),
            getattr(args, f"{split}_parquet"),
            args.valid_mask_parquet if split == "valid" else None,
            canonical_manifests,
        )
        for split in ("train", "valid")
    }
    raw_exact = all(
        result[split]["parquet"]["bytes"] == PARQUETS[split][0]
        and result[split]["parquet"]["sha256"] == PARQUETS[split][1]
        for split in ("train", "valid")
    )
    result["preflight_status"] = (
        "verified_r4_raw_parquets" if raw_exact else "blocked_raw_parquet_identity_mismatch"
    )
    result["manifest_generation_allowed"] = raw_exact
    if not raw_exact:
        result["preflight_block_reason"] = (
            "Raw train/valid Parquet hashes must match frozen R4 inputs before graph/fold "
            "derivation or fresh campaign manifest generation."
        )
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0 if raw_exact else 2


if __name__ == "__main__":
    raise SystemExit(main())
