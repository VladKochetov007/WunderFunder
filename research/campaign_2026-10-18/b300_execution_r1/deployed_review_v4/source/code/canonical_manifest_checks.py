#!/usr/bin/env python3
"""Validate canonical sequence identities and compare fresh split memberships."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from derive_l4_folds import MANIFEST_SHA256

MEMBERSHIP_FIELDS = ("fit", "evaluation", "excluded", "sequences")


def load_canonical_manifests(directory: Path) -> list[dict[str, object]]:
    """Load the four pinned manifests after checking their exact content hashes."""
    manifests = []
    for fold, expected_sha in MANIFEST_SHA256.items():
        path = directory / f"fold{fold}.json"
        payload = path.read_bytes()
        if len(payload) > 2_000_000 or hashlib.sha256(payload).hexdigest() != expected_sha:
            raise ValueError(f"canonical manifest fold{fold} identity mismatch")
        manifest = json.loads(payload)
        if manifest.get("fold") != fold or manifest.get("n_folds") != 4:
            raise ValueError(f"canonical manifest fold{fold} metadata mismatch")
        manifests.append(manifest)
    return manifests


def validate_manifest_sequence_ids(
    manifests: Sequence[Mapping[str, object]],
    split: str,
    row_groups: Sequence[int],
    sequence_ids: Sequence[int],
) -> None:
    """Require every canonical fold's sequence map to match the target cache index."""
    if split not in {"train", "valid"}:
        raise ValueError(f"unsupported source split: {split}")
    if len(row_groups) != len(sequence_ids):
        raise ValueError("target row groups and sequence IDs differ in length")
    expected: dict[int, int] | None = None
    for fold, manifest in enumerate(manifests):
        sequences = manifest.get("sequences")
        if not isinstance(sequences, Mapping) or split not in sequences:
            raise ValueError(f"canonical fold{fold} lacks {split} sequence identities")
        rows = sequences[split]
        if not isinstance(rows, list):
            raise ValueError(f"canonical fold{fold} {split} sequence rows are malformed")
        current: dict[int, int] = {}
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError(f"canonical fold{fold} {split} row is malformed")
            group, sequence_id = int(row["row_group"]), int(row["seq_ix"])
            if group in current:
                raise ValueError(f"canonical fold{fold} repeats {split} row group {group}")
            current[group] = sequence_id
        if expected is None:
            expected = current
        elif expected != current:
            raise ValueError(f"canonical fold{fold} {split} sequence IDs differ across folds")
    if expected is None:
        raise ValueError("no canonical manifests were provided")
    actual = dict(zip(map(int, row_groups), map(int, sequence_ids), strict=True))
    if actual != expected:
        groups = sorted(set(expected) | set(actual))
        mismatch = next(
            (group for group in groups if expected.get(group) != actual.get(group)), None
        )
        raise ValueError(
            f"{split} target index disagrees with canonical seq_ix at row group {mismatch}"
        )


def compare_fold_memberships(canonical_dir: Path, fresh_dir: Path) -> list[dict[str, int]]:
    """Require new manifests to preserve every original fit/evaluation/exclusion membership."""
    canonical = load_canonical_manifests(canonical_dir)
    comparisons = []
    for fold, original in enumerate(canonical):
        fresh_path = fresh_dir / f"fold{fold}.json"
        fresh = json.loads(fresh_path.read_bytes())
        if fresh.get("fold") != fold or fresh.get("n_folds") != 4:
            raise ValueError(f"fresh manifest fold{fold} metadata mismatch")
        assert_memberships_equal(original, fresh, fold)
        comparisons.append({"fold": fold, "membership_fields_checked": len(MEMBERSHIP_FIELDS)})
    return comparisons


def assert_memberships_equal(
    canonical: Mapping[str, object], fresh: Mapping[str, object], fold: int
) -> None:
    """Reject any changed fit, evaluation, exclusion, or source sequence membership."""
    for field in MEMBERSHIP_FIELDS:
        if canonical.get(field) != fresh.get(field):
            raise ValueError(f"fold{fold} {field} membership differs from canonical")


def main(argv: Sequence[str] | None = None) -> int:
    """Compare all four regenerated folds with the immutable canonical memberships."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-dir", type=Path, required=True)
    parser.add_argument("--fresh-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = {
        "schema": "tabdata-l4-membership-comparison.v1",
        "status": "exact_membership_match",
        "folds": compare_fold_memberships(args.canonical_dir, args.fresh_dir),
    }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
