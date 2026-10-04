"""Audit E426's real fold-0 event counts and first-epoch EMA schedules on CPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
STAGING_ROOT = Path(__file__).resolve().parents[1]


def _install_import_overlay() -> None:
    """Use staged event-clock code and hide accelerators before importing Torch."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    roots = [STAGING_ROOT / "src", REPO_ROOT / "src"]
    normalized = {str(path) for path in roots}
    sys.path[:] = [path for path in sys.path if path not in normalized]
    sys.path[0:0] = [str(path) for path in roots]


_install_import_overlay()


def _load_dependencies() -> tuple[object, ...]:
    """Import the staged clock only after installing its package overlay."""
    import numpy
    from wnn.event_ema import (
        endpoint_events_by_update as count_endpoint_events,
    )
    from wnn.event_ema import (
        make_event_ema_schedule as build_event_schedule,
    )
    from wnn.event_ema import (
        per_update_event_counts as aggregate_update_events,
    )

    from wnn.data import CORRUPT_TRAIN_ROW_GROUPS as corrupt_train_groups
    from wnn.splits import CampaignSplit as campaign_split_type
    from wnn.splits import file_sha256 as hash_file
    from wnn.splits import load_campaign_split as read_campaign_split
    from wnn.training_sampling import EpochOrderSampler as epoch_order_sampler

    return (
        numpy,
        corrupt_train_groups,
        count_endpoint_events,
        build_event_schedule,
        aggregate_update_events,
        campaign_split_type,
        hash_file,
        read_campaign_split,
        epoch_order_sampler,
    )


(
    np,
    CORRUPT_TRAIN_ROW_GROUPS,
    endpoint_events_by_update,
    make_event_ema_schedule,
    per_update_event_counts,
    CampaignSplit,
    file_sha256,
    load_campaign_split,
    EpochOrderSampler,
) = _load_dependencies()

FIT_TRAIN_SEQUENCES = 10_607
VALID_REPEAT = 1
SEQUENCE_LENGTH = 20_000


def _validated_source_rows(
    split: CampaignSplit,
    train_groups: Sequence[int],
    valid_groups: Sequence[int],
    *,
    valid_repeat: int,
) -> list[tuple[str, int]]:
    """Apply the production fold and corrupt-group checks before any Parquet read."""
    if valid_repeat != 1:
        raise ValueError("this audit matches the E426 production valid-repeat=1 recipe")
    split.assert_fit_groups(train_groups, valid_groups)
    corrupt_train = set(map(int, train_groups)) & CORRUPT_TRAIN_ROW_GROUPS
    if corrupt_train:
        raise ValueError(f"corrupt TRAIN row groups are forbidden: {sorted(corrupt_train)}")
    return [("train", int(group)) for group in train_groups] + [
        ("valid", int(group)) for group in valid_groups
    ]


def _sha256_json_rows(rows: Sequence[tuple[str, int]]) -> str:
    """Hash ordered source identities with the trainer's compact JSON encoding."""
    payload = json.dumps(rows, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _audit_forbidden_train_hole(
    split: CampaignSplit, train_groups: np.ndarray, valid_groups: np.ndarray
) -> str:
    """Require the real TRAIN-528 manifest violation to fail before counting."""
    bad_train = np.concatenate((train_groups, np.asarray([528], dtype=np.int64)))
    try:
        _validated_source_rows(split, bad_train, valid_groups, valid_repeat=VALID_REPEAT)
    except ValueError as error:
        return str(error)
    raise AssertionError("forbidden TRAIN row group 528 was accepted")


def _load_source_counts(
    root: Path,
    rows: Sequence[tuple[str, int]],
    update_rows: int,
) -> np.ndarray:
    """Read one input-only Parquet row group at a time through the staged helper."""
    grouped: dict[str, list[int]] = {"train": [], "valid": []}
    for split, row_group in rows:
        grouped[split].append(row_group)
    pieces = [
        endpoint_events_by_update(
            root / "datasets" / f"{split}.parquet",
            groups,
            update_rows,
            sequence_length=SEQUENCE_LENGTH,
        )
        for split, groups in grouped.items()
        if groups
    ]
    return np.concatenate(pieces, axis=0)


def audit(
    *,
    root: Path,
    manifest_path: Path,
    seed: int = 3,
    batch_size: int = 32,
    update_rows: int = 1_000,
    base_decay: float = 0.999,
    shuffle_seed: int = 86_173,
) -> dict[str, object]:
    """Build real fold-0 first-epoch event clocks without retaining source rows."""
    if update_rows <= 0 or SEQUENCE_LENGTH % update_rows:
        raise ValueError("update_rows must divide the production sequence length")
    split = load_campaign_split(manifest_path, root)
    manifest_sha256 = file_sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest_sha256 != split.sha256:
        raise AssertionError("loaded split identity differs from manifest file bytes")
    if split.fold != 0:
        raise ValueError(f"expected fold 0, received fold {split.fold}")

    train_membership = np.asarray(split.train_groups, dtype=np.int64)
    train_groups = np.random.default_rng(seed).permutation(train_membership)[:FIT_TRAIN_SEQUENCES]
    valid_groups = np.asarray(split.valid_groups, dtype=np.int64)
    if len(train_groups) != len(train_membership):
        raise AssertionError("E426 must include every declared fold-0 fit TRAIN group")
    if 528 in split.train_groups or 528 in train_groups:
        raise AssertionError("corrupt TRAIN row group 528 appears in real fit membership")
    if 529 not in train_groups:
        raise AssertionError("expected eligible TRAIN row group 529 in the full fit pool")

    # This rejection check runs before _load_source_counts is called. Its bad
    # source uses the real fold manifest, not a synthetic split or stub guard.
    rejection = _audit_forbidden_train_hole(split, train_groups, valid_groups)
    source_rows = _validated_source_rows(
        split, train_groups, valid_groups, valid_repeat=VALID_REPEAT
    )
    if len(source_rows) != len(train_groups) + len(valid_groups):
        raise AssertionError("ordered event-source list differs from the fit dataset")

    sequence_counts = _load_source_counts(root, source_rows, update_rows)
    updates_per_sequence = SEQUENCE_LENGTH // update_rows
    sampler = EpochOrderSampler(len(source_rows), seed, batch_size=batch_size, drop_last=True)
    sampler.set_epoch(0)
    epoch_order = list(iter(sampler))
    observed = per_update_event_counts(sequence_counts, epoch_order, batch_size, drop_last=True)
    updates_per_epoch = (len(epoch_order) // batch_size) * updates_per_sequence
    if len(observed) != updates_per_epoch:
        raise AssertionError("count schedule does not match complete optimizer updates")
    retained = (len(epoch_order) // batch_size) * batch_size
    retained_event_total = int(sequence_counts[np.asarray(epoch_order[:retained])].sum())
    if sum(observed) != retained_event_total:
        raise AssertionError("per-update counts do not sum to the retained sampler rows")
    if retained_event_total <= 0:
        raise ValueError("first fit epoch has no endpoint events")

    count_payload = np.asarray(sequence_counts, dtype="<i8").tobytes(order="C")
    schedules = {}
    for mode in ("constant", "aligned", "shuffled"):
        schedule = make_event_ema_schedule(
            observed,
            base_decay=base_decay,
            updates_per_epoch=updates_per_epoch,
            mode=mode,
            seed=shuffle_seed,
            epoch=0,
        )
        product = float(np.prod(schedule.decays, dtype=np.float64))
        expected_product = base_decay**updates_per_epoch
        if not np.isclose(product, expected_product, rtol=2e-10, atol=0.0):
            raise AssertionError(f"{mode} EMA product does not match constant retention")
        schedules[mode] = {
            "observed_counts": list(schedule.observed_counts),
            "applied_counts": list(schedule.applied_counts),
            "decays": list(schedule.decays),
            "count_sha256": schedule.count_sha256,
            "total_events": schedule.total_events,
            "decay_product": product,
        }
    aligned = schedules["aligned"]["observed_counts"]
    if sorted(schedules["shuffled"]["applied_counts"]) != sorted(aligned):
        raise AssertionError("shuffled arm changed the observed per-update count multiset")

    return {
        "schema": "e426-fold0-event-count-audit-v1",
        "purpose": "CPU input-only count/schedule audit; not a model fit or prediction",
        "manifest_path": str(manifest_path.relative_to(root)),
        "manifest_sha256": manifest_sha256,
        "parquet_source_identities": {
            name: manifest["sources"][name]
            for name in ("datasets/train.parquet", "datasets/valid.parquet")
        },
        "fit_membership": {
            "train_count": len(train_groups),
            "valid_count": len(valid_groups),
            "evaluation_count": len(split.evaluation_groups),
            "corrupt_train_528_excluded": 528 not in train_groups,
            "eligible_train_529_included": 529 in train_groups,
        },
        "forbidden_528_negative_check": {
            "rejected_before_count_source": True,
            "error": rejection,
        },
        "ordered_fit_rows": [[source, group] for source, group in source_rows],
        "ordered_fit_rows_sha256": _sha256_json_rows(source_rows),
        "sequence_event_counts_shape": list(sequence_counts.shape),
        "sequence_event_counts_sha256": hashlib.sha256(count_payload).hexdigest(),
        "sequence_event_counts_total": int(sequence_counts.sum()),
        "sampler": {
            "seed": seed,
            "epoch": 0,
            "batch_size": batch_size,
            "drop_last": True,
            "source_rows": len(source_rows),
            "retained_rows": retained,
            "dropped_rows": len(epoch_order) - retained,
            "updates_per_sequence": updates_per_sequence,
            "updates_per_epoch": updates_per_epoch,
            "retained_event_total": retained_event_total,
        },
        "event_definition": {
            "column": "a6",
            "float32_marker": -5.199337482452393,
            "update_rows": update_rows,
            "sequence_rows": SEQUENCE_LENGTH,
            "target_or_scored_columns_read": False,
            "streaming": "one Parquet row group at a time",
        },
        "schedules": schedules,
    }


def main() -> None:
    """Run and write the bounded CPU audit, refusing to overwrite evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="repository data root")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("research/campaign_2026-09-29/artifacts/splits/fold0.json"),
        help="fold-0 manifest relative to --root unless absolute",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/campaign_2026-10-04/checks/e426_event_count_audit_f0_s3.json"),
        help="new audit artifact path relative to --root unless absolute",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = args.manifest if args.manifest.is_absolute() else root / args.manifest
    output = args.output if args.output.is_absolute() else root / args.output
    if output.exists():
        raise FileExistsError(f"refusing to overwrite event-count audit: {output}")
    result = audit(root=root, manifest_path=manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    sys.stdout.write(
        json.dumps(
            {key: value for key, value in result.items() if key != "ordered_fit_rows"},
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
