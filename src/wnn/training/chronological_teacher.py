"""Admission checks for fixed checkpoints used as chronological teachers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath

import numpy as np

from wnn.splits import CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA, CampaignSplit

_CHECKPOINT_ARGUMENTS = (
    "checkpoint_policy",
    "checkpoint_epoch",
    "no_eval_during_fit",
    "train_source",
    "split_manifest_sha256",
)


def _members(value: object, name: str) -> set[int]:
    """Parse a recorded row-group list without accepting bools or duplicates."""
    if not isinstance(value, (list, tuple)) or any(
        type(item) is not int or item < 0 for item in value
    ):
        raise ValueError(f"chronological teacher {name} must be nonnegative integer row groups")
    if len(value) != len(set(value)):
        raise ValueError(f"chronological teacher {name} repeats row groups")
    return set(value)


def validate_chronological_teacher_membership(
    provenance: Mapping[str, object], split: CampaignSplit, name: str
) -> None:
    """Require the entire official train fit set, no valid fit rows, and full valid evaluation."""
    if split.schema != CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA:
        raise ValueError("chronological teacher admission requires the official primary split")
    if provenance.get("split_manifest_sha256") != split.sha256:
        raise ValueError(f"teacher {name} uses a different chronological split manifest")
    fitting = provenance.get("fit_row_groups")
    if not isinstance(fitting, Mapping):
        raise ValueError(f"teacher {name} has no recorded fitting membership")
    train = _members(fitting.get("train"), "fit train membership")
    valid = _members(fitting.get("valid"), "fit valid membership")
    evaluation = _members(provenance.get("evaluation_row_groups"), "evaluation membership")
    if train != set(split.train_groups):
        raise ValueError(f"teacher {name} must fit on the exact chronological train membership")
    if valid:
        raise ValueError(f"teacher {name} must not fit on official validation sequences")
    if evaluation != set(split.evaluation_groups):
        raise ValueError(
            f"teacher {name} must evaluate the complete official validation membership"
        )
    split.assert_fit_groups(train, valid)


def validate_chronological_teacher_checkpoint(
    checkpoint: Mapping[str, object],
    run_provenance: Mapping[str, object],
    split: CampaignSplit,
    checkpoint_file: str,
    checkpoint_sha256: str,
    name: str,
) -> None:
    """Bind a selected checkpoint to its exact chronological fit and fixed epoch receipt."""
    if split.schema != CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA:
        raise ValueError("chronological teacher checkpoint requires the official primary split")
    if PurePosixPath(checkpoint_file).name != checkpoint_file or checkpoint_file not in (
        "model.pt",
        f"model_ep{checkpoint.get('epoch')}.pt",
    ):
        raise ValueError(f"teacher {name} has an unsupported checkpoint filename")
    checkpoint_args = checkpoint.get("args")
    checkpoint_provenance = checkpoint.get("provenance")
    run_args = run_provenance.get("args")
    hashes = run_provenance.get("checkpoint_sha256")
    if not all(
        isinstance(value, Mapping)
        for value in (checkpoint_args, checkpoint_provenance, run_args, hashes)
    ):
        raise ValueError(f"teacher {name} lacks checkpoint or run provenance")
    epoch = checkpoint.get("epoch")
    if type(epoch) is not int:
        raise ValueError(f"teacher {name} checkpoint epoch is invalid")
    expected_args = {
        "checkpoint_policy": "fixed",
        "checkpoint_epoch": epoch,
        "no_eval_during_fit": False,
        "train_source": "train",
        "split_manifest_sha256": split.sha256,
    }
    for key, expected in expected_args.items():
        if checkpoint_args.get(key) != expected or run_args.get(key) != expected:
            raise ValueError(f"teacher {name} has incompatible fixed chronological argument {key}")
    if any(checkpoint_args.get(key) != run_args.get(key) for key in _CHECKPOINT_ARGUMENTS):
        raise ValueError(f"teacher {name} checkpoint arguments differ from run provenance")
    if run_provenance.get("split_manifest_sha256") != split.sha256:
        raise ValueError(f"teacher {name} run provenance uses a different split manifest")
    if hashes.get(checkpoint_file) != checkpoint_sha256:
        raise ValueError(f"teacher {name} checkpoint hash differs from run provenance")
    if checkpoint_file == "model.pt" and run_provenance.get("selected_epoch") != epoch:
        raise ValueError(f"teacher {name} selected checkpoint epoch differs from provenance")
    validate_chronological_teacher_membership(checkpoint_provenance, split, name)
    validate_chronological_teacher_membership(run_provenance, split, name)


def validate_chronological_train_index(
    row_groups: Sequence[int], split: CampaignSplit, name: str = "train"
) -> None:
    """Require source-index order to cover every training-file sequence exactly once."""
    if split.schema != CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA:
        raise ValueError("full train index export requires the official chronological split")
    groups = _members(list(row_groups), "full train source index")
    expected = {group for source, group, _ in split.components if source == "train"}
    if not groups or groups != expected:
        raise ValueError(f"{name} target index does not cover the full official training source")


def validate_chronological_train_index_order(
    data_root: Path, row_groups: Sequence[int], split: CampaignSplit, name: str
) -> None:
    """Require the private legacy target index to preserve the manifest-pinned source order."""
    validate_chronological_train_index(row_groups, split, name)
    with np.load(data_root / "cache/train_index.npz", allow_pickle=False) as source_index:
        raw_source_groups = np.asarray(source_index["row_group"])
    if raw_source_groups.ndim != 1 or raw_source_groups.dtype.kind not in "iu":
        raise ValueError(
            "verified chronological train index must be a one-dimensional integer array"
        )
    source_groups = raw_source_groups.astype(np.int64, copy=False)
    if not np.array_equal(np.asarray(row_groups, dtype=np.int64), source_groups):
        raise ValueError(
            f"{name} target index order differs from the frozen chronological source index"
        )


def select_chronological_full_train_index(
    data_root: Path,
    split: CampaignSplit,
    checkpoint: Mapping[str, object],
    checkpoint_file: str,
    checkpoint_sha256: str,
    run_provenance: Mapping[str, object],
    name: str,
) -> np.ndarray:
    """Load the full row-group index after proving the fixed train-only teacher receipt."""
    validate_chronological_teacher_checkpoint(
        checkpoint,
        run_provenance,
        split,
        checkpoint_file,
        checkpoint_sha256,
        name,
    )
    with np.load(data_root / "cache/train_targets.npz", allow_pickle=False) as targets:
        raw_groups = np.asarray(targets["row_group"])
    if raw_groups.ndim != 1 or raw_groups.dtype.kind not in "iu":
        raise ValueError("full train target index must be a one-dimensional integer array")
    groups = raw_groups.astype(np.int64, copy=False)
    validate_chronological_train_index_order(data_root, groups.tolist(), split, name)
    return groups
