"""Explicit final-fit admission and provenance checks, separate from research splits."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping
from pathlib import Path

from wnn.final_submit_manifest import (
    FINAL_FIT_POLICY,
    FINAL_FIT_SCHEMA,
    FinalFitMembership,
    load_final_fit_membership,
)
from wnn.splits import file_sha256

SOURCE_VALIDATION_SCHEMA = "wnn.final-submit-source-validation.v1"
SOURCE_VALIDATION_SHA256 = "1e8a8a794a0b8d9677a76030e570681074503d38aa2d54b80e36067566b94ce0"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def load_final_fit_arguments(
    parser: argparse.ArgumentParser, args: argparse.Namespace, root: Path
) -> FinalFitMembership | None:
    """Load the explicit final-fit receipt and attach its content identities to args."""
    path = args.final_fit_manifest
    receipt_path = args.final_fit_source_validation_receipt
    receipt_sha256 = args.final_fit_source_validation_sha256
    if path is None:
        if receipt_sha256 is not None or receipt_path is not None:
            parser.error("final-fit source receipt options require --final-fit-manifest")
        return None
    if args.split_manifest is not None:
        parser.error("--final-fit-manifest and --split-manifest are mutually exclusive")
    if not isinstance(receipt_sha256, str) or receipt_path is None:
        parser.error("--final-fit-manifest requires both source-validation receipt path and SHA256")
    try:
        membership = load_final_fit_membership(path, root, receipt_sha256)
        validate_source_validation_receipt(receipt_path, receipt_sha256, membership)
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    args.final_fit_manifest = str(membership.path)
    args.final_fit_manifest_sha256 = membership.manifest_sha256
    args.final_fit_members_sha256 = membership.fit_members_sha256
    args.final_fit_train_sha256 = membership.train_sha256
    args.final_fit_valid_sha256 = membership.valid_sha256
    args.final_fit_source_validation_sha256 = membership.source_validation_sha256
    args.final_fit_source_validation_receipt = str(receipt_path.resolve())
    return membership


def validate_source_validation_receipt(
    path: Path, expected_sha256: str, membership: FinalFitMembership
) -> None:
    """Require the pinned receiver receipt to bind the same sources and row-group proof."""
    if path.is_symlink() or not path.is_file() or file_sha256(path) != expected_sha256:
        raise ValueError("source-validation receipt bytes differ from the supplied identity")
    if expected_sha256 != SOURCE_VALIDATION_SHA256:
        raise ValueError("final-fit mode requires the reviewed receiver source-validation receipt")
    size = path.stat().st_size
    if size <= 0 or size > 64_000:
        raise ValueError("source-validation receipt exceeds its bounded read size")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict) or receipt.get("schema") != SOURCE_VALIDATION_SCHEMA:
        raise ValueError("unsupported final-fit source-validation receipt")
    sources = receipt.get("sources")
    row_group = receipt.get("row_group_528")
    if (
        receipt.get("fitting_member_count") != 12_480
        or not isinstance(sources, dict)
        or not isinstance(row_group, dict)
        or sources.get("train", {}).get("file_sha256") != membership.train_sha256
        or sources.get("valid", {}).get("file_sha256") != membership.valid_sha256
        or sources.get("train", {}).get("row_groups") != len(membership.train_groups)
        or sources.get("valid", {}).get("row_groups") != len(membership.valid_groups)
        or row_group.get("source") != "train"
        or row_group.get("row_group") != 528
        or row_group.get("rows") != 20_000
        or row_group.get("columns") != 117
        or row_group.get("nonfinite") != 0
    ):
        raise ValueError("source-validation receipt does not prove this exact labeled data set")


def validate_final_fit_training_options(args: argparse.Namespace) -> None:
    """Require complete labeled-data fitting, fixed endpoints and no scoring path."""
    forbidden = (
        args.split_manifest is not None
        or args.train_source != "train+valid"
        or args.train_seqs != 10_607
        or not args.include_heldout_valid
        or args.valid_train_fraction != 1
        or args.valid_repeat != 1
        or args.valid_fold >= 0
        or not args.no_eval_during_fit
        or args.checkpoint_policy != "fixed"
        or args.checkpoint_epoch is None
        or not 0 <= args.checkpoint_epoch < args.epochs
        or args.stop_after_epoch is not None
        or args.resume_state is not None
        or args.fit_subset_manifest is not None
        or args.fit_groups_manifest is not None
        or args.class_balance
        or bool(args.train_classes)
        or args.train_eval_seqs != 0
        or args.mask_weight
        or bool(args.propensity)
        or bool(args.prop_input)
        or args.prop_cache
        or args.privileged_mode != "none"
    )
    if forbidden:
        raise ValueError(
            "final-fit mode requires all train and valid labels, no learned weighting or ancestry leakage, no evaluation, and a fixed configured endpoint"
        )


def final_fit_identity(membership: FinalFitMembership) -> dict[str, object]:
    """Return the portable identity every final-fit descendant must preserve."""
    return {
        "schema": FINAL_FIT_SCHEMA,
        "policy": FINAL_FIT_POLICY,
        "manifest_sha256": membership.manifest_sha256,
        "fit_members_sha256": membership.fit_members_sha256,
        "source_sha256": {
            "train": membership.train_sha256,
            "valid": membership.valid_sha256,
        },
        "source_validation_sha256": membership.source_validation_sha256,
        "fit_row_groups": {
            "train": list(membership.train_groups),
            "valid": list(membership.valid_groups),
        },
        "independent_evaluation_members": [],
        "independent_score_claim": "forbidden",
        "post_fit_primary_valid_scoring": "forbidden",
    }


def validate_final_fit_artifact(
    args: Mapping[str, object],
    provenance: Mapping[str, object],
    membership: FinalFitMembership,
    *,
    checkpoint_epoch: int | None = None,
) -> None:
    """Reject checkpoints without exact final-fit membership, no-eval and data lineage."""
    validate_final_fit_provenance(provenance, membership)
    if (
        args.get("final_fit_manifest_sha256") != membership.manifest_sha256
        or args.get("final_fit_members_sha256") != membership.fit_members_sha256
        or args.get("final_fit_train_sha256") != membership.train_sha256
        or args.get("final_fit_valid_sha256") != membership.valid_sha256
        or args.get("final_fit_source_validation_sha256") != membership.source_validation_sha256
        or args.get("split_manifest") is not None
        or args.get("split_manifest_sha256") is not None
        or args.get("train_source") != "train+valid"
        or args.get("include_heldout_valid") is not True
        or args.get("no_eval_during_fit") is not True
        or args.get("checkpoint_policy") != "fixed"
        or type(args.get("checkpoint_epoch")) is not int
        or (checkpoint_epoch is not None and args.get("checkpoint_epoch") != checkpoint_epoch)
    ):
        raise ValueError("checkpoint arguments do not bind the required final-fit identity")


def validate_final_fit_provenance(
    provenance: Mapping[str, object], membership: FinalFitMembership
) -> None:
    """Require exact source fit membership and an empty evaluation set in provenance."""
    fit = provenance.get("fit_row_groups")
    evaluation = provenance.get("evaluation_row_groups")
    expected_fit = {
        "train": list(membership.train_groups),
        "valid": list(membership.valid_groups),
    }
    if provenance.get("final_fit") != final_fit_identity(membership):
        raise ValueError("checkpoint provenance belongs to a different final-fit source identity")
    if not isinstance(fit, Mapping):
        raise ValueError("final-fit checkpoint omits source-qualified fitting membership")
    for source in ("train", "valid"):
        groups = fit.get(source)
        expected = expected_fit[source]
        if (
            not isinstance(groups, list)
            or any(type(group) is not int or group < 0 for group in groups)
            or len(groups) != len(set(groups))
            or set(groups) != set(expected)
        ):
            raise ValueError(f"final-fit checkpoint has incomplete {source} fitting membership")
    if evaluation != []:
        raise ValueError(
            "final-fit checkpoint has evaluation membership and cannot claim independent scores"
        )


def validate_final_fit_parent(
    checkpoint: Mapping[str, object], membership: FinalFitMembership
) -> None:
    """Require a final-fit checkpoint parent with the exact same source and fixed epoch."""
    args = checkpoint.get("args")
    provenance = checkpoint.get("provenance")
    epoch = checkpoint.get("epoch")
    if (
        not isinstance(args, Mapping)
        or not isinstance(provenance, Mapping)
        or type(epoch) is not int
    ):
        raise ValueError("final-fit parent omits checkpoint or provenance fields")
    validate_final_fit_artifact(args, provenance, membership, checkpoint_epoch=epoch)
