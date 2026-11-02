from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from wnn.distillation_cache import (
    CACHE_FORMAT,
    CACHE_VERSION,
    MAX_CACHE_METADATA_BYTES,
    row_groups_sha256,
    validate_distillation_cache,
)
from wnn.final_submit_manifest import FinalFitMembership
from wnn.splits import CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA, CampaignSplit
from wnn.training.data_membership import select_membership
from wnn.training.final_submit_bridge import (
    SOURCE_VALIDATION_SHA256,
    final_fit_identity,
    validate_final_fit_artifact,
    validate_final_fit_training_options,
    validate_source_validation_receipt,
)
from wnn.training.final_submit_bridge_cache import (
    MAX_FINAL_CACHE_METADATA_BYTES,
    _read_final_cache_metadata,
    validate_final_distillation_cache,
)
from wnn.training.setup_lineage import require_chronological_primary_training

TRAIN_SHA256 = "a" * 64
VALID_SHA256 = "b" * 64
MANIFEST_SHA256 = "c" * 64
MEMBERS_SHA256 = "d" * 64


def _membership() -> FinalFitMembership:
    """Create the exact final-fit membership for bounded provenance checks."""
    return FinalFitMembership(
        path=Path("final-fit-membership.json"),
        manifest_sha256=MANIFEST_SHA256,
        fit_members_sha256=MEMBERS_SHA256,
        train_groups=tuple(range(10_607)),
        valid_groups=tuple(range(1_873)),
        train_sha256=TRAIN_SHA256,
        valid_sha256=VALID_SHA256,
        source_validation_sha256=SOURCE_VALIDATION_SHA256,
    )


def _final_args() -> argparse.Namespace:
    """Return one complete, fixed, no-evaluation final-fit CLI contract."""
    return argparse.Namespace(
        split_manifest=None,
        train_source="train+valid",
        train_seqs=10_607,
        include_heldout_valid=True,
        valid_train_fraction=1.0,
        valid_repeat=1,
        valid_fold=-1,
        no_eval_during_fit=True,
        checkpoint_policy="fixed",
        checkpoint_epoch=3,
        epochs=4,
        stop_after_epoch=None,
        resume_state=None,
        fit_subset_manifest=None,
        fit_groups_manifest=None,
        class_balance=False,
        train_classes="",
        train_eval_seqs=0,
        mask_weight=False,
        propensity="",
        prop_input="",
        prop_cache=False,
        privileged_mode="none",
    )


def test_final_fit_options_require_complete_sources_fixed_endpoint_and_no_eval() -> None:
    validate_final_fit_training_options(_final_args())
    invalid = _final_args()
    invalid.no_eval_during_fit = False
    with pytest.raises(ValueError, match="no evaluation"):
        validate_final_fit_training_options(invalid)


def test_no_final_flag_still_uses_the_chronological_primary_guard() -> None:
    split = CampaignSplit(
        path=Path("primary.json"),
        sha256=MANIFEST_SHA256,
        fold=0,
        n_folds=1,
        train_groups=(1,),
        valid_groups=(0,),
        evaluation_groups=(),
        components=(),
        schema=CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA,
    )
    parser = argparse.ArgumentParser()
    with pytest.raises(SystemExit):
        require_chronological_primary_training(parser, _final_args(), split)


def test_final_membership_selects_exact_12480_rows_including_group_528_and_no_eval() -> None:
    args = SimpleNamespace(
        train_source="train+valid",
        include_heldout_valid=True,
        valid_train_fraction=1.0,
        valid_repeat=1,
        valid_fold=-1,
        train_seqs=10_607,
        seed=17,
        train_classes="",
        class_balance=False,
        eval_source="dev",
        full_valid=False,
        valid_seqs=600,
        epoch_samples=0,
    )
    data = SimpleNamespace(
        campaign_split=None,
        final_fit_membership=_membership(),
        eligible_valid_count=None,
        extra_valid=None,
        fit_subset=None,
        inner_crossfit=None,
        n_valid=None,
        sampler=None,
        train_idx=None,
        train_split=None,
        valid_idx=None,
    )
    run = SimpleNamespace(
        environment=SimpleNamespace(root=Path(".")), options=SimpleNamespace(args=args), data=data
    )

    select_membership(run)

    assert data.train_split == "train"
    assert len(data.train_idx) == 10_607
    assert set(data.train_idx.tolist()) == set(range(10_607))
    assert 528 in data.train_idx
    assert data.extra_valid.tolist() == list(range(1_873))
    assert data.valid_idx.size == 0
    assert data.n_valid == 0


def test_final_artifact_rejects_old_checkpoint_and_any_evaluation_members() -> None:
    membership = _membership()
    provenance = {
        "final_fit": final_fit_identity(membership),
        "fit_row_groups": {
            "train": list(membership.train_groups),
            "valid": list(membership.valid_groups),
        },
        "evaluation_row_groups": [],
    }
    args = {
        "final_fit_manifest_sha256": membership.manifest_sha256,
        "final_fit_members_sha256": membership.fit_members_sha256,
        "final_fit_train_sha256": membership.train_sha256,
        "final_fit_valid_sha256": membership.valid_sha256,
        "final_fit_source_validation_sha256": membership.source_validation_sha256,
        "split_manifest": None,
        "split_manifest_sha256": None,
        "train_source": "train+valid",
        "include_heldout_valid": True,
        "no_eval_during_fit": True,
        "checkpoint_policy": "fixed",
        "checkpoint_epoch": 3,
    }

    validate_final_fit_artifact(args, provenance, membership, checkpoint_epoch=3)
    with pytest.raises(ValueError, match="arguments do not bind"):
        validate_final_fit_artifact({"train_source": "train+valid"}, provenance, membership)
    exposed = {**provenance, "evaluation_row_groups": [0]}
    with pytest.raises(ValueError, match="evaluation membership"):
        validate_final_fit_artifact(args, exposed, membership)


def test_receiver_receipt_binds_exact_sources_and_decoded_group_528() -> None:
    receipt_path = (
        Path(__file__).resolve().parents[1]
        / "research/experiment_drafts/final-submit-all-labeled-20261031/receiving/SOURCE_VALIDATION_R1.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    membership = FinalFitMembership(
        path=Path("final-fit-membership.json"),
        manifest_sha256=MANIFEST_SHA256,
        fit_members_sha256=MEMBERS_SHA256,
        train_groups=tuple(range(receipt["sources"]["train"]["row_groups"])),
        valid_groups=tuple(range(receipt["sources"]["valid"]["row_groups"])),
        train_sha256=receipt["sources"]["train"]["file_sha256"],
        valid_sha256=receipt["sources"]["valid"]["file_sha256"],
        source_validation_sha256=SOURCE_VALIDATION_SHA256,
    )

    validate_source_validation_receipt(receipt_path, SOURCE_VALIDATION_SHA256, membership)


def test_historical_cache_manifest_cannot_be_relabelled_as_final_fit(tmp_path: Path) -> None:
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "format": CACHE_FORMAT,
                "version": CACHE_VERSION,
                "split_manifest_sha256": MANIFEST_SHA256,
                "fold": 0,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="no-evaluation final fit"):
        validate_final_distillation_cache(tmp_path, _membership(), "source", require_families=True)


def test_final_cache_reader_accepts_repeated_12480_group_metadata_only_on_final_path(
    tmp_path: Path,
) -> None:
    membership = _membership()
    groups = {
        "train": list(membership.train_groups),
        "valid": list(membership.valid_groups),
    }
    files = {
        f"{target}_{split}": {
            "path": f"{target}_{split}.npy",
            "row_groups": split_groups,
            "row_groups_sha256": row_groups_sha256(split_groups),
            "shape": [len(split_groups), 20_000, 8 if target == "families" else 2],
            "dtype": "float32",
            "sha256": "e" * 64,
            "bytes": len(split_groups) * 20_000 * 4,
        }
        for target in ("main_source", "main_normalized", "main_homogeneous", "families")
        for split, split_groups in groups.items()
    }
    metadata = {
        "format": CACHE_FORMAT,
        "version": CACHE_VERSION,
        "fit_groups": groups,
        "final_fit": final_fit_identity(membership),
        "evaluation_valid_groups": [],
        "files": files,
    }
    raw = (json.dumps(metadata, indent=2) + "\n").encode()
    path = tmp_path / "metadata.json"
    path.write_bytes(raw)

    parsed, parsed_raw = _read_final_cache_metadata(path)

    assert len(groups["train"]) + len(groups["valid"]) == 12_480
    assert len(raw) > MAX_CACHE_METADATA_BYTES == 512_000
    assert len(raw) <= MAX_FINAL_CACHE_METADATA_BYTES
    assert parsed["fit_groups"] == groups
    assert parsed_raw == raw
    research_split = CampaignSplit(
        path=Path("research-split.json"),
        sha256=MANIFEST_SHA256,
        fold=0,
        n_folds=1,
        train_groups=(),
        valid_groups=(),
        evaluation_groups=(),
        components=(),
        schema=CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA,
    )
    with pytest.raises(ValueError, match="512000-byte read bound"):
        validate_distillation_cache(
            tmp_path, tmp_path, research_split, "source", require_families=False
        )
