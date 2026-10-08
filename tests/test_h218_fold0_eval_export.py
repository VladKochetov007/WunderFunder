"""Focused membership and provenance checks for the H218 fold-0 exporter."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

EXPORTER_PATH = (
    Path(__file__).resolve().parents[1]
    / "research/campaign_2026-10-06/bandit_continuation/attention_training_H218/export_fold0_eval.py"
)


def load_exporter() -> ModuleType:
    """Load the lightweight exporter module without importing model libraries."""
    spec = importlib.util.spec_from_file_location("h218_fold0_export", EXPORTER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_manifest(path: Path, evaluation_groups: list[int] | None = None) -> dict[str, object]:
    """Write a compact fixture with the campaign's exact H218 group counts."""
    if evaluation_groups is None:
        evaluation_groups = list(range(1_405, 1_873))
    manifest: dict[str, object] = {
        "version": 1,
        "fold": 0,
        "n_folds": 4,
        "fit": {"train": list(range(10_535)), "valid": list(range(1_405))},
        "evaluation": {"valid": evaluation_groups},
    }
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def test_loads_only_sorted_fold0_evaluation_groups(tmp_path: Path) -> None:
    exporter = load_exporter()
    path = tmp_path / "fold0.json"
    write_manifest(path)
    manifest_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    groups = list(range(1_405, 1_873))
    groups_sha256 = exporter.evaluation_group_digest(groups)

    manifest, selected, selected_sha256 = exporter.load_fold0_evaluation_groups(
        path, manifest_sha256, groups_sha256
    )

    assert manifest["fold"] == 0
    assert selected == groups
    assert selected_sha256 == groups_sha256


def test_rejects_changed_manifest_and_unsorted_membership(tmp_path: Path) -> None:
    exporter = load_exporter()
    path = tmp_path / "fold0.json"
    groups = list(range(1_405, 1_873))
    write_manifest(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    group_digest = exporter.evaluation_group_digest(groups)
    with pytest.raises(ValueError, match="manifest hash"):
        exporter.load_fold0_evaluation_groups(path, "0" * 64, group_digest)

    write_manifest(path, list(reversed(groups)))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="sorted 468-row fold"):
        exporter.load_fold0_evaluation_groups(path, digest, group_digest)


def test_rejects_evaluation_overlap_with_fit_valid(tmp_path: Path) -> None:
    exporter = load_exporter()
    path = tmp_path / "fold0.json"
    groups = list(range(1_404, 1_872))
    write_manifest(path, groups)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="appears in the fit-valid"):
        exporter.load_fold0_evaluation_groups(
            path, digest, exporter.evaluation_group_digest(groups)
        )


def test_valid_source_identity_uses_manifest_size_and_mtime(tmp_path: Path) -> None:
    exporter = load_exporter()
    source = tmp_path / "valid.parquet"
    source.write_bytes(b"bounded-identity-fixture")
    stat = source.stat()
    manifest = {
        "sources": {"datasets/valid.parquet": {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}}
    }
    exporter.validate_valid_parquet_identity(manifest, source)

    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="file identity"):
        exporter.validate_valid_parquet_identity(manifest, source)


def test_source_lock_binds_exporter_and_all_declared_files(tmp_path: Path) -> None:
    exporter = load_exporter()
    root = Path(__file__).resolve().parents[1]
    exporter.ROOT = root
    relative_path = EXPORTER_PATH.relative_to(root).as_posix()
    source_lock = tmp_path / "source-lock.json"
    lock = {"files": {relative_path: exporter.file_sha256(EXPORTER_PATH)}}
    source_lock.write_text(json.dumps(lock), encoding="utf-8")
    lock_sha256 = hashlib.sha256(source_lock.read_bytes()).hexdigest()

    verified = exporter.validate_source_lock(source_lock, lock_sha256)

    assert verified == lock["files"]
    lock["files"][relative_path] = "0" * 64
    source_lock.write_text(json.dumps(lock), encoding="utf-8")
    lock_sha256 = hashlib.sha256(source_lock.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="source changed"):
        exporter.validate_source_lock(source_lock, lock_sha256)


def test_checkpoint_membership_must_match_the_clean_fold(tmp_path: Path) -> None:
    exporter = load_exporter()
    manifest_path = tmp_path / "fold0.json"
    manifest = write_manifest(manifest_path)
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    exporter.SPLIT_PATH = manifest_path
    exporter.ROOT = tmp_path
    exporter.RUN_NAME = "fixture"
    checkpoint_path = tmp_path / "artifacts/runs/fixture/model_ep3.pt"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_bytes(b"fixture-checkpoint")
    checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    checkpoint = {
        "epoch": 3,
        "args": {
            "arch": "block-memory",
            "hidden": 128,
            "layers": 2,
            "block_heads": 4,
            "block_rows": 256,
            "batch": 4,
            "chunk": 2000,
            "seed": 7,
            "train_source": "train+valid",
            "valid_fold": 0,
            "epochs": 4,
            "checkpoint_policy": "fixed",
            "checkpoint_epoch": 3,
            "no_eval_during_fit": True,
            "privileged_mode": "none",
            "init": None,
        },
        "provenance": {
            "split_manifest_sha256": manifest_sha256,
            "fit_row_groups": {
                "train": list(manifest["fit"]["train"]),
                "valid": list(manifest["fit"]["valid"]),
            },
        },
    }

    exporter.validate_checkpoint(checkpoint, manifest, checkpoint_sha256)
    checkpoint["provenance"]["fit_row_groups"]["valid"].append(1_405)
    with pytest.raises(ValueError, match="fit membership differs"):
        exporter.validate_checkpoint(checkpoint, manifest, checkpoint_sha256)
