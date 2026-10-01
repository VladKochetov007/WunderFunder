"""Clean outer folds exclude every connected fitting sequence, including cycles."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from wnn.splits import connected_components, load_campaign_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_campaign_splits import build_manifests  # noqa: E402


@pytest.fixture
def campaign_root(tmp_path: Path) -> Path:
    """Supply a linked train/valid pair and a validation successor cycle."""
    (tmp_path / "cache").mkdir()
    (tmp_path / "datasets").mkdir()
    splits = np.array(["train"] * 4 + ["valid"] * 5)
    groups = np.array([0, 1, 2, 3, 0, 1, 2, 3, 4])
    successors = np.array([4, -1, 5, -1, -1, -1, 7, 6, -1])
    folds = np.array([0, 1, 1, 3, 0, 1, 2, 2, 3])
    np.savez(tmp_path / "cache/chains.npz", split=splits, row_group=groups, next=successors)
    np.savez(tmp_path / "cache/chain_folds.npz", split=splits, row_group=groups, fold=folds)
    for split, count in (("train", 4), ("valid", 5)):
        ids = np.arange(count) + (100 if split == "valid" else 0)
        np.savez(tmp_path / f"cache/{split}_targets.npz", row_group=np.arange(count), seq_ix=ids)
        pq.write_table(pa.table({"seq_ix": ids}), tmp_path / f"datasets/{split}.parquet")
    return tmp_path


def test_connected_components_recover_cycles_and_direction() -> None:
    components = connected_components(np.array([1, 0, 1, -1, 5, 4]))
    assert components.tolist() == [0, 0, 0, 1, 2, 2]
    with pytest.raises(ValueError, match="outside"):
        connected_components(np.array([2, -1]))


def test_manifest_excludes_linked_train_groups_and_keeps_fold_assignment(
    campaign_root: Path,
) -> None:
    paths = build_manifests(campaign_root, campaign_root / "splits")
    folds = [load_campaign_split(path, campaign_root) for path in paths]
    assert folds[0].evaluation_groups == (0,)
    assert folds[0].train_groups == (1, 2, 3)
    assert folds[1].train_groups == (0, 1, 3)
    assert folds[2].evaluation_groups == (2, 3)
    assert folds[3].evaluation_groups == (4,)
    assert set().union(*(set(fold.evaluation_groups) for fold in folds)) == set(range(5))
    for fold in folds:
        fold.assert_fit_groups(fold.train_groups, fold.valid_groups)
    with pytest.raises(ValueError, match="outside"):
        folds[0].assert_fit_groups([0], [])
    before = paths[0].stat().st_mtime_ns
    assert build_manifests(campaign_root, campaign_root / "splits") == paths
    assert paths[0].stat().st_mtime_ns == before


def test_manifest_rejects_component_leakage_and_source_changes(campaign_root: Path) -> None:
    path = build_manifests(campaign_root, campaign_root / "splits")[0]
    original = path.read_text()
    manifest = json.loads(original)
    manifest["fit"]["train"].append(0)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="component overlap"):
        load_campaign_split(path, campaign_root)
    path.write_text(original)
    with (campaign_root / "datasets/train.parquet").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="source changed"):
        load_campaign_split(path, campaign_root)


def test_builder_rejects_a_component_crossing_validation_folds(campaign_root: Path) -> None:
    path = campaign_root / "cache/chain_folds.npz"
    with np.load(path) as data:
        arrays = {name: data[name] for name in data.files}
    arrays["fold"][7] = 3
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="crosses cached folds"):
        build_manifests(campaign_root, campaign_root / "splits")
