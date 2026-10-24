"""Inner sequence splits exclude complete source-qualified graph components."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from wnn.sequence_inner_splits import prepare_sequence_inner_split
from wnn.splits import load_campaign_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_campaign_splits import build_manifests  # noqa: E402


@pytest.fixture
def campaign_root(tmp_path: Path) -> Path:
    """Provide linked train/valid sequences and a validation successor cycle."""
    (tmp_path / "cache").mkdir()
    (tmp_path / "datasets").mkdir()
    splits = np.array(["train"] * 4 + ["valid"] * 5)
    groups = np.array([0, 1, 2, 3, 0, 1, 2, 3, 4])
    successors = np.array([4, -1, 5, -1, -1, -1, 7, 6, -1])
    folds = np.array([0, 1, 1, 3, 0, 1, 2, 2, 3])
    np.savez(tmp_path / "cache/chains.npz", split=splits, row_group=groups, next=successors)
    np.savez(tmp_path / "cache/chain_folds.npz", split=splits, row_group=groups, fold=folds)
    for source, count in (("train", 4), ("valid", 5)):
        seq_ix = np.arange(count) + (100 if source == "valid" else 0)
        np.savez(
            tmp_path / f"cache/{source}_targets.npz",
            row_group=np.arange(count),
            seq_ix=seq_ix,
        )
        pq.write_table(pa.table({"seq_ix": seq_ix}), tmp_path / f"datasets/{source}.parquet")
    return tmp_path


def _outer_manifest(root: Path) -> Path:
    """Build and return one frozen canonical outer-fold manifest."""
    return build_manifests(root, root / "outer")[0]


def test_inner_split_excludes_outer_and_selected_components_and_is_deterministic(
    campaign_root: Path,
) -> None:
    parent_path = _outer_manifest(campaign_root)
    parent = load_campaign_split(parent_path, campaign_root)
    output_dir = campaign_root / "inner"
    output_dir.mkdir()
    output = output_dir / "inner.json"
    inner = prepare_sequence_inner_split(parent_path, campaign_root, output, 41001, 0.4)
    loaded = load_campaign_split(output, campaign_root)
    repeated = prepare_sequence_inner_split(
        parent_path, campaign_root, output_dir / "repeat.json", 41001, 0.4
    )
    assert output.read_bytes() == (output_dir / "repeat.json").read_bytes()
    assert loaded == inner
    assert inner.sha256 == repeated.sha256
    assert inner.sha256 == hashlib.sha256(output.read_bytes()).hexdigest()
    assert inner.fold == parent.fold and inner.n_folds == parent.n_folds
    assert set(inner.evaluation_groups) < set(parent.valid_groups)
    parent_manifest = json.loads(parent_path.read_text())
    inner_manifest = json.loads(output.read_text())
    assert inner_manifest["sources"] == parent_manifest["sources"]
    assert inner_manifest["sequences"] == parent_manifest["sequences"]
    with pytest.raises(FileExistsError):
        prepare_sequence_inner_split(parent_path, campaign_root, output, 41001, 0.4)

    component_of = {(source, group): component for source, group, component in parent.components}
    outer_components = {component_of[("valid", group)] for group in parent.evaluation_groups}
    inner_evaluation_components = {
        component_of[("valid", group)] for group in inner.evaluation_groups
    }
    inner_fit_components = {
        component_of[(source, group)]
        for source, groups in (("train", inner.train_groups), ("valid", inner.valid_groups))
        for group in groups
    }
    assert not outer_components & (inner_evaluation_components | inner_fit_components)
    assert not inner_evaluation_components & inner_fit_components
    for group in parent.train_groups:
        if component_of[("train", group)] in inner_evaluation_components:
            assert group not in inner.train_groups
    for group in parent.valid_groups:
        if component_of[("valid", group)] in inner_evaluation_components:
            assert group in inner.evaluation_groups


def test_inner_split_rejects_source_drift(campaign_root: Path) -> None:
    parent_path = _outer_manifest(campaign_root)
    output = campaign_root / "inner.json"
    prepare_sequence_inner_split(parent_path, campaign_root, output)
    with (campaign_root / "datasets/train.parquet").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="source changed"):
        load_campaign_split(output, campaign_root)


def test_inner_split_rederives_and_rejects_tampered_membership(campaign_root: Path) -> None:
    parent_path = _outer_manifest(campaign_root)
    output = campaign_root / "inner.json"
    prepare_sequence_inner_split(parent_path, campaign_root, output)
    manifest = json.loads(output.read_text())
    manifest["fit"]["train"].append(0)
    manifest["fit"]["train"].sort()
    output.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="membership differs"):
        load_campaign_split(output, campaign_root)


def test_inner_split_rejects_recursive_wrappers(campaign_root: Path) -> None:
    parent_path = _outer_manifest(campaign_root)
    output = campaign_root / "inner.json"
    prepare_sequence_inner_split(parent_path, campaign_root, output)
    with pytest.raises(ValueError, match="recursive"):
        prepare_sequence_inner_split(output, campaign_root, campaign_root / "nested.json")

    manifest = json.loads(output.read_text())
    manifest["parent"] = {"path": str(output.resolve()), "sha256": "0" * 64}
    output.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="recursive"):
        load_campaign_split(output, campaign_root)
