from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from wnn.splits import (
    MULTISTAGE_AUXILIARY_SPLIT_SCHEMA,
    CampaignSplit,
    file_sha256,
)
from wnn.training.sequence_cache import TeacherCache
from wnn.training.sequence_cache_ensemble_provenance import canonical_sha256, common_provenance
from wnn.training.sequence_cache_weighted_pool import merge_weighted_teacher_caches
from wnn.training.sequence_cache_weighted_pool_lineage import WEIGHTED_POOL_KIND
from wnn.training.sequence_normalization import (
    build_normalization_identity,
    moments_sha256,
    normalization_identity_sha256,
)
from wnn.training.sequence_precision import precision_identity, runtime_precision_identity

AUXILIARY_SHA = "a" * 64
PRIMARY_SHA = "b" * 64
FIT_MEMBERS = [("train", 0), ("train", 1)]
CONTROL_MEMBERS = [("train", 2)]
SOURCES = {
    "datasets/train.parquet": {"sha256": "1" * 64},
    "datasets/valid.parquet": {"sha256": "2" * 64},
}
SPLIT = CampaignSplit(
    Path("split.json"),
    AUXILIARY_SHA,
    0,
    1,
    (0, 1),
    (),
    (2,),
    (("train", 0, 0), ("train", 1, 1), ("train", 2, 2)),
    MULTISTAGE_AUXILIARY_SPLIT_SCHEMA,
    "train",
    "control",
    "all_after_warmup",
    PRIMARY_SHA,
)


def _producer(architecture: str, checkpoint: str, seed: int) -> dict[str, Any]:
    """Build a strict-FP32 fitting-only producer for one heterogeneous family."""
    config: dict[str, Any] = {
        "seed": seed,
        "epochs": 2,
        "batch_size": 16,
        "lr": 0.001,
        "precision_mode": "fp32",
        "training_mask": "all_after_warmup",
        "normalization_scope": "prefix",
        "normalization_sequences": 2,
        "kd_weight": 0.0,
        "hint_weight": 0.0,
        "model": {
            "architecture": architecture,
            "hint_dim": 0,
            "attention_backend": "math",
            "compile_blocks": False,
            "precision_mode": "fp32",
        },
    }
    source_fingerprints = {"datasets/train.parquet": SOURCES["datasets/train.parquet"]}
    mean, std = np.zeros(112, dtype=np.float32), np.ones(112, dtype=np.float32)
    normalization = {
        "scope": "prefix",
        "sequence_count": 2,
        "fit_prefix_sequences": 2,
        "source_fingerprints": source_fingerprints,
    }
    normalization_identity = build_normalization_identity(
        source_fingerprints, AUXILIARY_SHA, FIT_MEMBERS, "prefix", 112
    )
    normalization["identity_sha256"] = normalization_identity_sha256(normalization_identity)
    normalization["moments_sha256"] = moments_sha256(mean, std)
    data = {
        "split_sha256": AUXILIARY_SHA,
        "primary_split_sha256": PRIMARY_SHA,
        "fit_members": [list(member) for member in FIT_MEMBERS],
        "evaluation_members": [list(member) for member in CONTROL_MEMBERS],
        "evaluation_role": "control",
        "evaluation_source": "train",
        "evaluation_mask": "all_after_warmup",
        "training_mask": "all_after_warmup",
        "mask": "all_after_warmup AND row_index>=99",
        "input_schema": "raw112-v1",
        "sources": SOURCES,
        "normalization": normalization,
        "teacher": None,
    }
    source_files = {"scripts/train_sequence.py": "3" * 64}
    runtime = {
        "world_size": 1,
        "matmul_precision": "highest",
        "matmul_tf32": False,
        "cudnn_tf32": False,
        "autocast": False,
        "amp_scaler": False,
        "attention_backend": "math",
        "devices": [
            {
                "rank": 0,
                "local_rank": 0,
                "name": "NVIDIA B300",
                "capability": [10, 0],
                "total_memory_bytes": 100,
            }
        ],
    }
    runtime = runtime_precision_identity(config, runtime)
    identity = {
        "schema": "wnn.sequence-run-identity.v1",
        "config_sha256": canonical_sha256(config),
        "data_sha256": canonical_sha256(data),
        "data": data,
        "source_sha256": canonical_sha256(source_files),
        "source_files": source_files,
        "runtime": runtime,
        "world_size": 1,
    }
    provenance = {
        "schema": "wnn.sequence-provenance.v1",
        "identity": identity,
        "normalization": {"mean": mean.tolist(), "std": std.tolist()},
        "precision": precision_identity("fp32", "math"),
        "completed_epochs": 2,
        "global_updates": 2,
        "fixed_checkpoint_policy": "configured-final-epoch",
    }
    return {
        "checkpoint_sha256": checkpoint,
        "split_sha256": AUXILIARY_SHA,
        "fit_members": [list(member) for member in FIT_MEMBERS],
        "teacher": None,
        "config": config,
        "provenance": provenance,
    }


def _write_source_cache(
    root: Path, architecture: str, checkpoint: str, seed: int, value: float
) -> tuple[Path, dict[str, object]]:
    """Write one two-sequence fixed-fit export and its complete expected source pin."""
    root.mkdir()
    producer = _producer(architecture, checkpoint, seed)
    entries = []
    for source, group in FIT_MEMBERS:
        path = root / f"{source}-{group}.npy"
        np.save(path, np.full((20_000, 2), value, dtype=np.float32), allow_pickle=False)
        entries.append(
            {"split": source, "row_group": group, "file": path.name, "sha256": file_sha256(path)}
        )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "sequence-teacher-cache-v1",
                "complete": True,
                "input_schema": "raw112-v1",
                "membership": "fit",
                "split_sha256": AUXILIARY_SHA,
                "hint_dim": 0,
                "sequence_length": 20_000,
                "producer": producer,
                "sequences": entries,
            }
        )
    )
    manifest_sha = file_sha256(root / "manifest.json")
    identity = producer["provenance"]["identity"]
    return root, {
        "trial_id": f"trial-{seed}",
        "weight": 0.5,
        "cache_manifest_sha256": manifest_sha,
        "checkpoint_sha256": checkpoint,
        "config_sha256": canonical_sha256(producer["config"]),
        "producer_identity_sha256": canonical_sha256(identity),
        "producer_sha256": canonical_sha256(producer),
        "provenance_sha256": canonical_sha256(producer["provenance"]),
    }


def _setup_split(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Write a source-bound split stub and supply its frozen parsed membership."""
    path = tmp_path / "split.json"
    path.write_text(json.dumps({"sources": SOURCES}))
    split = SPLIT._replace(path=path)
    monkeypatch.setattr(
        "wnn.training.sequence_cache_weighted_pool.load_campaign_split",
        lambda _path, _root: split,
    )
    monkeypatch.setattr(
        "wnn.training.sequence_cache_weighted_pool._verify_split_unchanged",
        lambda _split, _manifest, _root: None,
    )
    return path


def _prepare(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, list[Path], list[dict[str, object]]]:
    """Build two distinct architecture producers with exact reference receipts."""
    split_path = _setup_split(tmp_path, monkeypatch)
    first, first_ref = _write_source_cache(tmp_path / "gru", "gru", "4" * 64, 101, 3.0)
    second, second_ref = _write_source_cache(tmp_path / "galam", "galam", "5" * 64, 202, -1.0)
    return split_path, [first, second], [first_ref, second_ref]


def test_weighted_pool_merges_heterogeneous_fit_exports_without_source_clipping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    split_path, inputs, refs = _prepare(tmp_path, monkeypatch)

    output = merge_weighted_teacher_caches(
        inputs,
        refs,
        tmp_path,
        split_path,
        tmp_path / "pool",
        pool_id="pool-gru-galam",
        request_sha256="6" * 64,
        minimum_free_bytes=0,
    )

    value = np.load(output / "train-0.npy", allow_pickle=False)
    assert value.dtype == np.float32 and np.all(value == 1.0)
    manifest = json.loads((output / "manifest.json").read_text())
    producer = manifest["producer"]
    assert producer["kind"] == WEIGHTED_POOL_KIND
    assert len(producer["parents"]) == 2
    for item in producer["parents"]:
        provenance = item["provenance"]
        runtime = provenance["identity"]["runtime"]
        assert provenance["precision"]["mode"] == "fp32"
        assert runtime["precision"] == provenance["precision"]
        assert runtime["compiler"] == {
            "enabled": False,
            "backend": None,
            "mode": None,
            "fullgraph": False,
            "dynamic": None,
            "recompile_limit": None,
        }
    assert {
        item["producer"]["config"]["model"]["architecture"] for item in producer["parents"]
    } == {
        "gru",
        "galam",
    }
    assert producer["composition"]["target_contract"]["parent_clipping"] == "none"
    cache = TeacherCache(output, SPLIT._replace(path=split_path), FIT_MEMBERS)
    assert cache.identity["composition_sha256"] == producer["composition_sha256"]


def test_weighted_pool_rejects_a_changed_pinned_checkpoint_or_cache_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    split_path, inputs, refs = _prepare(tmp_path, monkeypatch)
    refs[1]["checkpoint_sha256"] = "9" * 64

    with pytest.raises(ValueError, match="frozen trial/source receipt"):
        merge_weighted_teacher_caches(
            inputs,
            refs,
            tmp_path,
            split_path,
            tmp_path / "pool",
            pool_id="pool-invalid-pin",
            request_sha256="6" * 64,
            minimum_free_bytes=0,
        )
    assert not (tmp_path / "pool").exists()


def test_weighted_pool_rejects_coefficients_that_do_not_sum_to_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    split_path, inputs, refs = _prepare(tmp_path, monkeypatch)
    refs[0]["weight"] = 0.25
    refs[1]["weight"] = 0.5
    for index, root in enumerate(inputs):
        manifest_path = root / "manifest.json"
        manifest_sha = file_sha256(manifest_path)
        refs[index]["cache_manifest_sha256"] = manifest_sha

    with pytest.raises(ValueError, match="sum to one"):
        merge_weighted_teacher_caches(
            inputs,
            refs,
            tmp_path,
            split_path,
            tmp_path / "pool",
            pool_id="pool-invalid-weight",
            request_sha256="6" * 64,
            minimum_free_bytes=0,
        )


def test_current_teacher_lineage_rejects_control_in_normalizer_membership() -> None:
    producer = _producer("gru", "4" * 64, 101)
    provenance = producer["provenance"]
    identity = provenance["identity"]
    data = identity["data"]
    normalization = data["normalization"]
    contaminated_members = [("train", 0), ("train", 2)]
    contaminated_identity = build_normalization_identity(
        normalization["source_fingerprints"],
        AUXILIARY_SHA,
        contaminated_members,
        "prefix",
        112,
    )
    normalization["identity_sha256"] = normalization_identity_sha256(contaminated_identity)
    identity["data_sha256"] = canonical_sha256(data)

    with pytest.raises(ValueError, match="membership hash"):
        common_provenance(producer, provenance)
