from __future__ import annotations

import json
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from wnn import heterogeneous_pool_cache
from wnn.distillation_cache import validate_distillation_cache
from wnn.heterogeneous_pool_cache import (
    STUDENT_CONTRACT,
    build_pool_cache,
    mix_source_rows,
    validate_pool_student_options,
)
from wnn.heterogeneous_pool_sources import (
    PROFILES,
    SPLIT_SHA256,
    groups_hash,
    validate_profile,
    validate_source,
)
from wnn.splits import CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA, CampaignSplit, file_sha256
from wnn.training.sequence_epoch_prediction import digest


def test_pool_join_is_by_group_and_combines_before_clipping() -> None:
    # Two physical source indexes have different orders and include unused rows.
    first = np.asarray([[[90.0, 90.0]], [[10.0, -10.0]], [[4.0, -4.0]]], dtype=np.float32)
    second = np.asarray([[[-2.0, 2.0]], [[999.0, 999.0]], [[6.0, -6.0]]], dtype=np.float32)
    result = mix_source_rows(
        [first, second], [{99: 0, 7: 1, 2: 2}, {7: 0, 99: 1, 2: 2}], [0.5, 0.5], [2, 7]
    )
    np.testing.assert_array_equal(
        result, np.asarray([[[5.0, -5.0]], [[4.0, -4.0]]], dtype=np.float32)
    )
    assert result.dtype == np.float32
    assert np.max(result) > 2  # Cache construction must not clip individual teachers.


def test_pool_join_rejects_missing_members_and_nonfinite_targets() -> None:
    values = np.zeros((2, 3, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="omits a fitting group"):
        mix_source_rows([values], [{1: 0, 2: 1}], [1.0], [3])
    values[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        mix_source_rows([values], [{1: 0, 2: 1}], [1.0], [1])


def test_profiles_reject_changed_weights_or_teacher_order() -> None:
    sources = [
        dict(zip(("run", "family", "epoch", "coefficient"), item, strict=True))
        for item in PROFILES["heterogeneous"]
    ]
    validate_profile("heterogeneous", sources)
    sources[0]["coefficient"] = 0.3
    with pytest.raises(ValueError, match="frozen profile"):
        validate_profile("heterogeneous", sources)
    sources[0]["coefficient"] = 0.35
    with pytest.raises(ValueError, match="frozen profile"):
        validate_profile("heterogeneous", list(reversed(sources)))


@pytest.mark.parametrize(
    "field,value",
    [
        ("distill_boot_h", 8),
        ("distill_cache_variant", "normalized"),
        ("distill_family_weight", 0.1),
        ("distill_cache_scale", 0.8125),
        ("init", "old_parent"),
        ("distill_alpha", 0.9),
    ],
)
def test_new_cache_student_contract_rejects_historical_recipe_drift(
    field: str, value: object
) -> None:
    args = SimpleNamespace(**STUDENT_CONTRACT)
    validate_pool_student_options(args)
    setattr(args, field, value)
    with pytest.raises(ValueError, match=field):
        validate_pool_student_options(args)


def _write_json(path: Path, value: object) -> dict[str, str]:
    path.write_text(json.dumps(value), encoding="utf-8")
    return {"path": str(path.resolve()), "sha256": file_sha256(path)}


def _legacy_source(tmp_path: Path) -> tuple[dict, SimpleNamespace]:
    # Synthetic byte/metadata fixture checks admission, not a real model reconstruction.
    split = SimpleNamespace(
        train_groups=(2, 7),
        valid_groups=(),
        evaluation_groups=(1,),
        sha256=SPLIT_SHA256,
        schema=CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA,
        components=(("train", 2, 2), ("train", 7, 7), ("valid", 1, 1)),
    )
    split.assert_fit_groups = MethodType(CampaignSplit.assert_fit_groups, split)
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"synthetic non-executable checkpoint identity")
    checkpoint_pin = {"path": str(checkpoint.resolve()), "sha256": file_sha256(checkpoint)}
    values = np.zeros((3, 20000, 2), dtype=np.float32)
    values[0] = 7
    values[2] = 2
    prediction = tmp_path / "prediction.npy"
    np.save(prediction, values, allow_pickle=False)
    prediction_pin = {"path": str(prediction.resolve()), "sha256": file_sha256(prediction)}
    files = {"model.py": "a" * 64}
    source_sha = digest(files)
    provenance = {
        "args": {
            "name": "tl3c_fin256_s3",
            "train_source": "train",
            "precision": "fp32",
            "checkpoint_policy": "fixed",
            "checkpoint_epoch": 3,
            "arch": "gru",
            "cell": "gru",
            "hidden": 256,
            "layers": 2,
        },
        "selected_epoch": 3,
        "fit_row_groups": {"train": [2, 7], "valid": []},
        "evaluation_row_groups": [1],
        "split_manifest_sha256": SPLIT_SHA256,
        "checkpoint_sha256": {"model.pt": checkpoint_pin["sha256"]},
        "trainer_implementation": {"files": files, "sha256": source_sha},
        "normalization_identity": {
            "mean": {"dtype": "torch.float32"},
            "std": {"dtype": "torch.float32"},
        },
    }
    provenance_pin = _write_json(tmp_path / "provenance.json", provenance)
    side = {
        "run": "tl3c_fin256_s3",
        "epoch": 3,
        "split": "train",
        "feature_only": True,
        "inference_precision": "fp32",
        "row_groups": [7, 99, 2],
        "split_manifest_sha256": SPLIT_SHA256,
        "checkpoint_sha256": checkpoint_pin["sha256"],
        "prediction_sha256": prediction_pin["sha256"],
        "run_provenance_sha256": provenance_pin["sha256"],
        "campaign_export_scope": "chronological-full-train-index-feature-only",
        "campaign_split_manifest_sha256": SPLIT_SHA256,
        "evaluation_membership_sha256": groups_hash([7, 99, 2]),
    }
    side_pin = _write_json(tmp_path / "prediction.json", side)
    qualification = {
        "schema": "wnn.teacher-source-qualification.v1",
        "run": "tl3c_fin256_s3",
        "producer": "legacy_rnn_chronological",
        "epoch": 3,
        "checkpoint_sha256": checkpoint_pin["sha256"],
        "prediction_sha256": prediction_pin["sha256"],
        "sidecar_sha256": side_pin["sha256"],
        "split_manifest_sha256": SPLIT_SHA256,
        "fit_groups": {"train": [2, 7], "valid": []},
        "evaluation_valid_groups": [1],
        "source_sha256": source_sha,
        "normalization_fit_groups_sha256": groups_hash([2, 7]),
    }
    qualification_pin = _write_json(tmp_path / "qualification.json", qualification)
    return {
        "run": "tl3c_fin256_s3",
        "family": "gru",
        "epoch": 3,
        "coefficient": 0.5,
        "producer": "legacy_rnn_chronological",
        "prediction": prediction_pin,
        "sidecar": side_pin,
        "checkpoint": checkpoint_pin,
        "provenance": provenance_pin,
        "qualification": qualification_pin,
    }, split


def test_source_admission_joins_full_index_and_binds_raw_provenance(tmp_path: Path) -> None:
    source, split = _legacy_source(tmp_path)
    values, index = validate_source(source, split)
    np.testing.assert_array_equal(values[index[2]], np.full((20000, 2), 2, dtype=np.float32))
    assert index[2] == 2 and index[7] == 0
    Path(source["provenance"]["path"]).write_text("{}")
    with pytest.raises(ValueError, match="provenance bytes changed"):
        validate_source(source, split)


def test_self_asserted_qualification_cannot_override_actual_fit_membership(tmp_path: Path) -> None:
    source, split = _legacy_source(tmp_path)
    path = Path(source["provenance"]["path"])
    provenance = json.loads(path.read_text())
    provenance["fit_row_groups"]["valid"] = [1]
    source["provenance"] = _write_json(path, provenance)
    side_path = Path(source["sidecar"]["path"])
    side = json.loads(side_path.read_text())
    side["run_provenance_sha256"] = source["provenance"]["sha256"]
    source["sidecar"] = _write_json(side_path, side)
    qualification_path = Path(source["qualification"]["path"])
    qualification = json.loads(qualification_path.read_text())
    qualification["sidecar_sha256"] = source["sidecar"]["sha256"]
    source["qualification"] = _write_json(qualification_path, qualification)
    with pytest.raises(ValueError, match="must not fit on official validation"):
        validate_source(source, split)


def test_legacy_membership_order_can_differ_without_changing_prediction_alignment(
    tmp_path: Path,
) -> None:
    source, split = _legacy_source(tmp_path)
    provenance_path = Path(source["provenance"]["path"])
    provenance = json.loads(provenance_path.read_text())
    provenance["fit_row_groups"]["train"] = [7, 2]
    source["provenance"] = _write_json(provenance_path, provenance)
    side_path = Path(source["sidecar"]["path"])
    side = json.loads(side_path.read_text())
    side["run_provenance_sha256"] = source["provenance"]["sha256"]
    source["sidecar"] = _write_json(side_path, side)
    qualification_path = Path(source["qualification"]["path"])
    qualification = json.loads(qualification_path.read_text())
    qualification["sidecar_sha256"] = source["sidecar"]["sha256"]
    source["qualification"] = _write_json(qualification_path, qualification)
    values, positions = validate_source(source, split)
    assert positions == {7: 0, 99: 1, 2: 2}
    np.testing.assert_array_equal(values[positions[2]], np.full((20000, 2), 2, dtype=np.float32))
    provenance["fit_row_groups"]["train"] = [7, 7]
    source["provenance"] = _write_json(provenance_path, provenance)
    with pytest.raises(ValueError, match="repeats row groups"):
        validate_source(source, split)


def _rename_source(source: dict, run: str) -> dict:
    source["run"] = run
    provenance_path = Path(source["provenance"]["path"])
    provenance = json.loads(provenance_path.read_text())
    provenance["args"]["name"] = run
    source["provenance"] = _write_json(provenance_path, provenance)
    side_path = Path(source["sidecar"]["path"])
    side = json.loads(side_path.read_text())
    side.update(run=run, run_provenance_sha256=source["provenance"]["sha256"])
    source["sidecar"] = _write_json(side_path, side)
    qualification_path = Path(source["qualification"]["path"])
    qualification = json.loads(qualification_path.read_text())
    qualification.update(run=run, sidecar_sha256=source["sidecar"]["sha256"])
    source["qualification"] = _write_json(qualification_path, qualification)
    return source


def test_full_pool_build_and_consumer_dispatch_with_reduced_synthetic_membership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "first").mkdir()
    (tmp_path / "second").mkdir()
    first, split = _legacy_source(tmp_path / "first")
    second, _ = _legacy_source(tmp_path / "second")
    second = _rename_source(second, "tl3c_fin256_s4")
    split.fold = 0
    root = tmp_path / "data"
    (root / "cache").mkdir(parents=True)
    np.savez(root / "cache/train_targets.npz", row_group=np.array([7, 99, 2], dtype=np.int64))
    inputs = tmp_path / "inputs.json"
    _write_json(
        inputs,
        {
            "format": "wnn.heterogeneous-teacher-pool-inputs.v1",
            "profile": "gru-only",
            "sources": [first, second],
        },
    )
    # Only admission cardinality/source-file loading is reduced. Producer checks,
    # row join, arithmetic, payload validation and canonical dispatch remain real.
    monkeypatch.setattr(heterogeneous_pool_cache, "_require_split", lambda value: None)
    monkeypatch.setattr(heterogeneous_pool_cache, "load_campaign_split", lambda *args: split)
    output = build_pool_cache(
        inputs, root, tmp_path / "unused-split.json", tmp_path / "pool", min_free_bytes=0
    )
    result = validate_distillation_cache(output, root, split, "source", require_families=False)
    assert result["metadata"]["files"]["main_source_train"]["row_groups"] == [7, 2]
    actual = np.load(output / "main_source_train.npy", allow_pickle=False)
    np.testing.assert_array_equal(actual[:, 0, 0], np.array([7, 2], dtype=np.float32))
    assert np.load(output / "main_source_valid.npy", allow_pickle=False).shape == (0, 20000, 2)
    with pytest.raises(FileExistsError):
        build_pool_cache(inputs, root, tmp_path / "unused-split.json", output, min_free_bytes=0)
    with pytest.raises(ValueError, match="without family loss"):
        validate_distillation_cache(output, root, split, "source", require_families=True)
    np.save(output / "main_source_train.npy", actual + 1, allow_pickle=False)
    with pytest.raises(ValueError, match="payload identity"):
        validate_distillation_cache(output, root, split, "source", require_families=False)
