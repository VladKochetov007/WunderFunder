from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_ROOT / "src"))
sys.path.insert(0, str(SOURCE_ROOT / "scripts"))

_EXPORT_SPEC = importlib.util.spec_from_file_location(
    "h230_history_parent_exporter", SOURCE_ROOT / "scripts" / "export_all_data_parent.py"
)
assert _EXPORT_SPEC is not None and _EXPORT_SPEC.loader is not None
PARENT_EXPORTER = importlib.util.module_from_spec(_EXPORT_SPEC)
_EXPORT_SPEC.loader.exec_module(PARENT_EXPORTER)

from wnn.history_swap_cache import (  # noqa: E402
    BASE_CACHE_PATH,
    BASE_METADATA_SHA256,
    LEGACY_VALID_STRONG_SIDECARS,
    _resolve_inference_precision,
    build_history_cache,
    combine_history_chunk,
    validate_prediction_export,
)

from wnn.distillation_cache import row_groups_sha256  # noqa: E402
from wnn.splits import file_sha256  # noqa: E402


def _write_export(tmp_path: Path, *, precision: str = "fp32", features_only: bool = True):
    prediction_path = tmp_path / "prediction.npy"
    sidecar_path = tmp_path / "prediction.json"
    groups = [4, 9]
    values = np.arange(12, dtype=np.float32).reshape(2, 3, 2) / np.float32(7)
    np.save(prediction_path, values)
    sidecar = {
        "run": "parent_run",
        "split": "train",
        "epoch": 3,
        "checkpoint_sha256": "a" * 64,
        "prediction_sha256": file_sha256(prediction_path),
        "inference_precision": precision,
        "row_groups": groups,
        "row_groups_sha256": row_groups_sha256(groups),
        "features_only": features_only,
        "shape": [2, 3, 2],
        "dtype": "float32",
    }
    sidecar_path.write_text(json.dumps(sidecar))
    return prediction_path, sidecar_path, groups, sidecar


def test_history_formula_uses_fp64_accumulation_and_one_fp32_cast():
    source = np.array([[[1.0, -2.0], [3.0, 4.0]]], dtype=np.float32)
    seed3 = np.array([[[0.1, 0.2], [0.3, 0.4]]], dtype=np.float32)
    seed4 = np.array([[[0.5, 0.6], [0.7, 0.8]]], dtype=np.float32)
    parent = np.array([[[1.1, 1.2], [1.3, 1.4]]], dtype=np.float32)
    result = combine_history_chunk(source, seed3, seed4, parent)
    expected = (
        source.astype(np.float64)
        - np.float64(0.15) * seed3.astype(np.float64)
        - np.float64(0.15) * seed4.astype(np.float64)
        + np.float64(0.30) * parent.astype(np.float64)
    ).astype(np.float32)
    np.testing.assert_array_equal(result, expected)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(source, np.array([[[1.0, -2.0], [3.0, 4.0]]], dtype=np.float32))


@pytest.mark.parametrize(
    "bad_field",
    ["shape", "dtype", "finite"],
)
def test_history_formula_rejects_wrong_input_schema_or_values(bad_field: str):
    arrays = [np.zeros((1, 2, 2), dtype=np.float32) for _ in range(4)]
    if bad_field == "shape":
        arrays[2] = np.zeros((1, 2, 1), dtype=np.float32)
    elif bad_field == "dtype":
        arrays[2] = np.zeros((1, 2, 2), dtype=np.float64)
    else:
        arrays[2][0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        combine_history_chunk(*arrays)


def test_actual_prediction_sidecar_schema_accepts_fp32_feature_only_export(tmp_path: Path):
    prediction, sidecar, groups, expected = _write_export(tmp_path)
    verified = validate_prediction_export(
        prediction,
        sidecar,
        expected_run="parent_run",
        expected_split="train",
        expected_epoch=3,
        expected_checkpoint_sha256="a" * 64,
        expected_groups=groups,
        sequence_rows=3,
        require_features_only=True,
    )
    assert verified["sha256"] == expected["prediction_sha256"]
    assert verified["sidecar_sha256"] == hashlib.sha256(sidecar.read_bytes()).hexdigest()
    assert verified["shape"] == [2, 3, 2]
    assert verified["inference_precision"] == "fp32"


def test_parent_export_rejects_order_checkpoint_precision_and_non_feature_reads(tmp_path: Path):
    prediction, sidecar, groups, _ = _write_export(tmp_path)
    shared = {
        "prediction_path": prediction,
        "sidecar_path": sidecar,
        "expected_run": "parent_run",
        "expected_split": "train",
        "expected_epoch": 3,
        "sequence_rows": 3,
    }
    with pytest.raises(ValueError, match="target-index order"):
        validate_prediction_export(
            **shared,
            expected_groups=list(reversed(groups)),
            expected_checkpoint_sha256="a" * 64,
        )
    with pytest.raises(ValueError, match="checkpoint"):
        validate_prediction_export(
            **shared,
            expected_groups=groups,
            expected_checkpoint_sha256="b" * 64,
        )
    with pytest.raises(ValueError, match="features only"):
        sidecar_data = json.loads(sidecar.read_text())
        sidecar_data["features_only"] = False
        sidecar.write_text(json.dumps(sidecar_data))
        validate_prediction_export(
            **shared,
            expected_groups=groups,
            expected_checkpoint_sha256="a" * 64,
            require_features_only=True,
        )
    sidecar_data = json.loads(sidecar.read_text())
    sidecar_data["features_only"] = True
    sidecar_data["inference_precision"] = "fp16"
    sidecar.write_text(json.dumps(sidecar_data))
    with pytest.raises(ValueError, match="FP32"):
        validate_prediction_export(
            **shared,
            expected_groups=groups,
            expected_checkpoint_sha256="a" * 64,
        )


def test_existing_strong_source_sidecars_can_bind_fp16_inference(tmp_path: Path):
    prediction, sidecar, groups, _ = _write_export(tmp_path, precision="fp16", features_only=False)
    verified = validate_prediction_export(
        prediction,
        sidecar,
        expected_run="parent_run",
        expected_split="train",
        expected_epoch=3,
        expected_checkpoint_sha256="a" * 64,
        expected_groups=groups,
        sequence_rows=3,
        required_inference_precision=None,
    )
    assert verified["inference_precision"] == "fp16"


@pytest.mark.parametrize(
    ("seed", "run"),
    ((3, "c20261003_combo_s3_f0"), (4, "c20261003_combo_s4_f0")),
)
def test_actual_legacy_valid_sidecars_remain_precision_unspecified(seed: int, run: str):
    fixture = SOURCE_ROOT / "tests" / "fixtures" / f"legacy_strong_s{seed}_valid.json"
    sidecar = json.loads(fixture.read_text())
    binding = LEGACY_VALID_STRONG_SIDECARS[(run, "valid")]
    groups = sidecar["row_groups"]
    assert len(groups) == 1873
    assert set(sidecar) == {
        "run",
        "split",
        "epoch",
        "checkpoint_sha256",
        "prediction_sha256",
        "split_manifest_sha256",
        "quote_cache",
        "gate_mode",
        "row_groups",
    }
    precision, digest, provenance = _resolve_inference_precision(
        sidecar,
        fixture,
        expected_run=run,
        expected_split="valid",
        expected_epoch=3,
        expected_checkpoint_sha256=binding["checkpoint_sha256"],
        expected_groups=groups,
        required_inference_precision=None,
        expected_sidecar_sha256=binding["sidecar_sha256"],
        legacy_missing_binding=binding,
    )
    assert digest == binding["sidecar_sha256"]
    assert precision == "legacy_unspecified"
    assert provenance["classification"] == "legacy_unspecified"
    assert provenance["base_cache_metadata_sha256"] == BASE_METADATA_SHA256
    assert provenance["producer_identity"]["exporter_source_sha256"] is None


@pytest.mark.parametrize("seed", (3, 4))
def test_actual_strong_train_sidecars_explicitly_declare_fp16(seed: int):
    run = f"c20261003_combo_s{seed}_f0"
    fixture = SOURCE_ROOT / "tests" / "fixtures" / f"legacy_strong_s{seed}_train.json"
    sidecar = json.loads(fixture.read_text())
    groups = sidecar["row_groups"]
    precision, _, provenance = _resolve_inference_precision(
        sidecar,
        fixture,
        expected_run=run,
        expected_split="train",
        expected_epoch=3,
        expected_checkpoint_sha256=sidecar["checkpoint_sha256"],
        expected_groups=groups,
        required_inference_precision="fp16",
        expected_sidecar_sha256=(
            "d9fa1459057e93d481db28aa6388cc7e801242ddcc6eb3e658cf83a559068837"
            if seed == 3
            else "ea462accb17312f12c5a254df9aa94989952562f948234b5ebe2bcc6d5f8401e"
        ),
        legacy_missing_binding=None,
    )
    assert precision == "fp16"
    assert provenance["classification"] == "explicit_sidecar_value"


@pytest.mark.parametrize("precision", ("missing", None))
def test_parent_export_rejects_missing_or_null_precision(tmp_path: Path, precision):
    prediction, sidecar, groups, _ = _write_export(tmp_path)
    sidecar_data = json.loads(sidecar.read_text())
    if precision == "missing":
        del sidecar_data["inference_precision"]
    else:
        sidecar_data["inference_precision"] = None
    sidecar.write_text(json.dumps(sidecar_data))
    with pytest.raises(ValueError, match="must declare FP32"):
        validate_prediction_export(
            prediction,
            sidecar,
            expected_run="parent_run",
            expected_split="train",
            expected_epoch=3,
            expected_checkpoint_sha256="a" * 64,
            expected_groups=groups,
            sequence_rows=3,
            require_features_only=True,
        )


def test_legacy_precision_exception_rejects_noncanonical_binding():
    run = "c20261003_combo_s3_f0"
    fixture = SOURCE_ROOT / "tests" / "fixtures" / "legacy_strong_s3_valid.json"
    sidecar = json.loads(fixture.read_text())
    binding = dict(LEGACY_VALID_STRONG_SIDECARS[(run, "valid")])
    binding["base_metadata_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="not an allowlisted source"):
        _resolve_inference_precision(
            sidecar,
            fixture,
            expected_run=run,
            expected_split="valid",
            expected_epoch=3,
            expected_checkpoint_sha256=binding["checkpoint_sha256"],
            expected_groups=sidecar["row_groups"],
            required_inference_precision=None,
            expected_sidecar_sha256=binding["sidecar_sha256"],
            legacy_missing_binding=binding,
        )


@pytest.mark.parametrize("mutation", ("null", "extra-key"))
def test_legacy_precision_exception_rejects_changed_sidecar_schema(tmp_path: Path, mutation: str):
    run = "c20261003_combo_s3_f0"
    fixture = SOURCE_ROOT / "tests" / "fixtures" / "legacy_strong_s3_valid.json"
    sidecar = json.loads(fixture.read_text())
    if mutation == "null":
        sidecar["inference_precision"] = None
    else:
        sidecar["exporter_revision"] = "unrecorded"
    changed_path = tmp_path / "changed-sidecar.json"
    changed_path.write_text(json.dumps(sidecar))
    binding = LEGACY_VALID_STRONG_SIDECARS[(run, "valid")]
    with pytest.raises(ValueError, match="differs from its pinned cache metadata"):
        _resolve_inference_precision(
            sidecar,
            changed_path,
            expected_run=run,
            expected_split="valid",
            expected_epoch=3,
            expected_checkpoint_sha256=binding["checkpoint_sha256"],
            expected_groups=sidecar["row_groups"],
            required_inference_precision=None,
            expected_sidecar_sha256=binding["sidecar_sha256"],
            legacy_missing_binding=binding,
        )


def test_generic_missing_teacher_precision_is_rejected():
    run = "c20261003_combo_s3_f0"
    fixture = SOURCE_ROOT / "tests" / "fixtures" / "legacy_strong_s3_valid.json"
    sidecar = json.loads(fixture.read_text())
    with pytest.raises(ValueError, match="precision is invalid"):
        _resolve_inference_precision(
            sidecar,
            fixture,
            expected_run=run,
            expected_split="valid",
            expected_epoch=3,
            expected_checkpoint_sha256=sidecar["checkpoint_sha256"],
            expected_groups=sidecar["row_groups"],
            required_inference_precision=None,
            expected_sidecar_sha256=None,
            legacy_missing_binding=None,
        )


def test_parent_exporter_reads_features_without_targets_or_mask(monkeypatch):
    observed: list[tuple[int, bool, bool]] = []

    class FakeSequenceFile:
        def __init__(self, path: Path):
            self.path = path

        def read(self, row_group: int, *, targets: bool, mask: bool):
            observed.append((row_group, targets, mask))
            return SimpleNamespace(features=np.full((3, 2), row_group, dtype=np.float32))

    monkeypatch.setattr(PARENT_EXPORTER, "SequenceFile", FakeSequenceFile)
    dataset = PARENT_EXPORTER.FeatureOnlyGroups(Path("unused.parquet"), [7, 11])
    first = dataset[0]
    second = dataset[1]
    assert observed == [(7, False, False), (11, False, False)]
    np.testing.assert_array_equal(first.numpy(), np.full((3, 2), 7, dtype=np.float32))
    np.testing.assert_array_equal(second.numpy(), np.full((3, 2), 11, dtype=np.float32))


def test_parent_exporter_accepts_only_the_frozen_h230_parent_config():
    train_args = {
        "name": PARENT_EXPORTER.PARENT_RUN,
        "arch": "gru",
        "cell": "gru",
        "hidden": 256,
        "layers": 2,
        "seed": 3,
        "train_source": "train+valid",
        "valid_fold": -1,
        "include_heldout_valid": True,
        "no_eval_during_fit": True,
        "checkpoint_policy": "fixed",
        "checkpoint_epoch": 3,
        "epochs": 4,
        "precision": "fp16",
        "raw_only": False,
        "online_norm": False,
        "heuristics": False,
        "research_features": "none",
        "prop_input": "",
        "privileged_mode": "none",
        "qat": "none",
        "recurrent_rank": 0,
    }
    checkpoint = {"args": train_args, "epoch": 3, "n_in": 211}
    PARENT_EXPORTER._check_parent(checkpoint, SimpleNamespace(run=PARENT_EXPORTER.PARENT_RUN))
    checkpoint["args"] = {**train_args, "precision": "fp32"}
    with pytest.raises(ValueError, match="frozen all-data H230"):
        PARENT_EXPORTER._check_parent(checkpoint, SimpleNamespace(run=PARENT_EXPORTER.PARENT_RUN))


def test_history_student_loader_keeps_family_arrays_in_sealed_base_cache(
    tmp_path: Path, monkeypatch
):
    main_root = tmp_path / "history"
    family_root = tmp_path / "sealed-base"
    main_root.mkdir()
    family_root.mkdir()
    main = np.full((1, 3, 2), 2.0, dtype=np.float32)
    families = np.full((1, 3, 8), 5.0, dtype=np.float32)
    np.save(main_root / "main_history_train.npy", main)
    np.save(family_root / "families_train.npy", families)

    class FakeSequenceFile:
        def __init__(self, path: Path):
            self.path = path

        def read(self, row_group: int):
            return SimpleNamespace(
                features=np.zeros((3, 2), dtype=np.float32),
                targets=np.zeros((3, 2), dtype=np.float32),
                scored=np.ones(3, dtype=np.bool_),
            )

    train_rnn = PARENT_EXPORTER.train_rnn
    monkeypatch.setattr(train_rnn, "SequenceFile", FakeSequenceFile)
    dataset = train_rnn.Sequences(
        "train",
        [19],
        distill_cache_root=main_root,
        distill_cache_variant="history",
        distill_cache_metadata={
            "metadata": {"files": {"main_history_train": {"row_groups": [19]}}}
        },
        include_family_targets=True,
        distill_family_cache_root=family_root,
    )
    _, targets, _ = dataset[0]
    np.testing.assert_array_equal(targets.numpy()[:, 2:4], main[0])
    np.testing.assert_array_equal(targets.numpy()[:, 4:], families[0])


def test_cache_builder_refuses_to_write_inside_the_sealed_base_cache(tmp_path: Path):
    base_cache = tmp_path / BASE_CACHE_PATH
    base_cache.mkdir(parents=True)
    with pytest.raises(ValueError, match="sealed V3 base-cache tree"):
        build_history_cache(
            data_root=tmp_path,
            base_cache=base_cache,
            output=base_cache / "history-child",
            parent={},
            parent_prediction_paths={},
            expected_base_metadata_sha256="b" * 64,
        )
