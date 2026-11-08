from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.heterogeneous_ensemble_contract import (
    DEFAULT_ENSEMBLES,
    TL3_WEIGHTS,
    PredictionSource,
    load_bank,
    pinned_file,
    read_object,
    validate_role,
)
from scripts.heterogeneous_ensemble_score import (
    blend_predictions,
    ensemble_requirements,
    grade,
    markdown_table,
    score_groups,
)
from wnn.splits import file_sha256


def test_nested_raw_fp32_mixture_precedes_official_clipping() -> None:
    values = {
        "w18": np.array([[10, -10]], dtype=np.float32),
        "tiedlead_s5": np.array([[-1, 1]], dtype=np.float32),
        "tiedlead_s6": np.array([[-1, 1]], dtype=np.float32),
        "other": np.array([[-3, 3]], dtype=np.float32),
    }
    ensembles = {"TL3": TL3_WEIGHTS, "mixture": {"TL3": 0.9, "other": 0.1}}
    actual = blend_predictions("mixture", values.copy(), ensembles)
    # Independent decimal calculation: .9*(.34*10-.33-.33)-.3 = 2.166.
    np.testing.assert_allclose(actual, [[2.166, -2.166]], rtol=0, atol=1e-6)
    assert actual.dtype == np.float32
    prematurely_clipped = 0.9 * np.clip(
        blend_predictions("TL3", values.copy(), ensembles), -2, 2
    ) + 0.1 * np.clip(values["other"], -2, 2)
    assert not np.allclose(actual, prematurely_clipped)


def test_mixture_graph_preserves_canonical_baseline_and_rejects_cycles() -> None:
    members = {"w18", "tiedlead_s5", "tiedlead_s6", "other"}
    ensembles = {"TL3": TL3_WEIGHTS, "mixture": {"TL3": 0.9, "other": 0.1}}
    assert ensemble_requirements(ensembles, members)["mixture"] == members
    with pytest.raises(ValueError, match="TL3 weights"):
        ensemble_requirements({"TL3": {"other": 1}}, members)
    with pytest.raises(ValueError, match="cyclic or unknown"):
        ensemble_requirements({"TL3": TL3_WEIGHTS, "a": {"b": 1}, "b": {"a": 1}}, members)
    with pytest.raises(ValueError, match="summing to one"):
        ensemble_requirements({"TL3": TL3_WEIGHTS, "a": {"other": 0.8}}, members)


def test_teacher_pool_rows_match_the_frozen_distillation_comparison() -> None:
    assert len(DEFAULT_ENSEMBLES) == 12
    assert DEFAULT_ENSEMBLES["GRU_family"] == {"fin256_s3": 0.5, "fin256_s4": 0.5}
    assert DEFAULT_ENSEMBLES["heterogeneous_teacher_pool"] == {
        "fin256_s3": 0.35,
        "fin256_s4": 0.35,
        "galam": 0.1,
        "lstm": 0.1,
        "tcn": 0.1,
    }
    assert DEFAULT_ENSEMBLES["TL3_heterogeneous"] == {
        "TL3": 0.7,
        "galam": 0.1,
        "lstm": 0.1,
        "tcn": 0.1,
    }


def test_reserved_tl3_roles_cannot_be_replaced_by_another_completed_model() -> None:
    validate_role("w18", {"state": "pending", "reason": "cache repair"})
    validate_role("w18", {"run": "tl3c_w18final_distilled_s5", "epoch": 1, "kind": "legacy_bank"})
    with pytest.raises(ValueError, match="reserved TL3 role"):
        validate_role("w18", {"run": "tl3c_fbc1k_s5", "epoch": 3, "kind": "legacy_bank"})
    with pytest.raises(ValueError, match="reserved TL3 role"):
        validate_role(
            "tiedlead_s5", {"run": "tl3c_fttall_tlead_s5", "epoch": 2, "kind": "legacy_bank"}
        )


def test_deserialization_rejects_duplicate_names_and_unbound_payloads(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"TL3":1,"TL3":2}')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        read_object(duplicate)
    payload = tmp_path / "array.npy"
    np.save(payload, np.zeros((2, 2), dtype=np.float32))
    identity = {"path": str(payload), "sha256": file_sha256(payload)}
    assert pinned_file(identity, tmp_path) == payload.resolve()
    payload.write_bytes(b"changed after manifest creation")
    with pytest.raises(ValueError, match="SHA256 changed"):
        pinned_file(identity, tmp_path)


def _legacy_fixture(tmp_path: Path) -> tuple[dict, SimpleNamespace]:
    split = SimpleNamespace(
        schema="wnn.chronological-primary.v1",
        sha256="split",
        train_groups=[4, 5],
        valid_groups=[],
        evaluation_groups=[0, 1],
        assert_fit_groups=lambda *_: None,
    )
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"immutable checkpoint bytes")
    checkpoint_hash = file_sha256(checkpoint)
    provenance = {
        "split_manifest_sha256": "split",
        "fit_row_groups": {"train": [4, 5], "valid": []},
        "evaluation_row_groups": [0, 1],
        "checkpoint_sha256": {"model.pt": checkpoint_hash},
        "args": {
            "name": "probe",
            "precision": "fp32",
            "checkpoint_policy": "fixed",
            "checkpoint_epoch": 3,
        },
    }
    provenance_path = tmp_path / "provenance.json"
    provenance_path.write_text(json.dumps(provenance))
    predictions = tmp_path / "predictions.npy"
    np.save(predictions, np.zeros((2, 20_000, 2), dtype=np.float32))
    metadata = {
        "run": "probe",
        "split": "valid",
        "epoch": 3,
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_file": "model.pt",
        "run_provenance_sha256": file_sha256(provenance_path),
        "prediction_sha256": file_sha256(predictions),
        "split_manifest_sha256": "split",
        "inference_precision": "fp32",
        "row_groups": [0, 1],
    }
    metadata_path = tmp_path / "predictions.json"
    metadata_path.write_text(json.dumps(metadata))
    record = {"run": "probe", "epoch": 3}
    for name, path in (
        ("checkpoint", checkpoint),
        ("provenance", provenance_path),
        ("predictions", predictions),
        ("metadata", metadata_path),
    ):
        record[name] = {"path": str(path), "sha256": file_sha256(path)}
    return record, split


def test_legacy_bank_order_is_bound_to_the_real_source_index(tmp_path: Path) -> None:
    record, split = _legacy_fixture(tmp_path)
    source = load_bank(record, tmp_path, split, [0, 1])
    assert source.sequence(1).shape == (20_000, 2)
    metadata_path = Path(record["metadata"]["path"])
    metadata = json.loads(metadata_path.read_text())
    metadata["row_groups"] = [1, 0]
    metadata_path.write_text(json.dumps(metadata))
    record["metadata"]["sha256"] = file_sha256(metadata_path)
    with pytest.raises(ValueError, match="epoch/order identity"):
        load_bank(record, tmp_path, split, [0, 1])


def test_legacy_bank_rejects_wrong_epoch_and_validation_ancestry(tmp_path: Path) -> None:
    record, split = _legacy_fixture(tmp_path)
    record["epoch"] = 2
    with pytest.raises(ValueError, match="epoch/order identity"):
        load_bank(record, tmp_path, split, [0, 1])
    record["epoch"] = 3
    provenance_path = Path(record["provenance"]["path"])
    provenance = json.loads(provenance_path.read_text())
    provenance["fit_row_groups"]["valid"] = [0]
    provenance_path.write_text(json.dumps(provenance))
    record["provenance"]["sha256"] = file_sha256(provenance_path)
    with pytest.raises(ValueError, match="must not fit on official validation"):
        load_bank(record, tmp_path, split, [0, 1])


def test_full_sequence_streaming_uses_pooled_weighted_pearson_and_warmup(tmp_path: Path) -> None:
    groups, rows = 2, 20_000
    targets = np.zeros((groups, rows, 2), dtype=np.float32)
    predictions = np.zeros_like(targets)
    targets[:, :104] = np.linspace(-3, 3, 104)[None, :, None]
    targets[1, 99:104, 0] = [-1.8, -0.4, 0.3, 1.4, 3]
    targets[1, 99:104, 1] = [2, -1, 0.8, 0.5, -3]
    predictions[:, 99:104] = targets[:, 99:104] * [0.7, -0.3] + [0.2, 0.1]
    mask = np.zeros((groups, rows), dtype=bool)
    mask[:, :104] = True
    data = tmp_path / "valid.parquet"
    with pq.ParquetWriter(
        data, pa.schema([("t0", pa.float32()), ("t1", pa.float32()), ("is_scored", pa.bool_())])
    ) as writer:
        for group in range(groups):
            writer.write_table(
                pa.table(
                    {
                        "t0": targets[group, :, 0],
                        "t1": targets[group, :, 1],
                        "is_scored": mask[group],
                    }
                )
            )
    sources = {
        name: PredictionSource(
            {}, tmp_path / "producer.json", tmp_path / "model.pt", {0: 0, 1: 1}, predictions
        )
        for name in TL3_WEIGHTS
    }
    scores, _, count = score_groups(
        pq.ParquetFile(data), [0, 1], sources, {"TL3": TL3_WEIGHTS}, ["TL3"]
    )
    assert count == 10
    expected = []
    # Independent centered formula pools the ten selected rows, without averaging groups.
    for target in (0, 1):
        y = np.clip(targets[:, 99:104, target].reshape(-1), -2, 2).astype(np.float64)
        p = np.clip(predictions[:, 99:104, target].reshape(-1), -2, 2).astype(np.float64)
        weights = np.abs(y)
        yc = y - np.sum(weights * y) / weights.sum()
        pc = p - np.sum(weights * p) / weights.sum()
        expected.append(
            np.sum(weights * yc * pc) / np.sqrt(np.sum(weights * yc**2) * np.sum(weights * pc**2))
        )
    np.testing.assert_allclose(
        [scores["TL3"]["t0"], scores["TL3"]["t1"]], expected, rtol=0, atol=1e-7
    )
    assert scores["TL3"]["wp"] == pytest.approx(sum(expected) / 2, abs=1e-7)


def test_pending_baseline_has_no_imputed_zero_or_delta() -> None:
    text = markdown_table(
        [
            {
                "name": "TL3",
                "score": None,
                "delta_to_tl3": None,
                "status": "unavailable",
                "missing_members": ["w18"],
            },
            {
                "name": "candidate",
                "score": {"t0": 0.6, "t1": 0.7, "wp": 0.65},
                "delta_to_tl3": None,
                "status": "complete",
                "missing_members": [],
            },
        ]
    )
    assert "| TL3 | — | — | — | — | unavailable: w18 |" in text
    assert "0.650000000 | — | complete" in text


def test_existing_outputs_are_rejected_before_loading_inputs(tmp_path: Path) -> None:
    with pytest.raises(FileExistsError, match="preserve prior results"):
        grade(tmp_path / "missing-manifest.json", tmp_path)
