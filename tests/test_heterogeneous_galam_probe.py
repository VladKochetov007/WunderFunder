from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.heterogeneous_ensemble_contract import validate_role
from scripts.heterogeneous_galam_probe import (
    NATIVE_EPOCH,
    PROBE_CELL,
    PROBE_SCRIPT_SHA256,
    ScoredProbeSource,
    scored_row_layout,
    validate_probe_receipts,
)
from wnn.gmetric import global_wp
from wnn.training.sequence_epoch_prediction import digest


def _probe_receipts() -> tuple[dict, dict, dict, dict, SimpleNamespace, dict]:
    split = SimpleNamespace(sha256="split", train_groups=(10, 11), evaluation_groups=(1, 0))
    data = {
        "split_sha256": split.sha256,
        "fit_members": [["train", group] for group in split.train_groups],
        "evaluation_members": [["valid", group] for group in split.evaluation_groups],
        "teacher": None,
        "hint_mode": "aligned",
        "hint_members": None,
    }
    files = {"model.py": "a" * 64}
    identity = {
        "source_files": files,
        "source_sha256": digest(files),
        "data": data,
        "data_sha256": digest(data),
    }
    admission = {"entry": {"epoch": NATIVE_EPOCH}, "index": {"identity": identity}}
    record = {
        "run": PROBE_CELL,
        "epoch": NATIVE_EPOCH,
        "checkpoint": {"sha256": "b" * 64},
        "epoch_index_sha256": "c" * 64,
        "predictions": {"sha256": "d" * 64},
        "plan": {"sha256": "e" * 64},
    }
    endpoint = {
        "cell": PROBE_CELL,
        "epoch": NATIVE_EPOCH + 1,
        "checkpoint_sha256": record["checkpoint"]["sha256"],
        "epoch_index_sha256": record["epoch_index_sha256"],
    }
    receipt = {
        "endpoint": endpoint,
        "prediction_sha256": record["predictions"]["sha256"],
        "metrics": {"wp": 0.6, "selected_rows": 4},
        "row_order_sha256": "f" * 64,
        "scored_rows": 4,
    }
    plan = {
        "schema": "tabdata.galam-fixed-blend-discovery.v1",
        "split_sha256": split.sha256,
        "members": len(split.evaluation_groups),
        "training_source_sha256": identity["source_sha256"],
        "endpoints": [endpoint],
    }
    result = {
        "plan_sha256": record["plan"]["sha256"],
        "script_sha256": PROBE_SCRIPT_SHA256,
        "scientific_fits_started": 0,
        "training_processes_changed": False,
        "receipts": [dict(receipt, reference_correlations=[])],
    }
    return record, receipt, plan, result, split, admission


def test_probe_joins_original_receipt_and_native_epoch_with_source_ancestry() -> None:
    arguments = _probe_receipts()
    validate_probe_receipts(*arguments)
    record, receipt, plan, result, split, admission = arguments
    wrong_epoch = dict(record, epoch=3)
    with pytest.raises(ValueError, match="native-versus-human epoch"):
        validate_probe_receipts(wrong_epoch, receipt, plan, result, split, admission)
    wrong_plan = dict(plan, training_source_sha256="0" * 64)
    with pytest.raises(ValueError, match="source/fitting ancestry"):
        validate_probe_receipts(record, receipt, wrong_plan, result, split, admission)
    wrong_ancestry = copy.deepcopy(admission)
    wrong_ancestry["index"]["identity"]["data"]["fit_members"].append(["valid", 0])
    with pytest.raises(ValueError, match="source/fitting ancestry"):
        validate_probe_receipts(record, receipt, plan, result, split, wrong_ancestry)


def test_probe_rejects_terminal_script_and_prediction_hash_substitution() -> None:
    record, receipt, plan, result, split, admission = _probe_receipts()
    with pytest.raises(ValueError, match="terminal/source identity"):
        validate_probe_receipts(
            record, receipt, plan, dict(result, script_sha256="0" * 64), split, admission
        )
    wrong_receipt = dict(receipt, prediction_sha256="1" * 64)
    wrong_result = dict(result, receipts=[wrong_receipt])
    with pytest.raises(ValueError, match="predictions differ"):
        validate_probe_receipts(record, wrong_receipt, plan, wrong_result, split, admission)
    with pytest.raises(ValueError, match="receipt does not match"):
        validate_probe_receipts(
            record, dict(receipt, scored_rows=5), plan, result, split, admission
        )


def _validation_masks(path: Path) -> None:
    schema = pa.schema([("is_scored", pa.bool_())])
    with pq.ParquetWriter(path, schema) as writer:
        for positions in ([0, 98, 99, 100], [5, 101, 19_999]):
            mask = np.zeros(20_000, dtype=bool)
            mask[positions] = True
            writer.write_table(pa.table({"is_scored": mask}))


def test_scored_probe_preserves_original_source_order_and_warmup_mask(tmp_path: Path) -> None:
    validation = tmp_path / "valid.parquet"
    _validation_masks(validation)
    positions, offsets, row_hash, count = scored_row_layout(validation, [1, 0])
    assert count == 4 and offsets == {1: (0, 2), 0: (2, 4)}
    assert positions[1].tolist() == [101, 19_999]
    assert positions[0].tolist() == [99, 100]
    expected = hashlib.sha256()
    for source_group, rows in ((1, [101, 19_999]), (0, [99, 100])):
        expected.update(np.asarray(["valid", str(source_group)], dtype="U16").tobytes())
        expected.update(np.asarray(rows, dtype="<i4").tobytes())
    assert row_hash == expected.hexdigest()
    assert row_hash != scored_row_layout(validation, [0, 1])[2]


def test_scored_only_reconstruction_changes_no_observed_value_or_metric(tmp_path: Path) -> None:
    validation = tmp_path / "valid.parquet"
    _validation_masks(validation)
    positions, offsets, _, _ = scored_row_layout(validation, [1, 0])
    observed = np.array([[10, -4], [-0.7, 0.1], [0.2, -0.8], [1, 0.3]], dtype=np.float32)
    source = ScoredProbeSource(
        {},
        tmp_path / "receipt.json",
        tmp_path / "model.pt",
        {},
        observed,
        selected_positions=positions,
        offsets=offsets,
    )
    sequences = [source.sequence(group) for group in (1, 0)]
    np.testing.assert_array_equal(sequences[0][positions[1]], observed[:2])
    np.testing.assert_array_equal(sequences[1][positions[0]], observed[2:])
    assert sequences[0][101, 0] == 10  # Original raw outputs must not be clipped early.
    assert not sequences[0][:99].any()
    target = np.array([[-1.2, 0.3], [1.5, -0.4], [0.2, 1], [-0.8, -1]], dtype=np.float32)
    full_targets = np.zeros((2, 20_000, 2), dtype=np.float32)
    masks = np.zeros((2, 20_000), dtype=bool)
    for index, group in enumerate((1, 0)):
        begin, end = offsets[group]
        masks[index, positions[group]] = True
        full_targets[index, positions[group]] = target[begin:end]
    selected_score = global_wp(target, observed, np.ones(4, dtype=bool))
    reconstructed_score = global_wp(full_targets, np.stack(sequences), masks)
    assert selected_score == reconstructed_score


def test_student_alias_cannot_swap_the_compared_seed_or_arm() -> None:
    validate_role("control_s1301", {"run": "het6_control_s1301", "epoch": 3, "kind": "legacy_bank"})
    with pytest.raises(ValueError, match="reserved TL3 role"):
        validate_role(
            "control_s1301", {"run": "het6_control_s1302", "epoch": 3, "kind": "legacy_bank"}
        )
    with pytest.raises(ValueError, match="reserved TL3 role"):
        validate_role(
            "hetero_s1301", {"run": "het6_control_s1301", "epoch": 3, "kind": "legacy_bank"}
        )
