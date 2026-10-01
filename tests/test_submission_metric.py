"""Callback and verification artifacts must score and retain every sequence."""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import utils
from scripts.verify_submission import verify_submission


class _EchoModel:
    def predict(self, point: utils.DataPoint) -> np.ndarray | None:
        return point.state[:2].copy() if point.need_prediction else None


def _write_data(path: Path, *, unselected_nan: bool = False) -> None:
    length = utils.SEQUENCE_LENGTH
    step = np.arange(length, dtype=np.int64)
    need = step >= utils.WARMUP
    scored = np.ones(length, dtype=bool)
    scored[utils.WARMUP + 3] = False
    with pq.ParquetWriter(path, _table(0, step, need, scored, unselected_nan).schema) as writer:
        for seq in range(2):
            writer.write_table(
                _table(seq, step, need, scored, unselected_nan), row_group_size=length
            )


def _table(
    seq: int,
    step: np.ndarray,
    need: np.ndarray,
    scored: np.ndarray,
    unselected_nan: bool,
) -> pa.Table:
    length = len(step)
    values = np.array([-0.3, -0.5] if seq == 0 else [0.3, 0.5], dtype=np.float32)
    targets = np.broadcast_to(values, (length, 2)).copy()
    if unselected_nan:
        targets[~(need & scored)] = np.nan
    columns = {
        "seq_ix": np.full(length, seq, dtype=np.int64),
        "step_in_seq": step,
        "need_prediction": need,
        "is_scored": scored,
    }
    for k, column in enumerate(utils.FEATURE_COLUMNS):
        columns[column] = np.full(length, values[k] if k < 2 else 0.0, dtype=np.float32)
    for k, column in enumerate(utils.TARGET_COLUMNS):
        columns[column] = targets[:, k]
    return pa.table(columns)


def _submission(path: Path) -> None:
    source = """class PredictionModel:
    def predict(self, point):
        return point.state[:2].copy() if point.need_prediction else None
"""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("solution.py", source)


def test_step_by_step_uses_global_mask_and_pools_ineligible_sequences(tmp_path: Path) -> None:
    data = tmp_path / "validation.parquet"
    _write_data(data, unselected_nan=True)
    result = utils.ScorerStepByStep(data).score(_EchoModel())
    assert result["metric_id"] == "global_wp_clip2_v1"
    assert result["weighted_pearson"] == pytest.approx(1.0)
    assert result["rows_seen"] == 2 * utils.SEQUENCE_LENGTH
    assert result["predictions_required"] == 2 * (utils.SEQUENCE_LENGTH - utils.WARMUP)
    assert result["selected_rows"] == 2 * (utils.SEQUENCE_LENGTH - utils.WARMUP - 1)


@pytest.mark.parametrize("suffix", [".npz", ".npy"])
@pytest.mark.parametrize("unselected_nan", [False, True])
def test_verify_retains_all_sequences_and_handles_unavailable_diagnostic(
    tmp_path: Path, suffix: str, unselected_nan: bool
) -> None:
    data, archive = tmp_path / "validation.parquet", tmp_path / "submission.zip"
    _write_data(data, unselected_nan=unselected_nan)
    _submission(archive)
    output = tmp_path / f"predictions{suffix}"
    result = verify_submission(archive, data, output)
    assert result["metric_id"] == "global_wp_clip2_v1"
    assert result["global_wp"] == 1.0
    assert result["sequences"] == 2
    assert result["block_wp"] is None
    assert result["block_diagnostic_error"] is not None
    if suffix == ".npz":
        with np.load(output) as saved:
            predictions, targets, mask = saved["pred"], saved["y"], saved["mask"]
    else:
        predictions = np.load(output, mmap_mode="r")
        targets = np.load(output.with_suffix(".targets.npy"), mmap_mode="r")
        mask = np.load(output.with_suffix(".mask.npy"), mmap_mode="r")
        assert isinstance(predictions, np.memmap)
    assert predictions.shape == targets.shape == (2, utils.SEQUENCE_LENGTH, 2)
    assert mask.shape == (2, utils.SEQUENCE_LENGTH)
    assert not mask[:, : utils.WARMUP].any()
    assert not mask[:, utils.WARMUP + 3].any()
    np.testing.assert_allclose(predictions[0, utils.WARMUP], [-0.3, -0.5])
    np.testing.assert_allclose(predictions[1, utils.WARMUP], [0.3, 0.5])
    pooled = utils.GlobalAccumulator()
    pooled.add(targets, predictions, mask)
    assert pooled.result()["weighted_pearson"] == pytest.approx(1.0)


def test_step_by_step_requires_finite_predictions_even_on_unscored_rows(tmp_path: Path) -> None:
    data = tmp_path / "validation.parquet"
    _write_data(data)

    class BadModel(_EchoModel):
        def predict(self, point: utils.DataPoint) -> np.ndarray | None:
            if point.step_in_seq == utils.WARMUP + 3:
                return np.array([np.nan, 0.0], dtype=np.float32)
            return super().predict(point)

    with pytest.raises(ValueError, match="finite scores on every required row"):
        utils.ScorerStepByStep(data).score(BadModel())
