from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report_multistage_primary_ensemble as primary_parent  # noqa: E402
import report_multistage_primary_ensemble_blend as primary_blend  # noqa: E402
import report_multistage_primary_ensemble_worker as primary_worker  # noqa: E402
from report_multistage_primary_ensemble_blend import weighted_prediction_block  # noqa: E402

from wnn.gmetric import GlobalAccumulator  # noqa: E402


def test_fixed_weight_blend_averages_raw_predictions_before_metric_clipping() -> None:
    predictions = [
        np.asarray([[3.0, -3.0], [1.0, 5.0]], dtype=np.float32),
        np.asarray([[1.0, 1.0], [-1.0, 3.0]], dtype=np.float32),
    ]
    weights = np.asarray([0.5, 0.5], dtype=np.float64)

    actual = weighted_prediction_block(predictions, weights)

    np.testing.assert_array_equal(actual, np.asarray([[2.0, -1.0], [0.0, 4.0]], np.float32))
    clip_each_first = np.average(np.clip(np.stack(predictions), -2.0, 2.0), axis=0, weights=weights)
    assert not np.array_equal(actual, clip_each_first)


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        (np.asarray([0.4, 0.4]), "weights"),
        (np.asarray([1.1, -0.1]), "weights"),
    ],
)
def test_fixed_weight_blend_rejects_non_simplex_weights(weights: np.ndarray, message: str) -> None:
    predictions = [np.zeros((2, 2), np.float32), np.ones((2, 2), np.float32)]

    with pytest.raises(ValueError, match=message):
        weighted_prediction_block(predictions, weights)


def test_fixed_weight_blend_rejects_nonfinite_member_rows() -> None:
    predictions = [np.zeros((2, 2), np.float32), np.ones((2, 2), np.float32)]
    predictions[1][0, 0] = np.nan

    with pytest.raises(ValueError, match="finite rows"):
        weighted_prediction_block(predictions, np.asarray([0.5, 0.5]))


def test_canonical_pooled_metric_scores_the_unclipped_member_blend() -> None:
    predictions = [
        np.asarray([[3.0, -1.0], [0.0, 2.0]], dtype=np.float32),
        np.asarray([[1.0, 1.0], [2.0, 0.0]], dtype=np.float32),
    ]
    weights = np.asarray([0.5, 0.5], dtype=np.float64)
    target = np.asarray([[[1.0, 2.0], [-1.0, 1.0]]], dtype=np.float32)
    mask = np.asarray([[True, True]])
    blended = weighted_prediction_block(predictions, weights)[None]

    accumulator = GlobalAccumulator()
    accumulator.add(target, blended, mask)
    expected = GlobalAccumulator()
    expected.add(target, np.asarray([[[2.0, 0.0], [1.0, 1.0]]], dtype=np.float32), mask)

    assert accumulator.result() == expected.result()


def test_member_collection_preserves_each_worker_log_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "logs").mkdir()
    candidates = []
    for index in range(8):
        receipt_path = output / f"member-{index}.json"
        receipt_path.write_text(
            json.dumps({"predictions": {"path": "prediction.npy", "sha256": "fixed"}}),
            encoding="utf-8",
        )
        candidates.append({"receipt_path": str(receipt_path), "gpu_uuid": f"GPU-{index}"})
    request = {
        "source6": {"root": str(tmp_path)},
        "output_dir": str(output),
        "portable_data_registry": {"path": str(tmp_path / "registry.json")},
        "data_root": str(tmp_path),
        "candidates": candidates,
    }
    args = argparse.Namespace(
        dependency_root=tmp_path, gpu_uuid=[f"GPU-{index}" for index in range(8)], python="python"
    )
    monkeypatch.setattr(
        primary_parent.subprocess,
        "Popen",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, poll=lambda: 0),
    )

    def finish_blend(*args: object, **kwargs: object) -> SimpleNamespace:
        (output / "ensemble_primary_report.json").write_text('{"status":"complete"}')
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(primary_parent.subprocess, "run", finish_blend)

    report = primary_parent._run_workers(args, request, tmp_path / "score_request.json", "fixed")

    assert report["status"] == "complete"
    assert [row["log_path"] for row in report["execution_receipt"]["workers"]] == [
        str(output / "logs" / f"member-{index:02d}.log") for index in range(8)
    ]


def test_blend_reconstructs_complete_source6_mask_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prediction_path = tmp_path / "member.npy"
    np.save(prediction_path, np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    mask = np.zeros(101, dtype=bool)
    mask[99:] = True
    row_indices = np.flatnonzero(mask).astype("<u4")
    membership = [("valid", 7)]
    common_mask = {
        "membership_sha256": "fixed-membership",
        "mask_sha256": hashlib.sha256(np.packbits(mask, bitorder="little").tobytes()).hexdigest(),
        "selected_row_index_sha256": hashlib.sha256(
            np.asarray([7], dtype="<i8").tobytes() + row_indices.tobytes()
        ).hexdigest(),
        "sequence_order": "official primary evaluation membership order; source then row_group",
        "row_order": "ascending row_index among is_scored rows after warmup row 99",
        "selected_rows": 2,
        "per_sequence_selected_rows": [{"row_group": 7, "selected_rows": 2}],
        "mask": "is_scored AND row_index>=99",
        "mask_packed_bitorder": "little",
    }

    class Dataset:
        members = membership

        def __len__(self) -> int:
            return 1

        def validate_source_schema(self) -> None:
            pass

    monkeypatch.setattr(primary_blend, "PRIMARY_ROWS", 2)
    monkeypatch.setattr(primary_blend, "_verify_source6", lambda *args: None)
    monkeypatch.setattr(
        primary_blend,
        "_verify_splits",
        lambda *args: (SimpleNamespace(evaluation_groups=[7]), None),
    )
    monkeypatch.setattr(
        primary_blend,
        "_load_source6",
        lambda source: (
            2,
            lambda: SimpleNamespace(add=lambda *args: None, result=lambda: {"selected_rows": 2}),
            None,
            None,
            None,
            lambda *args, **kwargs: Dataset(),
            lambda *args: {
                "y": SimpleNamespace(numpy=lambda: np.ones((1, 101, 2), np.float32)),
                "mask": SimpleNamespace(numpy=lambda: mask[None]),
            },
            lambda members: "fixed-membership",
            None,
            lambda mode: None,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(set_num_threads=lambda n: None, set_num_interop_threads=lambda n: None),
    )
    request = {
        "source6": {"root": str(tmp_path)},
        "output_dir": str(tmp_path),
        "data_root": str(tmp_path),
        "candidates": [{"weight": 1.0}],
        "weight_vector_sha256": primary_blend._canonical_sha256([1.0]),
    }

    metrics, mixed, _ = primary_blend._blend_and_score(
        request, [(prediction_path, {})], common_mask
    )

    assert metrics["selected_rows"] == 2
    np.testing.assert_array_equal(np.load(mixed), np.load(prediction_path))


def test_member_worker_accepts_bounded_actual_kd_result_size(tmp_path: Path) -> None:
    candidate: dict[str, object] = {}
    for path_key, sha_key in (
        ("trial_result_path", "trial_result_sha256"),
        ("trial_request_path", "trial_request_sha256"),
        ("config_path", "config_sha256"),
    ):
        path = tmp_path / path_key
        with path.open("wb") as stream:
            stream.truncate(38_331_037 if path_key == "trial_result_path" else 2)
        candidate[path_key] = str(path)
        candidate[sha_key] = primary_worker._sha256(path)

    primary_worker._verify_candidate_artifact_hashes(candidate)

    with Path(str(candidate["trial_result_path"])).open("wb") as stream:
        stream.truncate(64_000_001)
    with pytest.raises(ValueError, match="bounded regular file"):
        primary_worker._verify_candidate_artifact_hashes(candidate)


def test_source6_identity_separates_predictor_scorer_and_report_wrapper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "scripts").mkdir()
    (tmp_path / "src" / "wnn").mkdir(parents=True)
    fixtures = {
        "src/wnn/__init__.py": "# package\n",
        "src/wnn/metric.py": "# metric\n",
        "scripts/predict_sequence.py": "# predictor entrypoint\n",
        "scripts/score_sequence.py": "# standalone scorer\n",
        "scripts/report_multistage_primary.py": "# distinct report wrapper\n",
    }
    for name, content in fixtures.items():
        (tmp_path / name).write_text(content, encoding="utf-8", newline="\n")
    expected_files = {
        name: primary_worker._sha256(tmp_path / name)
        for name in fixtures
        if name.startswith("src/") or name == "scripts/predict_sequence.py"
    }
    source_sha = primary_worker._canonical_sha256(expected_files)
    scorer_sha = primary_worker._sha256(tmp_path / "scripts/score_sequence.py")
    monkeypatch.setattr(primary_worker, "SOURCE6_SHA256", source_sha)
    monkeypatch.setattr(primary_worker, "SCORE_SCRIPT_SHA256", scorer_sha)

    identity = primary_worker._source_identity(tmp_path)
    info = primary_parent._source6_info(tmp_path)

    assert identity == {"sha256": source_sha, "files": expected_files}
    assert info["source_sha256"] == source_sha
    assert info["score_script_sha256"] == scorer_sha
    assert scorer_sha != primary_worker._sha256(tmp_path / "scripts/report_multistage_primary.py")


def test_actual_primary_prediction_shape_fits_bounded_reader(tmp_path: Path) -> None:
    path = tmp_path / "primary.npy"
    array = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float32, shape=(primary_blend.PRIMARY_ROWS, 2)
    )
    array.flush()
    del array

    loaded = primary_blend._load_scored_predictions(path)
    assert path.stat().st_size == 28_226_312
    assert loaded.shape == (3_528_273, 2)
    assert loaded.dtype == np.float32
    del loaded

    with path.open("wb") as stream:
        stream.truncate(32_000_001)
    with pytest.raises(ValueError, match="bounded regular file"):
        primary_blend._load_scored_predictions(path)


@pytest.mark.parametrize("shape,dtype", [((3, 2), np.float32), ((2, 2), np.float64)])
def test_primary_prediction_reader_preserves_shape_dtype_guards(
    tmp_path: Path, shape: tuple[int, int], dtype: object
) -> None:
    path = tmp_path / "wrong.npy"
    np.save(path, np.zeros(shape, dtype=dtype))
    with pytest.raises(ValueError, match="wrong dtype or shape"):
        primary_blend._load_scored_predictions(path)
