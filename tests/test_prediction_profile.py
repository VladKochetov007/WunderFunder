from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from scripts.predict_run import (
    build_parser,
    reserve_create_only_prediction_paths,
    source_hashes,
    validate_profile_checkpoint,
)
from wnn.prediction_profile import first_profile_groups, profile_record, write_json_create_only


def test_profile_group_selection_preserves_the_first_fixed_fit_membership() -> None:
    groups = list(range(20, 60))

    assert first_profile_groups(groups) == groups[:16]


def test_prediction_cli_accepts_explicit_create_only_profile_contract() -> None:
    args = build_parser().parse_args(
        [
            "teacher",
            "train",
            "--epoch",
            "2",
            "--campaign-split-manifest",
            "F0.json",
            "--inference-precision",
            "fp32",
            "--loader-workers",
            "0",
            "--profile-only",
            "--profile-output",
            "PROFILE.json",
        ]
    )

    assert args.profile_only and args.loader_workers == 0
    assert args.profile_output.name == "PROFILE.json"


def test_profile_checkpoint_identity_requires_raw112_model211_h384x2_teacher() -> None:
    checkpoint = {
        "args": {"arch": "gru", "hidden": 384, "layers": 2, "privileged_mode": "none"},
        "n_in": 211,
        "model": {"head.2.weight": torch.zeros((23, 384))},
        "readout": None,
        "quantization": None,
        "factorization": None,
    }

    validate_profile_checkpoint(checkpoint)
    with pytest.raises(ValueError, match="raw-112/model-211"):
        validate_profile_checkpoint({**checkpoint, "n_in": 212})


def test_profile_source_identity_covers_the_transitive_inference_path() -> None:
    sources = source_hashes()

    assert {
        "scripts/predict_run.py",
        "src/wnn/prediction_profile.py",
        "src/wnn/training/evaluation.py",
        "src/wnn/training/inputs.py",
        "src/wnn/training/forward.py",
        "src/wnn/nn.py",
        "src/wnn/data.py",
    }.issubset(sources)
    assert all(len(digest) == 64 for digest in sources.values())


@pytest.mark.parametrize("groups", [list(range(15)), [*range(15), 0]])
def test_profile_group_selection_rejects_short_or_repeated_membership(groups: list[int]) -> None:
    with pytest.raises(ValueError, match="profile membership"):
        first_profile_groups(groups)


def test_profile_record_reports_only_finite_scalar_prediction_summary() -> None:
    predictions = np.arange(12, dtype=np.float32).reshape(2, 3, 2)
    record = profile_record(
        predictions,
        [4, 9],
        sequence_length=3,
        checkpoint_sha256="c" * 64,
        source_sha256={"predict_run.py": "s" * 64},
        split_manifest_sha256="m" * 64,
        normalization_identity={"mean": {"bytes_sha256": "n" * 64}},
        elapsed_seconds=1.25,
        rss_peak_bytes=4096,
        cuda={"peak_allocated_bytes": 2048},
        profile_contract={"batch_size": 2, "chunk_size": 4000},
    )

    assert record["record_type"] == "PROFILE"
    assert record["status"] == "PASS"
    assert record["prediction_shape"] == [2, 3, 2]
    assert record["prediction_dtype"] == "float32"
    assert record["finite_output_count"] == 12
    assert record["rows"] == 6
    assert record["prediction_min"] == 0.0
    assert record["prediction_max"] == 11.0
    assert "predictions" not in record


def test_profile_record_rejects_nonfinite_or_non_fp32_output() -> None:
    common = {
        "groups": [3],
        "sequence_length": 2,
        "checkpoint_sha256": "c" * 64,
        "source_sha256": {},
        "split_manifest_sha256": "m" * 64,
        "normalization_identity": {},
        "elapsed_seconds": 1.0,
        "rss_peak_bytes": 1024,
        "cuda": {},
        "profile_contract": {},
    }
    nonfinite = np.zeros((1, 2, 2), dtype=np.float32)
    nonfinite[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite FP32"):
        profile_record(nonfinite, **common)
    with pytest.raises(ValueError, match="finite FP32"):
        profile_record(np.zeros((1, 2, 2), dtype=np.float64), **common)


def test_json_profile_writer_is_create_only(tmp_path) -> None:
    path = tmp_path / "PROFILE.json"
    write_json_create_only(path, {"record_type": "PROFILE", "value": 1})

    assert json.loads(path.read_text()) == {"record_type": "PROFILE", "value": 1}
    with pytest.raises(FileExistsError):
        write_json_create_only(path, {"value": 2})
    assert json.loads(path.read_text()) == {"record_type": "PROFILE", "value": 1}


def test_prediction_export_reserves_partial_and_rejects_any_existing_pair_member(
    tmp_path,
) -> None:
    final, partial, sidecar = reserve_create_only_prediction_paths(tmp_path, "teacher", "train")

    assert partial.is_file() and not final.exists() and not sidecar.exists()
    with pytest.raises(FileExistsError, match="create-only prediction outputs"):
        reserve_create_only_prediction_paths(tmp_path, "teacher", "train")
    with pytest.raises(ValueError, match="single filename component"):
        reserve_create_only_prediction_paths(tmp_path, "../teacher", "train")
