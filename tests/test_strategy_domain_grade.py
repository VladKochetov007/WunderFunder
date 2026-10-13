from __future__ import annotations

import argparse
import io
import json
import struct
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from grade_strategy_domains import bounded_target_header, grade, prediction  # noqa: E402

from wnn.gmetric import WARMUP, global_wp  # noqa: E402
from wnn.splits import file_sha256  # noqa: E402


def test_npz_header_is_bounded_before_payload() -> None:
    payload = b"\x93NUMPY\x02\x00" + struct.pack("<I", 10_001)
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("y.npy", payload)
    with zipfile.ZipFile(stream) as archive, pytest.raises(ValueError, match="header exceeds"):
        bounded_target_header(archive, "y", (1, 20_000, 2), np.dtype("float32"))


def write_prediction(path: Path, values: np.ndarray, split_hash: str) -> None:
    np.save(path, values)
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "run": path.stem,
                "epoch": 5,
                "checkpoint_sha256": "fixture",
                "prediction_sha256": file_sha256(path),
                "split_manifest_sha256": split_hash,
                "split": "valid",
                "row_groups": [0, 1, 2, 3],
            }
        )
    )


def test_grade_matches_official_metric_and_complete_component_replication(tmp_path: Path) -> None:
    rng = np.random.default_rng(38)
    target = rng.normal(size=(4, 20_000, 2)).astype(np.float32)
    candidate = (0.4 * target + rng.normal(size=target.shape)).astype(np.float32)
    reference = (0.8 * target + rng.normal(size=target.shape)).astype(np.float32)
    mask = rng.random(target.shape[:2]) > 0.7
    cache = tmp_path / "cache"
    cache.mkdir()
    np.savez_compressed(
        cache / "valid_targets.npz", row_group=np.arange(4, dtype=np.int64), y=target, mask=mask
    )
    split = tmp_path / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
    split.parent.mkdir(parents=True)
    split.write_text(
        json.dumps(
            {
                "evaluation": {"valid": [0, 1, 2, 3]},
                "sequences": {
                    "valid": [{"row_group": i, "component": c} for i, c in enumerate([0, 0, 1, 2])]
                },
            }
        )
    )
    split_hash = file_sha256(split)
    write_prediction(cache / "candidate.npy", candidate, split_hash)
    write_prediction(cache / "reference.npy", reference, split_hash)
    panels = tmp_path / "panels.json"
    panels.write_text(
        json.dumps(
            {
                "folds": [
                    {
                        "fold": 0,
                        "split_sha256": split_hash,
                        "evaluation_row_groups": [0, 1, 2, 3],
                        "panels": {
                            "activity": {
                                "status": "available",
                                "evaluation_component_ids": [0, 1, 2],
                                "evaluation_component_domains": [0, 0, 1],
                                "minimum_components": 1,
                            }
                        },
                    }
                ]
            }
        )
    )
    args = argparse.Namespace(
        root=tmp_path,
        fold=0,
        panels=panels,
        candidate=cache / "candidate.npy",
        reference=cache / "reference.npy",
        output=tmp_path / "result.json",
        comparison_label="fixture",
    )
    result = grade(args)
    mask[:, :WARMUP] = False
    assert result["selected_rows"] == int(mask.sum())
    assert result["ordinary"]["candidate_wp"] == pytest.approx(
        global_wp(target, candidate, mask)["wp"], abs=1e-12
    )
    assert result["ordinary"]["reference_wp"] == pytest.approx(
        global_wp(target, reference, mask)["wp"], abs=1e-12
    )
    for scored, repeats in zip(
        result["panels"]["activity"], ([3, 3, 3, 2], [1, 1, 1, 6]), strict=True
    ):
        selection = np.repeat(np.arange(4), repeats)
        expected = global_wp(target[selection], candidate[selection], mask[selection])["wp"]
        assert scored["candidate_wp"] == pytest.approx(expected, abs=1e-12)
    sidecar = cache / "candidate.json"
    metadata = json.loads(sidecar.read_text())
    metadata["row_groups"] = [1, 0, 2, 3]
    sidecar.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="row order"):
        prediction(cache / "candidate.npy", [0, 1, 2, 3], split_hash)
