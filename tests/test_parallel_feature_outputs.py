"""CPU checks for exported prediction metadata consumed by the official grader."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

CODE_DIR = (
    Path(__file__).resolve().parents[1]
    / "research/campaign_2026-10-18/b300_execution_r1/parallel_expansion_v1/code"
)
sys.path.insert(0, str(CODE_DIR))

import feature_outputs  # noqa: E402
from feature_outputs import validate_export  # noqa: E402


@pytest.fixture
def export_fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, object], Path]:
    """Build a tiny on-disk endpoint with the official producer's sidecar schema."""
    run = "feature_group_f2_s1003"
    job: dict[str, object] = {"id": run, "fold": 2}
    root = tmp_path
    split_path = root / "splits/fold2.json"
    split_path.parent.mkdir(parents=True)
    row_groups = [1, 7]
    split_path.write_text(json.dumps({"evaluation": {"valid": row_groups}}))
    work = root / "work" / run
    cache = work / "cache"
    cache.mkdir(parents=True)
    prediction = cache / f"pred_{run}_valid.npy"
    np.save(prediction, np.zeros((len(row_groups), 20_000, 2), dtype=np.float32))
    prediction_digest = _sha(prediction)
    split_digest = _sha(split_path)
    membership_digest = hashlib.sha256(np.asarray(row_groups, dtype="<i8").tobytes()).hexdigest()
    metadata = {
        "run": run,
        "split": "valid",
        "epoch": 5,
        "checkpoint_sha256": "a" * 64,
        "prediction_sha256": prediction_digest,
        "split_manifest_sha256": split_digest,
        "campaign_split_manifest_sha256": split_digest,
        "quote_cache": None,
        "gate_mode": "soft",
        "inference_precision": "fp32",
        "row_groups": row_groups,
        "evaluation_membership_sha256": membership_digest,
        "campaign_export_scope": "evaluation",
        "feature_only": True,
    }
    sidecar = prediction.with_suffix(".json")
    sidecar.write_text(json.dumps(metadata))
    return root, work, job, sidecar


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_valid_export_matches_official_grader_metadata(export_fixture: tuple) -> None:
    root, work, job, _ = export_fixture

    evidence = validate_export(root, work, job, "a" * 64)

    assert evidence["shape"] == [2, 20_000, 2]
    assert evidence["dtype"] == "float32"
    assert evidence["checkpoint_sha256"] == "a" * 64


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("split", "train"),
        ("split_manifest_sha256", "0" * 64),
        ("feature_only", False),
        ("inference_precision", "fp16"),
        ("evaluation_membership_sha256", "0" * 64),
        ("campaign_split_manifest_sha256", "0" * 64),
        ("checkpoint_sha256", "0" * 64),
        ("row_groups", [True, 7]),
    ],
)
def test_export_rejects_metadata_the_official_grader_would_reject(
    export_fixture: tuple, field: str, value: object
) -> None:
    root, work, job, sidecar = export_fixture
    metadata = json.loads(sidecar.read_text())
    metadata[field] = value
    sidecar.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="metadata"):
        validate_export(root, work, job, "a" * 64)


def test_export_rejects_wrong_checkpoint_even_with_matching_prediction_hash(
    export_fixture: tuple,
) -> None:
    root, work, job, _ = export_fixture

    with pytest.raises(ValueError, match="metadata"):
        validate_export(root, work, job, "b" * 64)


def test_grade_pair_matches_official_cli_and_result_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    state = tmp_path / "state"
    family, fold, seed = "age", 2, 1003
    split_path = root / f"splits/fold{fold}.json"
    split_path.parent.mkdir(parents=True)
    row_groups = [1, 7]
    split_path.write_text(json.dumps({"evaluation": {"valid": row_groups}}))
    split_digest = _sha(split_path)
    root.joinpath("data/datasets").mkdir(parents=True)
    for filename in ("chains.npz", "chain_folds.npz", "train_targets.npz", "valid_targets.npz"):
        target = root / "data/cache" / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"fixture")
    grader = root / "runtime/b300-r01-v1/grader/grade_endpoint.py"
    grader.parent.mkdir(parents=True)
    shutil.copyfile(
        Path(__file__).resolve().parents[1]
        / "research/campaign_2026-10-18/tabdata_l4_r1/grader/grade_endpoint.py",
        grader,
    )

    def make_endpoint(job_id: str) -> dict[str, object]:
        work = state / "work" / job_id
        cache = work / "cache"
        run_dir = work / "artifacts/runs" / job_id
        cache.mkdir(parents=True)
        run_dir.mkdir(parents=True)
        (work / "FIT.json").write_text(job_id)
        (work / "EXPORTED.json").write_text(job_id)
        checkpoint = "c" * 64 if job_id == "candidate" else "d" * 64
        prediction = cache / f"pred_{job_id}_valid.npy"
        np.save(prediction, np.zeros((len(row_groups), 20_000, 2), dtype=np.float32))
        metadata = {
            "run": job_id,
            "split": "valid",
            "epoch": 5,
            "checkpoint_sha256": checkpoint,
            "prediction_sha256": _sha(prediction),
            "split_manifest_sha256": split_digest,
            "campaign_split_manifest_sha256": split_digest,
            "inference_precision": "fp32",
            "row_groups": row_groups,
            "evaluation_membership_sha256": hashlib.sha256(
                np.asarray(row_groups, dtype="<i8").tobytes()
            ).hexdigest(),
            "campaign_export_scope": "evaluation",
            "feature_only": True,
        }
        prediction.with_suffix(".json").write_text(json.dumps(metadata))
        job = {"id": job_id, "family": family, "fold": fold, "seed": seed}
        return {
            "job": job,
            "fit": {"checkpoint_sha256": checkpoint},
            "fit_receipt_sha256": _sha(work / "FIT.json"),
            "prediction": {"split_sha256": split_digest},
        }

    candidate = make_endpoint("candidate")
    control = make_endpoint("control")
    expected = {
        "metric_id": "global_wp_clip2_v1",
        "fold": fold,
        "split_manifest_sha256": split_digest,
        "endpoint": {"run": "candidate", "score": {"wp": 0.25}},
        "reference": {"run": "control", "score": {"wp": 0.2}},
    }
    observed: dict[str, object] = {}

    def run(argv: list[str], **kwargs: object) -> SimpleNamespace:
        observed["argv"] = argv
        output = Path(argv[argv.index("--output") + 1])
        output.write_text(json.dumps(expected))
        return SimpleNamespace(returncode=0)

    def fake_symlink(self: Path, target: Path, target_is_directory: bool = False) -> None:
        if target_is_directory:
            self.mkdir(parents=True)
        else:
            self.write_bytes(b"mock-link")

    monkeypatch.setattr(feature_outputs.subprocess, "run", run)
    monkeypatch.setattr(Path, "symlink_to", fake_symlink)

    receipt = feature_outputs.grade_pair(root, state, candidate, control)

    argv = observed["argv"]
    assert isinstance(argv, list)
    assert argv[argv.index("--run") + 1] == "candidate"
    assert argv[argv.index("--reference") + 1] == "control@5"
    assert argv[argv.index("--split") + 1] == str(split_path)
    assert argv[argv.index("--epoch") + 1] == "5"
    assert argv[argv.index("--bootstrap") + 1] == "2000"
    assert receipt["candidate_wp"] == 0.25
    assert receipt["control_wp"] == 0.2
