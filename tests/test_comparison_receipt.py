"""Focused tests for strict completed-comparison receipts."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from comparison_receipt import ComparisonReceiptError, record_comparison_receipt
from warm_queue_worker import QueueWorker

METRIC_ID = "global_wp_clip2_v1"


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _fixture(tmp_path: Path) -> tuple[dict[str, object], Path, Path, bytes]:
    split_path = tmp_path / "fold2.json"
    split_bytes = json.dumps({"fold": 2, "evaluation": {"valid": [1, 2]}}).encode()
    split_path.write_bytes(split_bytes)
    split_hash = _sha(split_bytes)
    grade_path = tmp_path / "grade.json"
    receipt_path = tmp_path / "receipt.json"
    ledger_path = tmp_path / "comparisons.jsonl"

    def member(run: str, epoch: int, weight: float = 1.0) -> dict[str, object]:
        return {
            "run": run,
            "epoch": epoch,
            "checkpoint_sha256": "a" * 64,
            "prediction_sha256": "b" * 64,
            "split_manifest_sha256": split_hash,
            "weight": weight,
        }

    comparison: dict[str, object] = {
        "experiment_id": "exp_fold2_001",
        "hypothesis_id": "hyp_mechanism_01",
        "candidate": "candidate_run@ep3",
        "references": ["reference_run@ep2"],
        "grade_path": str(grade_path),
        "receipt_path": str(receipt_path),
        "ledger_path": str(ledger_path),
        "split_path": str(split_path),
        "expected_fold": 2,
        "fidelity": {"budget": "20% training updates", "precision": "fp32"},
        "input_hashes": {"targets_sha256": "c" * 64},
    }
    grade = {
        "metric_id": METRIC_ID,
        "candidate": "candidate_run@ep3",
        "expected_fold": 2,
        "split_path": str(split_path),
        "split_manifest_sha256": split_hash,
        "candidate_members": [member("candidate_run", 3)],
        "official": {"metric_id": METRIC_ID, "wp": 0.72, "selected_rows": 1234},
        "evaluation_groups": 468,
        "evaluation_components": 141,
        "valid_targets_sha256": "c" * 64,
        "exposure": "exposed development fold",
        "ancestry": {"initializer": "parent_checkpoint"},
        "uncertainty": {"ci95": [-0.01, 0.02]},
        "contrasts": [
            {
                "reference": "reference_run@ep2",
                "reference_members": [member("reference_run", 2)],
                "reference_official": {"metric_id": METRIC_ID, "wp": 0.71},
                "paired": {
                    "metric_id": METRIC_ID,
                    "candidate_wp": 0.72,
                    "reference_wp": 0.71,
                    "delta": 0.01,
                    "ci95": [-0.01, 0.02],
                },
            }
        ],
    }
    grade_path.write_text(json.dumps(grade))
    spec_bytes = json.dumps({"run_id": "candidate_run", "comparison": comparison}).encode()
    return comparison, grade_path, ledger_path, spec_bytes


def test_receipt_preserves_metrics_hashes_timing_and_uncertainty(tmp_path: Path) -> None:
    comparison, grade_path, ledger_path, spec_bytes = _fixture(tmp_path)
    record = record_comparison_receipt(
        comparison,
        run_id="candidate_run",
        spec_bytes=spec_bytes,
        steps=[
            {"name": "fit", "status": "succeeded", "wall_seconds": 12.5},
            {"name": "grade", "status": "succeeded", "wall_seconds": 3.0},
        ],
    )

    receipt = json.loads(Path(record["receipt_path"]).read_text())
    ledger_rows = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    assert record["status"] == "valid"
    assert record["ledger_appended"] is True
    assert receipt["hashes"]["grade_sha256"] == _sha(grade_path.read_bytes())
    assert receipt["hashes"]["spec_sha256"] == _sha(spec_bytes)
    assert receipt["candidate"]["official"]["selected_rows"] == 1234
    assert receipt["grade_metadata"]["evaluation_groups"] == 468
    assert receipt["grade_metadata"]["evaluation_components"] == 141
    assert receipt["grade_metadata"]["valid_targets_sha256"] == "c" * 64
    assert receipt["grade_metadata"]["uncertainty"]["ci95"] == [-0.01, 0.02]
    assert receipt["phase_timing"]["total_wall_seconds"] == 15.5
    assert receipt["phase_timing"]["measurement"] == "subprocess_wall_seconds"
    assert "GPU busy time" in receipt["limits"][0]
    assert len(ledger_rows) == 1


def test_declared_blend_with_explicit_weights_is_validated_and_retained(tmp_path: Path) -> None:
    comparison, grade_path, _, spec_bytes = _fixture(tmp_path)
    grade = json.loads(grade_path.read_text())
    blend = "candidate_a@ep1*0.25,candidate_b@ep2*0.75"
    comparison["candidate"] = blend
    grade["candidate"] = blend
    grade["candidate_members"] = [
        {**grade["candidate_members"][0], "run": "candidate_a", "epoch": 1, "weight": 0.25},
        {**grade["candidate_members"][0], "run": "candidate_b", "epoch": 2, "weight": 0.75},
    ]
    grade_path.write_text(json.dumps(grade))

    result = record_comparison_receipt(
        comparison,
        run_id="candidate_run",
        spec_bytes=spec_bytes,
        steps=[],
    )

    receipt = json.loads(Path(result["receipt_path"]).read_text())
    assert result["status"] == "valid"
    assert [member["weight"] for member in receipt["candidate"]["members"]] == [0.25, 0.75]


@pytest.mark.parametrize("weight", ["", "NaN", "inf", "not-a-number", "1e999"])
def test_malformed_or_nonfinite_blend_weights_fail(weight: str, tmp_path: Path) -> None:
    comparison, _, _, spec_bytes = _fixture(tmp_path)
    comparison["candidate"] = f"candidate_run@ep3*{weight}"

    with pytest.raises(ComparisonReceiptError, match="blend weight"):
        record_comparison_receipt(
            comparison,
            run_id="candidate_run",
            spec_bytes=spec_bytes,
            steps=[],
        )


def test_old_grade_fold_is_proved_by_split_manifest(tmp_path: Path) -> None:
    comparison, grade_path, _, spec_bytes = _fixture(tmp_path)
    grade = json.loads(grade_path.read_text())
    grade.pop("expected_fold")
    grade.pop("split_path")
    grade_path.write_text(json.dumps(grade))

    result = record_comparison_receipt(
        comparison,
        run_id="candidate_run",
        spec_bytes=spec_bytes,
        steps=[],
    )

    assert result["status"] == "valid"


def test_grade_fold_and_split_path_can_bind_manifest_without_fold_field(tmp_path: Path) -> None:
    comparison, grade_path, _, spec_bytes = _fixture(tmp_path)
    split_path = Path(comparison["split_path"])
    split_path.write_text(json.dumps({"evaluation": {"valid": [1, 2]}}))
    grade = json.loads(grade_path.read_text())
    grade["split_manifest_sha256"] = _sha(split_path.read_bytes())
    grade["candidate_members"][0]["split_manifest_sha256"] = grade["split_manifest_sha256"]
    grade["contrasts"][0]["reference_members"][0]["split_manifest_sha256"] = grade[
        "split_manifest_sha256"
    ]
    grade_path.write_text(json.dumps(grade))

    result = record_comparison_receipt(
        comparison,
        run_id="candidate_run",
        spec_bytes=spec_bytes,
        steps=[],
    )

    assert result["status"] == "valid"


def test_same_experiment_identity_does_not_double_append(tmp_path: Path) -> None:
    comparison, _, ledger_path, spec_bytes = _fixture(tmp_path)
    args = {
        "run_id": "candidate_run",
        "spec_bytes": spec_bytes,
        "steps": [{"name": "fit", "status": "succeeded", "wall_seconds": 1.0}],
    }
    first = record_comparison_receipt(comparison, **args)
    second = record_comparison_receipt(comparison, **args)

    assert first["ledger_appended"] is True
    assert second["ledger_appended"] is False
    assert len(ledger_path.read_text().splitlines()) == 1


def test_conflicting_experiment_identity_fails_without_replacing_receipt(tmp_path: Path) -> None:
    comparison, grade_path, ledger_path, spec_bytes = _fixture(tmp_path)
    args = {
        "run_id": "candidate_run",
        "spec_bytes": spec_bytes,
        "steps": [{"name": "fit", "status": "succeeded", "wall_seconds": 1.0}],
    }
    first = record_comparison_receipt(comparison, **args)
    receipt_path = Path(first["receipt_path"])
    prior_receipt = receipt_path.read_bytes()
    grade = json.loads(grade_path.read_text())
    grade["official"]["wp"] = 0.73
    grade["contrasts"][0]["paired"]["candidate_wp"] = 0.73
    grade["contrasts"][0]["paired"]["delta"] = 0.02
    grade_path.write_text(json.dumps(grade))

    with pytest.raises(ComparisonReceiptError, match="conflicting ledger row"):
        record_comparison_receipt(comparison, **args)

    assert receipt_path.read_bytes() == prior_receipt
    assert len(ledger_path.read_text().splitlines()) == 1


def test_missing_reference_fails_without_receipt_or_ledger(tmp_path: Path) -> None:
    comparison, grade_path, ledger_path, spec_bytes = _fixture(tmp_path)
    grade = json.loads(grade_path.read_text())
    grade["contrasts"] = []
    grade_path.write_text(json.dumps(grade))

    with pytest.raises(ComparisonReceiptError, match="missing declared reference"):
        record_comparison_receipt(
            comparison,
            run_id="candidate_run",
            spec_bytes=spec_bytes,
            steps=[],
        )

    assert not Path(comparison["receipt_path"]).exists()
    assert not ledger_path.exists()


def test_wrong_split_hash_and_nonfinite_metric_fail_closed(tmp_path: Path) -> None:
    comparison, grade_path, _, spec_bytes = _fixture(tmp_path)
    grade = json.loads(grade_path.read_text())
    grade["split_manifest_sha256"] = "d" * 64
    grade_path.write_text(json.dumps(grade))
    with pytest.raises(ComparisonReceiptError, match="split SHA256"):
        record_comparison_receipt(
            comparison,
            run_id="candidate_run",
            spec_bytes=spec_bytes,
            steps=[],
        )

    grade["split_manifest_sha256"] = _sha(Path(comparison["split_path"]).read_bytes())
    grade["expected_fold"] = 3
    grade_path.write_text(json.dumps(grade))
    with pytest.raises(ComparisonReceiptError, match="grade fold"):
        record_comparison_receipt(
            comparison,
            run_id="candidate_run",
            spec_bytes=spec_bytes,
            steps=[],
        )

    grade["expected_fold"] = 2
    grade["official"]["wp"] = float("nan")
    grade_path.write_text(json.dumps(grade))
    with pytest.raises(ComparisonReceiptError, match="finite number"):
        record_comparison_receipt(
            comparison,
            run_id="candidate_run",
            spec_bytes=spec_bytes,
            steps=[],
        )


def test_worker_fails_when_declared_prediction_artifact_is_missing(tmp_path: Path) -> None:
    comparison, _, ledger_path, _ = _fixture(tmp_path)
    queue = tmp_path / "queue"
    pending = queue / "pending"
    pending.mkdir(parents=True)
    spec = {
        "run_id": "candidate_run",
        "steps": [{"name": "complete", "argv": [sys.executable, "-c", "pass"]}],
        "expected_outputs": [str(tmp_path / "pred_candidate_run_valid.npy")],
        "comparison": comparison,
    }
    (pending / "candidate_run.json").write_text(json.dumps(spec))

    assert QueueWorker(queue, poll_seconds=0, once=True).run() == 0

    result = json.loads((queue / "results.jsonl").read_text().splitlines()[-1])
    assert result["status"] == "failed"
    assert result["steps"][-1]["name"] == "expected_outputs"
    assert (queue / "failed" / "candidate_run.json").exists()
    assert not Path(comparison["receipt_path"]).exists()
    assert not ledger_path.exists()


def test_worker_records_comparison_only_after_expected_outputs_validate(tmp_path: Path) -> None:
    comparison, grade_path, ledger_path, _ = _fixture(tmp_path)
    queue = tmp_path / "queue"
    pending = queue / "pending"
    pending.mkdir(parents=True)
    spec = {
        "run_id": "candidate_run",
        "ancestry": {"initializer": "declared_parent"},
        "exposure": "declared fold exposure",
        "steps": [{"name": "complete", "argv": [sys.executable, "-c", "pass"]}],
        "expected_outputs": [str(grade_path)],
        "comparison": comparison,
    }
    (pending / "candidate_run.json").write_text(json.dumps(spec))

    assert QueueWorker(queue, poll_seconds=0, once=True).run() == 0

    result = json.loads((queue / "results.jsonl").read_text().splitlines()[-1])
    receipt = json.loads(Path(comparison["receipt_path"]).read_text())
    assert result["status"] == "succeeded"
    assert result["comparison_receipt"]["ledger_appended"] is True
    assert receipt["declared_metadata"]["ancestry"] == spec["ancestry"]
    assert receipt["declared_metadata"]["exposure"] == spec["exposure"]
    assert len(ledger_path.read_text().splitlines()) == 1
    assert (queue / "done" / "candidate_run.json").exists()


def test_legacy_worker_spec_succeeds_without_comparison_changes(tmp_path: Path) -> None:
    queue = tmp_path / "queue"
    pending = queue / "pending"
    pending.mkdir(parents=True)
    spec = {
        "run_id": "legacy_run",
        "steps": [{"name": "complete", "argv": [sys.executable, "-c", "pass"]}],
    }
    (pending / "legacy_run.json").write_text(json.dumps(spec))

    assert QueueWorker(queue, poll_seconds=0, once=True).run() == 0

    result = json.loads((queue / "results.jsonl").read_text().splitlines()[-1])
    assert result["status"] == "succeeded"
    assert "comparison_receipt" not in result
    assert (queue / "done" / "legacy_run.json").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("valid_targets_sha256", None),
        ("ci95", None),
        ("ci95", [float("nan"), 0.1]),
        ("ci95", [0.1, -0.1]),
    ],
)
def test_receipt_rejects_missing_target_binding_or_invalid_uncertainty(tmp_path, field, value):
    comparison, grade_path, ledger_path, spec_bytes = _fixture(tmp_path)
    grade = json.loads(grade_path.read_text())
    target = grade if field == "valid_targets_sha256" else grade["contrasts"][0]["paired"]
    target[field] = value
    grade_path.write_text(json.dumps(grade))
    with pytest.raises(ComparisonReceiptError):
        record_comparison_receipt(comparison, run_id="run", spec_bytes=spec_bytes, steps=[])
    assert not ledger_path.exists()
