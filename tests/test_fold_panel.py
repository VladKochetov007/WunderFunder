"""No-device checks for source-qualified cohort generation and grading."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts import score_fold0_panel, score_fold_panel
from scripts.make_cohort_specs import build_spec
from wnn.splits import CampaignSplit, file_sha256


@pytest.fixture
def fold3_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, CampaignSplit]:
    """Create a small, valid fold-3 producer and matching source-qualified membership."""
    root = tmp_path
    (root / "cache").mkdir()
    run = "candidate_s3_f3"
    run_root = root / "artifacts" / "runs" / run
    run_root.mkdir(parents=True)
    (root / "fold3.json").write_text(json.dumps({"fold": 3, "n_folds": 4}))
    split = CampaignSplit(
        root / "fold3.json",
        file_sha256(root / "fold3.json"),
        3,
        4,
        (0,),
        (0, 1, 2),
        (1, 2),
        (
            ("train", 0, 0),
            ("valid", 0, 1),
            ("valid", 1, 2),
            ("valid", 2, 3),
        ),
    )
    monkeypatch.setattr(score_fold_panel, "load_campaign_split", lambda _path, _root: split)

    rng = np.random.default_rng(44)
    targets = rng.normal(size=(3, 20_000, 2)).astype(np.float32)
    mask = np.ones((3, 20_000), dtype=bool)
    np.savez(
        root / "cache" / "valid_targets.npz", row_group=np.array([0, 1, 2]), y=targets, mask=mask
    )
    checkpoint = run_root / "model_ep2.pt"
    checkpoint.write_bytes(b"synthetic checkpoint bytes")
    provenance = {"fit_row_groups": {"train": [0], "valid": [0]}}
    (run_root / "provenance.json").write_text(json.dumps(provenance))
    prediction_path = root / "cache" / f"pred_{run}_valid.npy"
    prediction = np.zeros((3, 20_000, 2), dtype=np.float32)
    np.save(prediction_path, prediction)
    sidecar = {
        "run": run,
        "split": "valid",
        "epoch": 2,
        "split_manifest_sha256": split.sha256,
        "checkpoint_sha256": file_sha256(checkpoint),
        "row_groups": [0, 1, 2],
        "prediction_sha256": file_sha256(prediction_path),
        "inference_precision": "fp32",
    }
    prediction_path.with_suffix(".json").write_text(json.dumps(sidecar))
    return root, split


def test_generic_grader_scores_explicit_source_qualified_fold3(
    fold3_artifacts: tuple[Path, CampaignSplit],
) -> None:
    root, split = fold3_artifacts
    output_path = root / "grade.json"
    report = score_fold_panel.score_panel(
        "candidate_s3_f3@ep2",
        references=["candidate_s3_f3@ep2"],
        split_path=split.path,
        expected_fold=3,
        output_path=output_path,
        root=root,
        n_boot=8,
    )
    assert report["fold"] == 3
    assert report["expected_fold"] == 3
    assert report["evaluation_groups"] == 2
    assert report["components"] == 2
    assert report["metric_id"] == "global_wp_clip2_v1"
    assert len(report["contrasts"]) == 1
    assert json.loads(output_path.read_text())["fold"] == 3


def test_generic_grader_rejects_wrong_fold_and_evaluation_fit_membership(
    fold3_artifacts: tuple[Path, CampaignSplit],
) -> None:
    root, split = fold3_artifacts
    with pytest.raises(ValueError, match="declares fold 3, expected fold 1"):
        score_fold_panel.score_panel(
            "candidate_s3_f3@ep2",
            references=[],
            split_path=split.path,
            expected_fold=1,
            output_path=root / "wrong-fold.json",
            root=root,
        )

    provenance_path = root / "artifacts" / "runs" / "candidate_s3_f3" / "provenance.json"
    provenance_path.write_text(json.dumps({"fit_row_groups": {"train": [0], "valid": [1]}}))
    with pytest.raises(ValueError, match="evaluation groups occur in direct fit membership"):
        score_fold_panel.score_panel(
            "candidate_s3_f3@ep2",
            references=[],
            split_path=split.path,
            expected_fold=3,
            output_path=root / "membership.json",
            root=root,
        )


def test_generic_grader_requires_prediction_npy_before_grading(
    fold3_artifacts: tuple[Path, CampaignSplit],
) -> None:
    root, split = fold3_artifacts
    (root / "cache" / "pred_candidate_s3_f3_valid.npy").unlink()
    with pytest.raises(FileNotFoundError, match="expected prediction array is missing"):
        score_fold_panel.score_panel(
            "candidate_s3_f3@ep2",
            references=[],
            split_path=split.path,
            expected_fold=3,
            output_path=root / "missing-prediction.json",
            root=root,
        )


def test_generic_grader_rejects_checkpoint_mismatch_in_prediction_sidecar(
    fold3_artifacts: tuple[Path, CampaignSplit],
) -> None:
    root, split = fold3_artifacts
    sidecar_path = root / "cache" / "pred_candidate_s3_f3_valid.json"
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["checkpoint_sha256"] = "0" * 64
    sidecar_path.write_text(json.dumps(sidecar))
    with pytest.raises(ValueError, match="prediction sidecar disagrees with the fixed checkpoint"):
        score_fold_panel.score_panel(
            "candidate_s3_f3@ep2",
            references=[],
            split_path=split.path,
            expected_fold=3,
            output_path=root / "bad-sidecar.json",
            root=root,
        )


def test_legacy_fold0_cli_keeps_epoch_and_group_guard(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    captured: dict[str, object] = {}

    def fake_score_panel(candidate: str, **kwargs: object) -> dict[str, object]:
        captured["candidate"] = candidate
        captured.update(kwargs)
        return {"official": {"wp": 0.5}}

    monkeypatch.setattr(score_fold0_panel, "score_panel", fake_score_panel)
    monkeypatch.setattr(
        "sys.argv",
        ["score_fold0_panel.py", "legacy_run", "--epoch", "4", "--out", "grade.json"],
    )
    score_fold0_panel.main()

    assert captured["candidate"] == "legacy_run"
    assert captured["expected_fold"] == 0
    assert captured["expected_group_count"] == 468
    assert captured["candidate_epoch"] == 4
    assert captured["include_fold_metadata"] is False
    assert capsys.readouterr().out.startswith('{"candidate": "legacy_run"')


def _cohort() -> tuple[dict[str, object], dict[str, object]]:
    """Return a compact four-fold cohort input with source and comparison fields."""
    cohort: dict[str, object] = {
        "seed": 3,
        "fold": 3,
        "run_prefix": "synthetic",
        "trainer": "scripts/train_rnn.py",
        "fit_timeout": 3600,
        "split_manifest": "research/splits/fold3.json",
        "experiment_id": "E-FOLD3",
        "hypothesis_id": "H-FOLD3",
        "score_references": ["control_s3_f3@ep2"],
        "receipt_path": "research/receipts/{seed}.json",
        "ledger_path": "research/experiments.jsonl",
        "fidelity": {"stage": "screen"},
        "base_argv": [
            "--checkpoint-epoch",
            "2",
            "--valid-fold",
            "3",
            "--n-valid-folds",
            "4",
            "--split-manifest",
            "{root}/research/splits/fold3.json",
        ],
    }
    arm: dict[str, object] = {"id": "A01", "diff": {"--hidden": 64}}
    return cohort, arm


def _prepare_cohort_root(tmp_path: Path, split_fold: int = 3) -> Path:
    """Create a minimal declared manifest for a cohort-generation test."""
    root = tmp_path / "repo"
    split_path = root / "research" / "splits" / "fold3.json"
    split_path.parent.mkdir(parents=True)
    split_path.write_text(json.dumps({"fold": split_fold, "n_folds": 4}))
    return root


def test_cohort_spec_binds_fold_comparison_and_prediction_output(tmp_path: Path) -> None:
    cohort, arm = _cohort()
    root = _prepare_cohort_root(tmp_path)
    spec = build_spec(cohort, arm, str(root), "/usr/bin/python3", True)
    assert spec["run_id"] == "synthetic_a01_s3_f3"
    assert spec["comparison"] == {
        "experiment_id": "E-FOLD3",
        "hypothesis_id": "H-FOLD3",
        "candidate": "synthetic_a01_s3_f3@ep2",
        "references": ["control_s3_f3@ep2"],
        "grade_path": f"{root}/artifacts/runs/synthetic_a01_s3_f3/grade_fold3.json",
        "receipt_path": f"{root}/research/receipts/3.json",
        "ledger_path": f"{root}/research/experiments.jsonl",
        "split_path": f"{root}/research/splits/fold3.json",
        "fold": 3,
        "expected_fold": 3,
        "fidelity": {"stage": "screen"},
    }
    outputs = spec["expected_outputs"]
    assert f"{root}/cache/pred_synthetic_a01_s3_f3_valid.npy" in outputs
    assert f"{root}/cache/pred_synthetic_a01_s3_f3_valid.json" in outputs
    score_argv = spec["steps"][2]["argv"]
    assert "score_fold_panel.py" in score_argv[1]
    assert score_argv[score_argv.index("--expected-fold") + 1] == "3"
    assert not any("receipt" in output for output in outputs)


def test_cohort_spec_keeps_legacy_schema_and_rejects_conflicting_flags(
    tmp_path: Path,
) -> None:
    cohort, arm = _cohort()
    legacy = cohort | {
        "fold": 0,
        "split_manifest": None,
        "experiment_id": None,
        "base_argv": ["--checkpoint-epoch", "1"],
        "score_references": [],
    }
    root = _prepare_cohort_root(tmp_path)
    spec = build_spec(legacy, arm, str(root), "/usr/bin/python3", True)
    assert "comparison" not in spec
    assert "score_fold0_panel.py" in spec["steps"][2]["argv"][1]
    assert f"{root}/cache/pred_{spec['run_id']}_valid.npy" in spec["expected_outputs"]

    conflicting_fold = cohort | {
        "base_argv": cohort["base_argv"][:-7]
        + [
            "--valid-fold",
            "1",
            "--n-valid-folds",
            "4",
            "--split-manifest",
            "{root}/research/splits/fold3.json",
        ]
    }
    with pytest.raises(ValueError, match="fold argument disagrees"):
        build_spec(conflicting_fold, arm, str(root), "/usr/bin/python3", True)

    conflicting_split = cohort | {
        "base_argv": cohort["base_argv"][:-1] + ["{root}/research/splits/fold2.json"]
    }
    with pytest.raises(ValueError, match="split path disagrees"):
        build_spec(conflicting_split, arm, str(root), "/usr/bin/python3", True)

    wrong_manifest = _prepare_cohort_root(tmp_path / "wrong", split_fold=2)
    with pytest.raises(ValueError, match="manifest fold disagrees"):
        build_spec(cohort, arm, str(wrong_manifest), "/usr/bin/python3", True)


def test_cohort_spec_rejects_malformed_score_references(tmp_path: Path) -> None:
    cohort, arm = _cohort()
    root = _prepare_cohort_root(tmp_path)
    cohort["score_references"] = ["control_s3_f3"]
    with pytest.raises(ValueError, match="malformed score reference"):
        build_spec(cohort, arm, str(root), "/usr/bin/python3", True)

    cohort["score_references"] = ["control_s3_f3@ep2*nan"]
    with pytest.raises(ValueError, match="non-finite score reference weight"):
        build_spec(cohort, arm, str(root), "/usr/bin/python3", True)


@pytest.mark.parametrize(
    "missing", ["hypothesis_id", "receipt_path", "ledger_path", "score_references"]
)
def test_cohort_rejects_incomplete_comparison_before_fit(tmp_path: Path, missing: str) -> None:
    cohort, arm = _cohort()
    root = _prepare_cohort_root(tmp_path)
    cohort.pop(missing)
    with pytest.raises(ValueError, match="comparison requires"):
        build_spec(cohort, arm, str(root), "/usr/bin/python3", True)


def test_cohort_rejects_disabled_references_and_invalid_fidelity(tmp_path: Path) -> None:
    cohort, arm = _cohort()
    root = _prepare_cohort_root(tmp_path)
    with pytest.raises(ValueError, match="comparison requires"):
        build_spec(cohort, arm, str(root), "/usr/bin/python3", False)
    cohort["fidelity"] = "screen"
    with pytest.raises(ValueError, match="fidelity must be an object"):
        build_spec(cohort, arm, str(root), "/usr/bin/python3", True)


def test_cohort_checks_argparse_equals_form(tmp_path: Path) -> None:
    cohort, arm = _cohort()
    root = _prepare_cohort_root(tmp_path)
    cohort["base_argv"] = [
        "--checkpoint-epoch=2",
        "--valid-fold=1",
        "--n-valid-folds=4",
        "--split-manifest={root}/research/splits/fold3.json",
    ]
    with pytest.raises(ValueError, match="fold argument disagrees"):
        build_spec(cohort, arm, str(root), "/usr/bin/python3", True)
    cohort["base_argv"][1] = "--valid-fold=3"
    spec = build_spec(cohort, arm, str(root), "/usr/bin/python3", True)
    assert spec["comparison"]["split_path"] == str(root / "research/splits/fold3.json")


def test_real_grade_blend_roundtrips_to_idempotent_receipt(
    fold3_artifacts: tuple[Path, CampaignSplit],
) -> None:
    from scripts.comparison_receipt import record_comparison_receipt

    root, split = fold3_artifacts
    candidate = "candidate_s3_f3@ep2*0.25,candidate_s3_f3@ep2*0.75"
    reference = "candidate_s3_f3@ep2"
    grade_path = root / "grade.json"
    grade = score_fold_panel.score_panel(
        candidate,
        references=[reference],
        split_path=split.path,
        expected_fold=3,
        output_path=grade_path,
        root=root,
        n_boot=8,
    )
    comparison = {
        "experiment_id": "E-PRODUCER-CONSUMER",
        "hypothesis_id": "H-FIXTURE",
        "candidate": candidate,
        "references": [reference],
        "expected_fold": 3,
        "split_path": str(split.path),
        "grade_path": str(grade_path),
        "receipt_path": str(root / "receipt.json"),
        "ledger_path": str(root / "comparisons.jsonl"),
        "fidelity": {"stage": "synthetic_cpu_integration"},
    }
    kwargs = dict(run_id="fixture", spec_bytes=b"fixed-spec", steps=[], base_dir=root)
    result = record_comparison_receipt(comparison, **kwargs)
    assert result["ledger_appended"] is True
    receipt = json.loads((root / "receipt.json").read_text())
    assert receipt["candidate"]["wp"] == grade["official"]["wp"]
    assert receipt["grade_metadata"]["valid_targets_sha256"] == grade["valid_targets_sha256"]
    assert receipt["references"][0]["paired_delta"] == 0.0
    assert record_comparison_receipt(comparison, **kwargs)["ledger_appended"] is False
    assert len((root / "comparisons.jsonl").read_text().splitlines()) == 1
