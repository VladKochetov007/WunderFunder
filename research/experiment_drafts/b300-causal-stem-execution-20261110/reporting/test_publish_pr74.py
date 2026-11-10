from __future__ import annotations

from publish_pr74 import CELLS, _members_match, _report


def test_exact_sparse_training_membership_is_required() -> None:
    split = {"fit": {"train": [7, 2]}, "evaluation": {"valid": [13]}}
    exact = {"fit_members": [["train", 7], ["train", 2]], "evaluation_members": [["valid", 13]]}
    dense_range = {
        "fit_members": [["train", 0], ["train", 1]],
        "evaluation_members": [["valid", 13]],
    }

    assert _members_match(exact, split)
    assert not _members_match(dense_range, split)


def test_report_prefers_completed_rank0_step_to_snapshot_step() -> None:
    cells = [
        {
            "cell": name,
            "status": "running",
            "complete": False,
            "rank0": {"step": 817, "total_loss": 1.25, "gradient_norm_preclip": 0.5},
            "snapshot_step": 0,
        }
        for name in CELLS
    ]
    admission = {
        "admission_sha256": "a" * 64,
        "profiles": {name: {"pair": [0, 1]} for name in CELLS},
    }

    report = _report(cells, {"now": 1_800_000_000, "age": None, "errors": []}, admission, "fixture")

    assert "817/3,280 from rank0.jsonl; snapshot step 0" in report


def test_incomplete_terminal_without_model_sha_still_renders() -> None:
    cells = [{"cell": name, "status": "running", "complete": False} for name in CELLS]
    cells[0]["terminal"] = {"status": "failed", "model_sha256": None}
    admission = {
        "admission_sha256": "a" * 64,
        "profiles": {name: {"pair": [0, 1]} for name in CELLS},
    }

    report = _report(cells, {"now": 1_800_000_000, "age": None, "errors": []}, admission, "fixture")

    assert "failed (model SHA unavailable for incomplete terminal)" in report
