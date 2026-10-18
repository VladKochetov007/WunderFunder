"""Small CPU-only contract tests for PR15 progress accounting."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))
import collect_pr15  # noqa: E402


def _jobs() -> list[dict[str, object]]:
    jobs = []
    for arm, control in collect_pr15.ARMS.items():
        for fold in range(4):
            stop = 1009 if fold < 2 else 1008
            for seed in range(1001, stop):
                jobs.append(
                    {
                        "id": f"pr15_b300_v1_{arm}_f{fold}_s{seed}",
                        "arm": arm,
                        "fold": fold,
                        "seed": seed,
                        "control": control,
                    }
                )
    return jobs


def test_panel_is_exactly_120_paired_cells_with_30_per_arm() -> None:
    assert collect_pr15.ARMS == {
        "baseline": "all",
        "moments": "moments_only",
        "dictionary": "dictionary_only",
        "combined": "none",
    }
    by_id = collect_pr15._panel(_jobs())
    assert len(by_id) == 120
    for arm in collect_pr15.ARMS:
        assert sum(job["arm"] == arm for job in by_id.values()) == 30


def test_panel_rejects_duplicate_or_drifted_cell_identity() -> None:
    jobs = _jobs()
    jobs[-1] = dict(jobs[0])
    with pytest.raises(ValueError, match="duplicate or drifted"):
        collect_pr15._panel(jobs)

    jobs = _jobs()
    jobs[0]["control"] = "none"
    with pytest.raises(ValueError, match="drifted"):
        collect_pr15._panel(jobs)


def test_progress_counts_only_started_unfinished_jobs_as_active() -> None:
    ids = {f"cell-{index}" for index in range(8)}
    events = [
        {"event": "fit_started", "job": {"id": "cell-0"}},
        {"event": "fit_started", "job": {"id": "cell-1"}},
        {"event": "fit_started", "job": {"id": "cell-2"}},
        {"event": "process_finished", "process": "cell-0", "status": "completed"},
        {"event": "admission_failed", "job": {"id": "cell-3"}, "error": "cap"},
    ]
    active, pending, errors, started, ended = collect_pr15._progress(events, ids)
    assert (active, pending) == (2, 5)
    assert started == {"cell-0", "cell-1", "cell-2"}
    assert ended == {"cell-0"}
    assert errors == ["admission failure cell-3: cap"]


def test_unknown_start_does_not_change_active_or_pending_counts() -> None:
    active, pending, errors, started, ended = collect_pr15._progress(
        [{"event": "fit_started", "job": {"id": "foreign"}}], {"cell"}
    )
    assert (active, pending, started, ended) == (0, 1, set(), set())
    assert errors == ["unexpected/duplicate start: foreign"]


def test_wp_requires_a_finite_official_score(tmp_path: Path) -> None:
    endpoint = tmp_path / "endpoint.json"
    endpoint.write_text("{}", encoding="utf-8")
    job = {"id": "cell", "arm": "baseline", "fold": 0, "seed": 1001}
    full = {
        "checkpoint": {
            "checkpoint_sha256": "a" * 64,
            "checkpoint_identity": {"payload_sha256": "b" * 64},
        }
    }
    score = {
        "score": {"wp": 0.61},
        "prediction_sha256": "c" * 64,
        "score_sha256": "d" * 64,
        "prediction_metadata_sha256": "e" * 64,
    }
    assert collect_pr15._row(job, endpoint, full, score, {}, ([], []), True)["wp"] == pytest.approx(
        0.61
    )
    with pytest.raises(ValueError, match="non-finite"):
        collect_pr15._row(
            job, endpoint, full, {**score, "score": {"wp": float("nan")}}, {}, ([], []), True
        )


def test_archive_paths_include_sidecar_and_weights_but_not_prediction_array(tmp_path: Path) -> None:
    work = tmp_path / "state/work/cell"
    endpoint = tmp_path / "state/endpoints/cell.json"
    run = work / "artifacts/runs/cell"
    metadata = work / "cache/pred_cell_valid.json"
    prediction = work / "cache/pred_cell_valid.npy"
    score = run / "score.json"
    for path in (metadata, prediction, score):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    endpoint.parent.mkdir(parents=True, exist_ok=True)
    endpoint.write_text("{}", encoding="utf-8")
    for name in (
        "model_ep2.pt",
        "model.pt",
        "training_state.pt",
        "history.json",
        "provenance.json",
    ):
        path = run / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")

    evidence, weights = collect_pr15._paths(
        tmp_path,
        work,
        endpoint,
        {"score_path": str(score), "prediction_metadata_path": str(metadata)},
    )
    assert str(metadata.resolve()) in evidence
    assert str(prediction.resolve()) not in evidence
    assert {Path(path).name for path in weights} == {
        "model_ep2.pt",
        "model.pt",
        "training_state.pt",
    }
