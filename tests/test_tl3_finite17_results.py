from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import tl3_finite17_results as results
from scripts.tl3_finite17_dashboard_core import _sha256_file
from tests.tl3_finite17_watcher_fixtures import _fixture


def test_results_include_only_source_bound_full_valid_and_train_diagnostic(tmp_path: Path) -> None:
    root, _state, plan, status, train, valid = _fixture(tmp_path)

    snapshot = results.build_snapshot(root, plan, status, train, valid)
    target = next(row for row in snapshot["nodes"] if row["id"] == "fin256_s4")
    text = results.format_update(snapshot, "initial")

    assert target["metrics"]["latest_valid_wp"] == pytest.approx(0.23)
    assert target["metrics"]["latest_valid_wp_t0"] == pytest.approx(0.18)
    assert target["metrics"]["train_eval_mask"] == "all_after_warmup"
    assert target["metrics"]["best_observed_valid_epoch"] == 4
    assert (
        target["metrics"]["fit_source_sha256"] == plan["source_identity"]["training_source_sha256"]
    )
    first = next(row for row in snapshot["nodes"] if row["id"] == "fin256_s3")
    assert first["metrics"] is None
    assert "not a leaderboard score" in text
    assert "is_scored" in text
    assert "Fit source SHA" in text


def test_results_reject_mislabelled_train_mask(tmp_path: Path) -> None:
    root, _state, plan, status, train, valid = _fixture(tmp_path)
    run_dir = Path(plan["nodes"][3]["run_directory"])
    history_path = run_dir / "history.json"
    history = json.loads(history_path.read_text(encoding="utf-8"))
    history[2]["train_eval_mask"] = "is_scored"
    history_path.write_text(json.dumps(history), encoding="utf-8")
    status["nodes"]["fin256_s4"]["verified_run"]["history_sha256"] = _sha256_file(history_path)

    with pytest.raises(results.BridgeError, match="train-eval cohort or mask"):
        results.build_snapshot(root, plan, status, train, valid)
