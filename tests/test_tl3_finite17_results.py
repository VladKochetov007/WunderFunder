from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import tl3_finite17_dashboard_final_validation as final_validation
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


def test_terminal_update_includes_only_the_verified_fixed_blend_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _state_dir, plan, status, train, valid = _fixture(tmp_path)
    status["status"] = "completed"
    report = {
        "blend": {"wp": 0.68, "t0": 0.67, "t1": 0.69, "selected_rows": 3_528_273},
        "run_weights": {
            "tl3c_w18final_distilled_s5": 0.34,
            "tl3c_fttall_tlead_s5": 0.33,
            "tl3c_fttall_tlead_s6": 0.33,
        },
        "clean_gru_reference_wp": 0.6778561232160607,
        "delta_to_clean_gru": 0.68 - 0.6778561232160607,
        "score_sha256": "f" * 64,
    }
    monkeypatch.setattr(final_validation, "validate_final_score", lambda *_args: report)

    snapshot = results.build_snapshot(root, plan, status, train, valid)
    terminal_text = results.format_update(snapshot, "DAG terminal")
    fit_text = results.format_update(snapshot, "fit complete", only_node="fin256_s4")

    assert (
        "WP 0.680000 (t0 0.670000, t1 0.690000) over 3,528,273 `is_scored` rows." in terminal_text
    )
    assert "`tl3c_w18final_distilled_s5` 0.34" in terminal_text
    assert "Delta to clean GRU reference (0.677856): +0.002144." in terminal_text
    assert "Terminal fixed blend" not in fit_text
