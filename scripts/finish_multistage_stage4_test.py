from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from finish_multistage_stage4 import history_snapshot  # noqa: E402


def _database(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            CREATE TABLE studies (study_id INTEGER, study_name TEXT);
            CREATE TABLE trials (trial_id INTEGER, number INTEGER, study_id INTEGER, state TEXT);
            CREATE TABLE trial_values (id INTEGER, trial_id INTEGER, value REAL);
            CREATE TABLE trial_params (id INTEGER, trial_id INTEGER, param_name TEXT, param_value REAL);
            CREATE TABLE trial_user_attributes (id INTEGER, trial_id INTEGER, key TEXT, value_json TEXT);
            CREATE TABLE trial_system_attributes (id INTEGER, trial_id INTEGER, key TEXT, value_json TEXT);
            INSERT INTO studies VALUES (1, 'original');
            INSERT INTO trial_user_attributes VALUES (1, 1, 'search_role', '"convex_weighted_mse_warmstart"');
        """)
        for number in range(23):
            connection.execute("INSERT INTO trials VALUES (?,?,1,'COMPLETE')", (number + 1, number))
            connection.execute(
                "INSERT INTO trial_values VALUES (?,?,?)", (number + 1, number + 1, 0.4)
            )
            connection.execute(
                "INSERT INTO trial_params VALUES (?,?,'logit_0',0.5)", (number + 1, number + 1)
            )


def test_same_study_one_new_trial_preserves_original_history(tmp_path: Path) -> None:
    path = tmp_path / "optuna.db"
    _database(path)
    before = history_snapshot(path, "original", 23)
    with sqlite3.connect(path) as connection:
        connection.execute("INSERT INTO trials VALUES (24,23,1,'COMPLETE')")
        connection.execute("INSERT INTO trial_values VALUES (24,24,0.45)")
        connection.execute("INSERT INTO trial_params VALUES (24,24,'logit_0',0.8)")
    after = history_snapshot(path, "original", 24)
    assert before["original_23_sha256"] == after["original_23_sha256"]
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE trial_params SET param_value=0.9 WHERE trial_id=1")
    assert (
        history_snapshot(path, "original", 24)["original_23_sha256"] != before["original_23_sha256"]
    )


@pytest.mark.parametrize("state", ["WAITING", "RUNNING", "PRUNED", "FAIL"])
def test_recovery_refuses_any_noncomplete_trial(tmp_path: Path, state: str) -> None:
    path = tmp_path / "optuna.db"
    _database(path)
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE trials SET state=? WHERE number=22", (state,))
    with pytest.raises(ValueError, match="23 COMPLETE trials and zero others"):
        history_snapshot(path, "original", 23)


def test_recovery_refuses_an_extra_trial_and_missing_warmstart(tmp_path: Path) -> None:
    path = tmp_path / "optuna.db"
    _database(path)
    with pytest.raises(ValueError, match="24 COMPLETE"):
        history_snapshot(path, "original", 24)
    with sqlite3.connect(path) as connection:
        connection.execute("DELETE FROM trial_user_attributes")
    with pytest.raises(ValueError, match="warmstart must already be COMPLETE"):
        history_snapshot(path, "original", 23)
