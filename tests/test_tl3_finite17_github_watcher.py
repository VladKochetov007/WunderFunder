from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts import tl3_finite17_github_posts as posts
from scripts import tl3_finite17_github_watcher as watcher
from scripts import tl3_finite17_results as results
from scripts.tl3_finite17_dashboard_core import _canonical_sha256
from tests.tl3_finite17_watcher_fixtures import FakeGitHub, _fixture


def test_watcher_posts_pr_specific_comments_and_recovers_pending_marker(tmp_path: Path) -> None:
    root, state_dir, plan, status, train, valid = _fixture(tmp_path)
    snapshot = results.build_snapshot(root, plan, status, train, valid)
    fake = FakeGitHub()
    state_path = tmp_path / "watcher_state.json"
    state: dict[str, Any] = {
        "posted_event_keys": [],
        "pending_event_keys": [],
        "posts": [],
    }
    key = "start:test"
    assert posts._post_event(
        fake,
        state_path,
        state,
        123,
        plan["sha256"],
        snapshot,
        key,
        "initial status",
        9999,
        1000,
    )
    assert fake.requests[0][0] == "POST"
    assert fake.requests[0][1] == "/repos/VladKochetov007/WunderFunder/issues/123/comments"
    marker = posts._event_marker(plan["sha256"], key)

    crashed_state = {
        "posted_event_keys": [],
        "pending_event_keys": [key],
        "posts": [],
    }
    assert posts._post_event(
        fake,
        state_path,
        crashed_state,
        123,
        plan["sha256"],
        snapshot,
        key,
        "initial status",
        9999,
        1001,
    )
    assert marker in fake.comments[0]["body"]
    assert len([request for request in fake.requests if request[0] == "POST"]) == 1
    assert fake.requests[-1][0] == "GET"


def test_watcher_full_terminal_cycle_writes_secret_free_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, state_dir, _plan, _status, _train, _valid = _fixture(tmp_path)
    output = tmp_path / "private"
    output.mkdir()
    fake = FakeGitHub()
    monkeypatch.setattr(watcher, "_private_directory", lambda _path: None)

    watcher._run_loop(
        root, state_dir, 123, output, fake, 60, 168, lambda: 1_000.0, lambda _seconds: None
    )

    state_text = (output / "watcher_state.json").read_text(encoding="utf-8")
    status_record = json.loads((output / "watcher_status.json").read_text(encoding="utf-8"))
    receipt = json.loads((output / "watcher_receipt.json").read_text(encoding="utf-8"))
    posts = [request for request in fake.requests if request[0] == "POST"]
    assert fake.account_checks == 1
    assert len(posts) == 3  # initial, verified fit, finite DAG terminal
    assert status_record["status"] == "terminal"
    assert receipt["pr_number"] == 123
    assert "secret" not in state_text.lower()
    assert "token" not in state_text.lower()


def test_hourly_event_requires_elapsed_hour_and_changed_snapshot() -> None:
    snapshot = {"driver_status": "running", "fits_used": 1}
    state = {
        "last_hourly_epoch": 10_000,
        "last_hourly_snapshot_sha256": _canonical_sha256(snapshot),
    }
    assert not posts._hourly_due(state, snapshot, 13_599)
    assert not posts._hourly_due(state, snapshot, 13_600)
    assert posts._hourly_due(state, {**snapshot, "fits_used": 2}, 13_600)


def test_changed_snapshots_in_consecutive_hours_get_distinct_posts(tmp_path: Path) -> None:
    root, _state_dir, plan, status, train, valid = _fixture(tmp_path)
    first = results.build_snapshot(root, plan, status, train, valid)
    second = {**first, "fits_used": first["fits_used"] + 1}
    fake = FakeGitHub()
    state: dict[str, Any] = {"posted_event_keys": [], "pending_event_keys": [], "posts": []}
    state_path = tmp_path / "watcher_state.json"

    for snapshot, now in ((first, 3_600.0), (second, 7_200.0)):
        key = posts._hourly_event_key(snapshot, now)
        assert posts._post_event(
            fake,
            state_path,
            state,
            123,
            plan["sha256"],
            snapshot,
            key,
            "hourly changed status",
            9_999,
            now,
        )

    assert len([request for request in fake.requests if request[0] == "POST"]) == 2
    assert state["posted_event_keys"][0] != state["posted_event_keys"][1]
