from __future__ import annotations

import copy
import json
import subprocess
import time
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import watch_heterogeneous_results as watcher
from scripts.heterogeneous_results_github import GitHub, PublicationError, publish
from scripts.heterogeneous_results_report import (
    INITIAL_INPUT_SHA256,
    NEW,
    SPLIT_SHA256,
    comment_body,
    read_json,
    ready_snapshot,
    report_ready,
    signature,
    validate_report,
)


def initial_report() -> dict:
    """Use the actual committed five-row report, without regrading model artifacts."""
    return read_json(
        Path(__file__).resolve().parents[1]
        / "research/experiment_drafts/heterogeneous-tl3-20261030/results/R1/REPORT.json"
    )


def test_measured_initial_report_and_missing_rows_are_preserved(tmp_path: Path) -> None:
    report = initial_report()
    assert report["input_manifest_sha256"] == INITIAL_INPUT_SHA256
    weights = validate_report(report)
    assert len(report_ready(report)) == 8
    later = {
        "common": [],
        "ready": {name: [[name, 1, 1]] for name in report_ready(report) | {"lstm"}},
    }
    assert signature(later, report_ready(report)) != signature(later)
    assert sum(row["status"] == "complete" for row in report["rows"]) == 5
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    body = comment_body(report, path, {"ready": {}}, {}, "result-test", False)
    assert "5/12" in body and "pending independent review" in body
    assert "unavailable: lstm" in body and "no public score" in body
    changed = copy.deepcopy(report)
    changed["rows"][0]["score"]["wp"] += 0.01
    with pytest.raises(ValueError, match="actual TL3 baseline"):
        validate_report(changed, weights)


def test_signature_ignores_nonfinal_epoch_updates(tmp_path: Path) -> None:
    root, legacy, galam = tmp_path / "root", tmp_path / "legacy", tmp_path / "galam"
    (root / "datasets").mkdir(parents=True)
    (root / "cache").mkdir()
    (root / "datasets/valid.parquet").write_bytes(b"source")
    (root / "cache/valid_index.npz").write_bytes(b"index")
    split = tmp_path / "split.json"
    split.write_text("{}")
    directory = root / "artifacts/runs" / NEW["lstm"]
    directory.mkdir(parents=True)
    (directory / "model.pt").write_bytes(b"checkpoint")
    bank = root / "cache" / f"pred_{NEW['lstm']}_valid.npy"
    bank.write_bytes(b"complete-bank-placeholder")
    side = {
        "run": NEW["lstm"],
        "split": "valid",
        "epoch": 3,
        "checkpoint_file": "model.pt",
        "inference_precision": "fp32",
        "split_manifest_sha256": SPLIT_SHA256,
        "prediction_sha256": "a" * 64,
    }
    bank.with_suffix(".json").write_text(json.dumps(side))
    provenance = {
        "selected_epoch": 0,
        "args": {"checkpoint_policy": "fixed", "checkpoint_epoch": 3},
    }
    path = directory / "provenance.json"
    path.write_text(json.dumps(provenance))
    before = ready_snapshot(root, legacy, galam, split)
    provenance["selected_epoch"] = 1
    path.write_text(json.dumps(provenance))
    assert signature(before) == signature(ready_snapshot(root, legacy, galam, split))
    provenance["selected_epoch"] = 3
    path.write_text(json.dumps(provenance))
    final = ready_snapshot(root, legacy, galam, split)
    assert signature(before) != signature(final) and "lstm" in final["ready"]


def test_successful_post_is_deduplicated_and_capped(tmp_path: Path) -> None:
    class Fake:
        calls = 0

        def post(self, body: str) -> str:
            self.calls += 1
            return "https://github.com/VladKochetov007/WunderFunder/pull/52#issuecomment-123"

    client = Fake()
    state = {"posts": {}, "posts_attempted": 0}
    path = tmp_path / "state.json"
    assert publish(client, state, path, "result-one", "body")
    assert publish(client, state, path, "result-one", "body")
    assert client.calls == 1 and state["posts_attempted"] == 1 and "pending_post" not in state
    state["posts_attempted"] = 24
    assert not publish(client, state, path, "result-two", "body")
    assert client.calls == 1


def test_first_post_failure_blocks_further_network_actions(tmp_path: Path) -> None:
    class Fake:
        calls = 0

        def post(self, body: str) -> str:
            self.calls += 1
            raise PublicationError("HTTP429 test; not retried")

    client = Fake()
    state = {"posts": {}, "posts_attempted": 0}
    path = tmp_path / "state.json"
    assert not publish(client, state, path, "result-one", "body")
    first = copy.deepcopy(state["publication_blocked"])
    assert not publish(client, state, path, "result-two", "body")
    assert client.calls == 1 and state["publication_blocked"] == first and state["pending_post"]


def test_http_rate_limit_has_one_attempt_and_no_credential_in_error() -> None:
    class Fake:
        calls = 0

        def open(self, request: object, timeout: int) -> None:
            self.calls += 1
            raise urllib.error.HTTPError("https://api.github.com/user", 429, "rate limit", {}, None)

    client = GitHub.__new__(GitHub)
    client.token, client.opener = "fake-unit-test-credential", Fake()
    with pytest.raises(PublicationError, match="HTTP429") as error:
        client.request("GET", "/user")
    assert client.opener.calls == 1 and client.token not in str(error.value)


def test_report_commands_are_cpu_artifact_grades_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = SimpleNamespace(
        root=tmp_path,
        legacy_root=tmp_path / "old",
        split=tmp_path / "split",
        galam_probe=tmp_path / "galam",
        tools=tmp_path,
        source=tmp_path / "source",
    )
    calls = []

    def fake(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        assert kwargs["env"]["OMP_NUM_THREADS"] == "1"
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(watcher.os, "sched_getaffinity", lambda pid: {58, 59}, raising=False)
    monkeypatch.setattr(watcher.subprocess, "run", fake)
    watcher.score(args, tmp_path / "auto-0001", time.time() + 60)
    assert [command[2] for command in calls] == [
        "scripts.prepare_heterogeneous_ensemble_inputs",
        "scripts.heterogeneous_ensemble_score",
    ]
    assert all("train_rnn" not in " ".join(command) for command in calls)
