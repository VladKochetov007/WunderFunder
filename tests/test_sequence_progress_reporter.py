from __future__ import annotations

import base64
import io
import json
import urllib.error
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from scripts import report_sequence_progress as runner
from scripts.collect_sequence_monitor_r2 import collect_runs
from scripts.sequence_monitor_mirror_io import apply_snapshot
from scripts.sequence_progress_github import ACCOUNT, REPOSITORY, ProgressGitHub
from scripts.sequence_progress_report import SPLIT, canonical_sha, observe_host, render_report


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _fit(root: Path, host: str = "l4", complete: bool = False) -> Path:
    """Create synthetic receipt fixtures only; no model or scientific evaluation runs."""
    cell = "gru_s101001" if host == "l4" else "labels_s101001"
    folder = root / f"{host}-{cell}-fit"
    config = {
        "epochs": 12 if host == "l4" else 30,
        "batch_size": 16 if host == "l4" else 2,
        "chunk_rows": 2000,
        "accumulate_chunks": 1,
    }
    data = {
        "split_sha256": SPLIT,
        "fit_members": [["train", i] for i in range(10325)],
        "evaluation_members": [["valid", i] for i in range(1873)],
    }
    world = 1 if host == "l4" else 2
    metadata = {
        "source": {"sha256": "a" * 64},
        "config": config,
        "provenance": {"world_size": world, "data": data, "data_sha256": canonical_sha(data)},
    }
    _write(folder / "run_metadata.json", metadata)
    _write(
        folder / "monitor_progress.json",
        {
            "phase": "training",
            "epoch": 0.5,
            "step": 12,
            "last_successful_update": 12,
            "heartbeat_at": "2020-01-01T00:00:00Z",
        },
    )
    if complete:
        _terminal(folder, metadata, world, 77520 if host == "l4" else 774600)
    return folder


def _terminal(folder: Path, metadata: dict[str, object], world: int, updates: int) -> None:
    identity = {
        "source_sha256": "a" * 64,
        "config_sha256": canonical_sha(metadata["config"]),
        "data_sha256": metadata["provenance"]["data_sha256"],
        "world_size": world,
    }
    _write(
        folder / "terminal_status.json",
        {
            "status": "completed",
            "completed_epochs": metadata["config"]["epochs"],
            "global_updates": updates,
            "model_sha256": "b" * 64,
            "identity": identity,
        },
    )


def test_completion_requires_full_budget_identity_and_actual_final_endpoint(tmp_path: Path) -> None:
    folder = _fit(tmp_path, complete=True)
    row = observe_host(tmp_path, "l4", 1_700_000_000)["cells"][0]
    assert row["fit_complete"] and not row["endpoint_complete"]
    terminal = json.loads((folder / "terminal_status.json").read_bytes())
    event = {
        "schema": "wnn.monitor.v1",
        "kind": "evaluation",
        "epoch": 12,
        "step": 77520,
        "wall_time": "2020-01-01T00:00:00Z",
        "context": {
            "valid": {"split_sha256": SPLIT},
            "run_identity_sha256": canonical_sha(terminal["identity"]),
        },
        "metrics": {
            "valid/wp": 0.5,
            "valid/wp/t0": 0.4,
            "valid/wp/t1": 0.6,
            "valid/sequence_count": 1873,
            "valid/selected_rows": 3528273,
        },
    }
    (folder / "monitor_evaluations.jsonl").write_text(json.dumps(event) + "\n")
    assert observe_host(tmp_path, "l4", 1_700_000_000)["cells"][0]["endpoint_complete"]
    terminal["global_updates"] -= 1
    _write(folder / "terminal_status.json", terminal)
    assert not observe_host(tmp_path, "l4", 1_700_000_000)["cells"][0]["fit_complete"]
    terminal["global_updates"] += 1
    terminal["model_sha256"] = "not-a-sha"
    _write(folder / "terminal_status.json", terminal)
    assert not observe_host(tmp_path, "l4", 1_700_000_000)["cells"][0]["fit_complete"]


def test_b300_final_only_requires_explicit_full_fixed_blend_grade(tmp_path: Path) -> None:
    folder = _fit(tmp_path, "b300", complete=True)
    snapshot = observe_host(tmp_path, "b300", 1_700_000_000)
    assert snapshot["cells"][0]["fit_complete"] and not snapshot["cells"][0]["endpoint_complete"]
    metadata = json.loads((folder / "run_metadata.json").read_bytes())
    score = {
        "split_sha256": SPLIT,
        "sequences": 1873,
        "score": {
            "metric_id": "global_wp_clip2_v1",
            "selected_rows": 3528273,
            "wp": 0.5,
            "t0": 0.4,
            "t1": 0.6,
        },
        "producer": {
            "kind": "sequence-reference-blend-v1",
            "composition": {
                "method": "linear_fp32_no_clip",
                "alpha": 0.1,
                "candidate_checkpoint_sha256": "b" * 64,
            },
            "candidate": {"checkpoint_sha256": "b" * 64, "config": metadata["config"]},
        },
    }
    score["producer"]["composition"]["reference_sha256"] = "c" * 64
    score["producer"]["reference"] = {"reference_sha256": "c" * 64}
    score["producer"]["composition_sha256"] = canonical_sha(score["producer"]["composition"])
    _write(folder / "score.json", score)
    snapshot = observe_host(tmp_path, "b300", 1_700_000_000)
    assert snapshot["cells"][0]["endpoint_complete"]
    body = render_report(snapshot)
    assert "fixed blend grade" in body and "independent byte verification remain pending" in body
    assert "TL3 delta" not in body
    score["score"]["selected_rows"] -= 1
    _write(folder / "score.json", score)
    assert not observe_host(tmp_path, "b300", 1_700_000_000)["cells"][0]["endpoint_complete"]


def test_profiles_are_excluded_and_source_age_is_cached_not_generated_time(tmp_path: Path) -> None:
    folder = _fit(tmp_path, complete=True)
    folder.rename(folder.with_name(folder.name.removesuffix("fit") + "profile"))
    snapshot = observe_host(tmp_path, "l4", 1_700_000_000)
    assert not any(row["fit_complete"] for row in snapshot["cells"])
    folder = _fit(tmp_path)
    snapshot = observe_host(tmp_path, "l4", 1_700_000_000)
    assert snapshot["cells"][0]["source_age_seconds"] > 100_000_000
    assert "2020-01-01" in render_report(snapshot)
    _write(folder / "monitor_progress.json", {"heartbeat_at": "2099-01-01T00:00:00Z"})
    assert "source clock ahead" in render_report(observe_host(tmp_path, "l4", 1_700_000_000))


class FakeGitHub(ProgressGitHub):
    """Simulate the complete account/PR/comment API in memory, never using auth/network."""

    def __init__(self) -> None:
        super().__init__({67, 68})
        self.comments = []
        self.calls = []
        self.login = ACCOUNT
        self.lose_response = False

    def request(self, method: str, path: str, payload: object = None) -> object:
        self.calls.append((method, path))
        if path == "/user":
            return {"login": self.login}
        if "/pulls/" in path:
            return {
                "number": int(path.rsplit("/", 1)[1]),
                "base": {"repo": {"full_name": REPOSITORY}},
            }
        if method == "GET":
            return list(self.comments)
        comment = {
            "id": len(self.comments) + 1,
            "body": payload["body"],
            "user": {"login": ACCOUNT},
        }
        self.comments.append(comment)
        if self.lose_response:
            raise RuntimeError("GitHub POST request failed; response outcome may be uncertain")
        return comment


def test_restart_deduplicates_and_ambiguous_post_reconciles_marker(tmp_path: Path) -> None:
    root = tmp_path / "native"
    _fit(root)
    state, path, client = {"pull_requests": {}}, tmp_path / "github-report-state.json", FakeGitHub()
    client.lose_response = True
    runner.report_once(root, {"l4": 68}, state, path, client, 1_700_000_000)
    assert len(client.comments) == 1
    restarted = json.loads(path.read_bytes())
    client.lose_response = False
    runner.report_once(root, {"l4": 68}, restarted, path, client, 1_700_000_301)
    assert len(client.comments) == 1
    assert restarted["pull_requests"]["68"]["last_comment"]["kind"] == "reconciled"
    calls = len(client.calls)
    runner.report_once(root, {"l4": 68}, restarted, path, client, 1_700_000_400)
    assert len(client.calls) == calls
    assert "Authorization" not in path.read_text() and "WNN_GITHUB_BASIC" not in path.read_text()


def test_wrong_account_never_posts_and_unconfirmed_post_never_repeats(tmp_path: Path) -> None:
    root = tmp_path / "native"
    _fit(root)
    state, path, client = {"pull_requests": {}}, tmp_path / "github-report-state.json", FakeGitHub()
    client.login = "another-account"
    runner.report_once(root, {"l4": 68}, state, path, client, 1_700_000_000)
    assert not client.comments and all(method == "GET" for method, _ in client.calls)
    client.login, client.lose_response = ACCOUNT, True
    runner.report_once(root, {"l4": 68}, state, path, client, 1_700_000_031)
    client.comments.clear()  # An ambiguous request with no visible marker must fail closed.
    posts = sum(method == "POST" for method, _ in client.calls)
    runner.report_once(root, {"l4": 68}, state, path, client, 1_700_000_400)
    assert sum(method == "POST" for method, _ in client.calls) == posts


def test_periodic_and_terminal_updates_are_batched_without_epoch_spam(tmp_path: Path) -> None:
    root = tmp_path / "native"
    folder = _fit(root)
    state, path, client = {"pull_requests": {}}, tmp_path / "github-report-state.json", FakeGitHub()
    runner.report_once(root, {"l4": 68}, state, path, client, 1_700_000_000)
    _write(folder / "monitor_progress.json", {"phase": "evaluation", "epoch": 1, "step": 6460})
    runner.report_once(root, {"l4": 68}, state, path, client, 1_700_000_060)
    assert len(client.comments) == 1
    runner.report_once(root, {"l4": 68}, state, path, client, 1_700_001_801)
    assert len(client.comments) == 2
    metadata = json.loads((folder / "run_metadata.json").read_bytes())
    _terminal(folder, metadata, 1, 77520)
    runner.report_once(root, {"l4": 68}, state, path, client, 1_700_001_810)
    assert len(client.comments) == 3
    assert "1/8 fits complete" in client.comments[-1]["body"]


def test_transport_rejects_other_origins_and_never_echoes_credentials() -> None:
    client = ProgressGitHub({67})
    credential = base64.b64encode(b"fixture:fake-pat-never-real").decode()
    with patch("scripts.sequence_progress_github.os.environ", {"WNN_GITHUB_BASIC": credential}):
        with pytest.raises(ValueError, match="scope"):
            client.request("POST", "https://elsewhere.invalid/anything", {})
        client.opener = Mock()
        client.opener.open.side_effect = urllib.error.HTTPError(
            "https://api.github.com/user", 401, "failure", {}, io.BytesIO(b"fake-pat-never-real")
        )
        with pytest.raises(RuntimeError) as caught:
            client.verify_account()
        assert "fake-pat-never-real" not in str(caught.value) and credential not in str(
            caught.value
        )


def test_dry_run_never_constructs_api_client(tmp_path: Path) -> None:
    native = tmp_path / "native"
    _fit(native)
    config = tmp_path / "mirror.json"
    _write(config, {"native_root": str(native)})
    with (
        patch(
            "sys.argv",
            ["report", "--mirror-config", str(config), "--l4-pr", "68", "--dry-run", "--once"],
        ),
        patch.object(
            runner, "ProgressGitHub", side_effect=AssertionError("dry run attempted auth/network")
        ),
    ):
        assert runner.main() == 0
    assert (tmp_path / "github-report-pr68-dry-run.md").is_file()
    assert not (tmp_path / "github-report-state.json").exists()


def test_collector_r2_preserves_cursors_and_mirrors_native_score(tmp_path: Path) -> None:
    run = tmp_path / "remote" / "cell" / "fit"
    _write(run / "score.json", {"score": {"wp": 0.5}})
    snapshot = collect_runs(str(tmp_path / "remote"), "b300")
    cursors = apply_snapshot(snapshot, tmp_path / "view", tmp_path / "native", {}, "b300")
    assert (tmp_path / "view" / "b300-cell-fit" / "score.json").read_bytes() == (
        run / "score.json"
    ).read_bytes()
    again = collect_runs(str(tmp_path / "remote"), "b300", cursors)
    assert again["runs"][0]["files"][0]["bytes"] == 0
