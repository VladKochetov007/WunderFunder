from __future__ import annotations

import base64
import io
import json
import shutil
import urllib.error
from collections.abc import Callable
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from scripts import report_sequence_progress as runner
from scripts.collect_sequence_monitor_r2 import collect_runs
from scripts.sequence_monitor_mirror_io import apply_snapshot
from scripts.sequence_progress_github import ACCOUNT, REPOSITORY, ProgressGitHub
from scripts.sequence_progress_panel import load_panel_manifest
from scripts.sequence_progress_report import (
    CELLS,
    SPLIT,
    canonical_sha,
    observe_host,
    render_report,
)


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


def _optimized_panel(root: Path) -> tuple[Path, Path, dict[str, object]]:
    folder = _fit(root / "fixture", "b300")
    folder = folder.rename(root / "b300-opt-labels_s101001-fit")
    metadata = json.loads((folder / "run_metadata.json").read_bytes())
    metadata["config"].update(batch_size=32, chunk_rows=256, accumulate_chunks=4)
    _write(folder / "run_metadata.json", metadata)
    updates = 30 * 162 * 20
    _terminal(folder, metadata, 2, updates)
    path = root.parent / "panel.json"
    _write(
        path,
        {
            "schema": "wnn.sequence-progress-panel.v1",
            "panels": {
                "b300": {
                    "run_prefix": "b300-opt",
                    "epochs": 30,
                    "source_sha256": "a" * 64,
                    "split_sha256": SPLIT,
                    "cells": {
                        cell: {
                            "config_sha256": canonical_sha(metadata["config"]),
                            "expected_updates": updates,
                        }
                        for cell in CELLS["b300"]
                    },
                }
            },
        },
    )
    return folder, path, load_panel_manifest(path)


def test_opt_panel_binds_new_budget_without_loosening_original_default(tmp_path: Path) -> None:
    native = tmp_path / "native"
    _fit(native, "b300", complete=True)
    folder, path, panels = _optimized_panel(native)
    optimized = observe_host(native, "b300", 1_700_000_000, panels["b300"])
    original = observe_host(native, "b300", 1_700_000_000)
    assert optimized["cells"][0]["fit_complete"]
    assert optimized["cells"][0]["budget_updates"] == 97200
    assert original["cells"][0]["budget_updates"] == 774600
    assert optimized["terminal_signature"] != original["terminal_signature"]
    assert "b300-opt" in render_report(optimized)
    assert panels["b300"]["manifest_sha256"] in render_report(optimized)
    # Neither a convenient terminal count nor stale source/config identities can pass.
    for key, replacement in (("batch_size", 64), ("epochs", 12)):
        metadata = json.loads((folder / "run_metadata.json").read_bytes())
        previous = metadata["config"][key]
        metadata["config"][key] = replacement
        _write(folder / "run_metadata.json", metadata)
        assert not observe_host(native, "b300", 1_700_000_000, panels["b300"])["cells"][0][
            "fit_complete"
        ]
        metadata["config"][key] = previous
        _write(folder / "run_metadata.json", metadata)
    metadata["source"]["sha256"] = "c" * 64
    _write(folder / "run_metadata.json", metadata)
    assert not observe_host(native, "b300", 1_700_000_000, panels["b300"])["cells"][0][
        "fit_complete"
    ]
    # Loaded identity stays frozen even if the operator edits the file during execution.
    path.write_text("{}")
    assert panels["b300"]["cells"]["labels_s101001"]["expected_updates"] == 97200


@pytest.mark.parametrize(
    "change",
    [
        lambda panel: panel.update(run_prefix="../b300"),
        lambda panel: panel.update(epochs=12),
        lambda panel: panel.update(split_sha256="c" * 64),
        lambda panel: panel["cells"].pop("complement_s101002"),
        lambda panel: panel["cells"]["labels_s101001"].update(expected_updates=True),
        lambda panel: panel["cells"]["labels_s101001"].update(config_sha256="bad"),
    ],
)
def test_panel_manifest_rejects_unbound_recipe(
    tmp_path: Path, change: Callable[[dict[str, object]], None]
) -> None:
    _, path, _ = _optimized_panel(tmp_path / "native")
    value = json.loads(path.read_bytes())
    change(value["panels"]["b300"])
    _write(path, value)
    with pytest.raises(ValueError):
        load_panel_manifest(path)


def test_panel_transition_preserves_old_report_and_pending_reconciliation(tmp_path: Path) -> None:
    native = tmp_path / "native"
    _fit(native, "b300", complete=True)
    state, path, client = {"pull_requests": {}}, tmp_path / "github-report-state.json", FakeGitHub()
    client.lose_response = True
    runner.report_once(native, {"b300": 67}, state, path, client, 1_700_000_000)
    _, _, panels = _optimized_panel(native)
    # Reconcile an earlier recipe's ambiguous write before publishing a new recipe.
    client.lose_response = False
    runner.report_once(native, {"b300": 67}, state, path, client, 1_700_000_301, panels=panels)
    assert len(client.comments) == 1
    runner.report_once(native, {"b300": 67}, state, path, client, 1_700_000_302, panels=panels)
    assert len(client.comments) == 2
    entry = state["pull_requests"]["67"]
    history = entry["last_comments_by_panel"]
    assert history["original"]["comment_id"] == 1
    assert history[panels["b300"]["manifest_sha256"]]["comment_id"] == 2
    assert "b300-opt" in client.comments[-1]["body"]
    restarted = json.loads(path.read_bytes())
    runner.report_once(native, {"b300": 67}, restarted, path, client, 1_700_000_303, panels=panels)
    assert len(client.comments) == 2


def test_panel_manifest_cli_dry_run_remains_credential_free(tmp_path: Path) -> None:
    native = tmp_path / "native"
    _, panel, _ = _optimized_panel(native)
    config = tmp_path / "mirror.json"
    _write(config, {"native_root": str(native)})
    with (
        patch(
            "sys.argv",
            [
                "report",
                "--mirror-config",
                str(config),
                "--b300-pr",
                "69",
                "--panel-manifest",
                str(panel),
                "--dry-run",
                "--once",
            ],
        ),
        patch.object(runner, "ProgressGitHub", side_effect=AssertionError("unexpected network")),
    ):
        assert runner.main() == 0
    assert "b300-opt" in (tmp_path / "github-report-pr69-dry-run.md").read_text()


def _l4_continuations(path: Path, config_sha: str) -> dict[str, object]:
    value = {
        "schema": "wnn.sequence-progress-panel.v1",
        "panels": {},
        "l4_continuations": {
            f"gru_s{seed}": {
                "run_id": f"l4-gru_s{seed}_monitor_opt-fit",
                "source_sha256": "c" * 64,
                "config_sha256": config_sha,
                "expected_updates": 77520,
                "epochs": 12,
            }
            for seed in (101001, 101002)
        },
    }
    _write(path, value)
    return value


def test_l4_continuation_manifest_refuses_other_aliases_digests_and_budgets(tmp_path: Path) -> None:
    path = tmp_path / "panel.json"
    value = _l4_continuations(path, "a" * 64)
    assert set(load_panel_manifest(path)) == {"l4"}
    for seed in (101001, 101002):
        value["l4_continuations"][f"gru_s{seed}"]["run_id"] = f"l4-gru_s{seed}_monitor_opt_r2-fit"
    _write(path, value)
    assert load_panel_manifest(path)["l4"]["continuations"] == value["l4_continuations"]
    for seed in (101001, 101002):
        value["l4_continuations"][f"gru_s{seed}"]["run_id"] = f"l4-gru_s{seed}_monitor_opt_r3-fit"
    _write(path, value)
    assert load_panel_manifest(path)["l4"]["continuations"] == value["l4_continuations"]
    for field, replacement in (
        ("run_id", "b300-opt-gru_s101001-fit"),
        ("run_id", "l4-gru_s101002_monitor_opt-fit"),
        ("run_id", "l4-gru_s101002_monitor_opt_r2-fit"),
        ("run_id", "l4-gru_s101002_monitor_opt_r3-fit"),
        ("run_id", "l4-gru_s101001_monitor_opt_r4-fit"),
        ("run_id", "../l4-gru_s101001_monitor_opt-fit"),
        ("source_sha256", "bad"),
        ("config_sha256", "bad"),
        ("expected_updates", 77519),
        ("epochs", 13),
    ):
        changed = json.loads(json.dumps(value))
        changed["l4_continuations"]["gru_s101001"][field] = replacement
        _write(path, changed)
        with pytest.raises(ValueError):
            load_panel_manifest(path)
    value["l4_continuations"]["mamba2_s101001"] = value["l4_continuations"]["gru_s101001"]
    _write(path, value)
    with pytest.raises(ValueError):
        load_panel_manifest(path)


def test_l4_exact_continuation_preserves_other_rows_history_and_final_identity(
    tmp_path: Path,
) -> None:
    native = tmp_path / "native"
    original = _fit(native, complete=True)
    shutil.copytree(original, native / "l4-gru_local_s101001-fit")
    continuation = native / "l4-gru_s101001_monitor_opt-fit"
    shutil.copytree(original, continuation)
    metadata = json.loads((continuation / "run_metadata.json").read_bytes())
    metadata["source"]["sha256"] = "c" * 64
    metadata["config"].update(seed=101001, model={"architecture": "gru"})
    _write(continuation / "run_metadata.json", metadata)
    _terminal(continuation, metadata, 1, 77520)
    terminal = json.loads((continuation / "terminal_status.json").read_bytes())
    terminal["identity"]["source_sha256"] = "c" * 64
    _write(continuation / "terminal_status.json", terminal)
    path = tmp_path / "panel.json"
    _l4_continuations(path, canonical_sha(metadata["config"]))
    panels = load_panel_manifest(path)
    default = observe_host(native, "l4", 1_700_000_001)
    selected = observe_host(native, "l4", 1_700_000_001, panels["l4"])
    assert len(selected["cells"]) == 8 and selected["cells"][2:] == default["cells"][2:]
    assert selected["cells"][0]["ids"]["source"] == "c" * 64
    assert default["cells"][0]["ids"]["source"] == "a" * 64
    assert selected["cells"][0]["fit_complete"] and not selected["cells"][0]["endpoint_complete"]
    event = {
        "kind": "evaluation",
        "epoch": 12,
        "step": 77520,
        "context": {"split_sha256": SPLIT, "run_identity_sha256": "wrong-source"},
        "metrics": {
            "valid/sequence_count": 1873,
            "valid/selected_rows": 3528273,
            "valid/wp": 0.5,
            "valid/wp/t0": 0.4,
            "valid/wp/t1": 0.6,
        },
    }
    journal = continuation / "monitor_evaluations.jsonl"
    journal.write_text(json.dumps(event) + "\n")
    assert not observe_host(native, "l4", 1_700_000_001, panels["l4"])["cells"][0][
        "endpoint_complete"
    ]
    event["context"]["run_identity_sha256"] = canonical_sha(terminal["identity"])
    journal.write_text(json.dumps(event) + "\n")
    assert observe_host(native, "l4", 1_700_000_001, panels["l4"])["cells"][0]["endpoint_complete"]
    state, state_path, client = {}, tmp_path / "state.json", FakeGitHub()
    runner.report_once(native, {"l4": 68}, state, state_path, client, 1_700_000_000)
    runner.report_once(native, {"l4": 68}, state, state_path, client, 1_700_000_001, panels=panels)
    assert len(client.comments) == 2
    assert "S1 epoch/log prefix" in client.comments[-1]["body"]
    assert "original" in state["pull_requests"]["68"]["last_comments_by_panel"]
    # A valid-looking old completion never substitutes for mismatched continuation evidence.
    metadata["source"]["sha256"] = "d" * 64
    _write(continuation / "run_metadata.json", metadata)
    assert not observe_host(native, "l4", 1_700_000_002, panels["l4"])["cells"][0]["fit_complete"]
    recovery_manifest = json.loads(path.read_bytes())
    for suffix in ("_r2", "_r3"):
        for cell, recipe in recovery_manifest["l4_continuations"].items():
            recipe["run_id"] = f"l4-{cell}_monitor_opt{suffix}-fit"
        _write(path, recovery_manifest)
        recovery = load_panel_manifest(path)
        row = observe_host(native, "l4", 1_700_000_003, recovery["l4"])["cells"][0]
        assert row["phase"] == "fit not observed" and not row["fit_complete"]
