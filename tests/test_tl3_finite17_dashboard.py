from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from scripts import tl3_finite17_dashboard as dashboard


def _sha(value: object) -> str:
    return dashboard._canonical_sha256(value)


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, Any]]:
    root = tmp_path / "flat"
    state = root / "finite17-state"
    runs = tmp_path / "dashboard-runs"
    root.mkdir()
    state.mkdir(parents=True)
    runs.mkdir()
    train = list(range(10_325))
    valid = list(range(10_325, 12_198))
    manifest = root / "split_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "wnn.chronological-primary.v1",
                "fit": {"train": train, "valid": []},
                "evaluation": {"valid": valid},
            }
        ),
        encoding="utf-8",
    )
    source_sha = "a" * 64
    split_sha = dashboard._sha256_file(manifest)
    run = "tl3c_fin256_s4"
    args = [
        "--name",
        run,
        "--train-source",
        "train",
        "--train-seqs",
        "10325",
        "--train-eval-seqs",
        "32",
        "--batch",
        "25",
        "--chunk",
        "2000",
        "--epochs",
        "1",
        "--eval-source",
        "dev",
        "--valid-fold",
        "0",
        "--n-valid-folds",
        "1",
        "--checkpoint-epoch",
        "0",
        "--checkpoint-policy",
        "fixed",
        "--precision",
        "fp32",
        "--full-valid",
    ]
    nodes: list[dict[str, Any]] = []
    for index in range(17):
        node_id = "fin256_s3" if index == 0 else f"node{index:02d}"
        node_run = "tl3c_fin256_s3" if index == 0 else (run if index == 1 else f"tl3c_{node_id}")
        node_args = args if index == 1 else ["--name", node_run]
        nodes.append(
            {
                "id": node_id,
                "run": node_run,
                "args": node_args,
                "args_sha256": _sha(node_args),
                "parents": [],
                "expected_updates": 4_130 if index == 1 else 1,
                "epochs": 1,
                "endpoint": 0,
                "run_directory": str(root / "artifacts" / "runs" / node_run),
            }
        )
    plan: dict[str, Any] = {
        "schema": dashboard.PLAN_SCHEMA,
        "root": str(root.resolve()),
        "state_dir": str(state.resolve()),
        "split": {
            "split_manifest": str(manifest),
            "split_manifest_sha256": split_sha,
            "fit_train_count": 10_325,
            "fit_valid_count": 0,
            "evaluation_valid_count": 1_873,
        },
        "source_identity": {"training_source_sha256": source_sha},
        "nodes": nodes,
    }
    plan["sha256"] = _sha(plan)
    node_status = {
        node["id"]: {
            "state": "running" if node["id"] == "node01" else "not_started",
            "phase": "fit" if node["id"] == "node01" else "not_started",
        }
        for node in nodes
    }
    command = [
        "/opt/venv/bin/python",
        str(root / "scripts" / "train_rnn.py"),
        *args,
    ]
    task = {
        "kind": "fit",
        "node_id": "node01",
        "state": "running",
        "source_sha256": source_sha,
        "args_sha256": nodes[1]["args_sha256"],
        "command": command,
        "command_sha256": _sha(command),
        "log": str(state / "logs" / "fit__node01.log"),
        "pid": 4242,
        "start_ticks": "987654",
        "gpu_uuid": "GPU-test-uuid",
    }
    status = {
        "schema": dashboard.STATE_SCHEMA,
        "status": "running",
        "plan_sha256": plan["sha256"],
        "nodes": node_status,
        "tasks": {"fit:node01": task},
    }
    (state / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    (state / "status.json").write_text(json.dumps(status), encoding="utf-8")

    prefix = sorted(train)[:32]
    members_sha = hashlib.sha256(
        json.dumps([["train", group] for group in prefix], separators=(",", ":")).encode("ascii")
    ).hexdigest()
    checkpoint = b"synthetic checkpoint bytes"
    run_dir = root / "artifacts" / "runs" / run
    run_dir.mkdir(parents=True)
    checkpoint_path = run_dir / "model_ep0.pt"
    checkpoint_path.write_bytes(checkpoint)
    provenance_args: dict[str, Any] = {
        "name": run,
        "train_source": "train",
        "train_seqs": 10_325,
        "train_eval_seqs": 32,
        "batch": 25,
        "chunk": 2_000,
        "epochs": 1,
        "eval_source": "vfold",
        "valid_fold": 0,
        "n_valid_folds": 1,
        "checkpoint_epoch": 0,
        "checkpoint_policy": "fixed",
        "precision": "fp32",
        "full_valid": True,
        "no_eval_during_fit": False,
    }
    provenance = {
        "split_manifest_sha256": split_sha,
        "trainer_implementation": {"combined_sha256": source_sha},
        "fit_row_groups": {"train": train, "valid": []},
        "evaluation_row_groups": valid,
        "args": provenance_args,
        "checkpoint_sha256": {checkpoint_path.name: hashlib.sha256(checkpoint).hexdigest()},
    }
    (run_dir / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    row = {
        "epoch": 0,
        "split": "vfold1873",
        "selected_rows": 3_528_273,
        "weighted_pearson": 0.3,
        "global_t0": 0.25,
        "global_t1": 0.35,
        "valid_wmse": 0.8,
        "train_eval_mask": "all_after_warmup",
        "train_eval_sequences": 32,
        "train_eval_row_groups": prefix,
        "train_eval_members_sha256": members_sha,
        "train_eval_selected_rows": 640_000,
        "train_eval_wp": 0.4,
        "train_eval_wp_t0": 0.35,
        "train_eval_wp_t1": 0.45,
        "train_eval_wmse": 0.7,
        "update_diagnostics": {"cumulative": {"optimizer_attempts": 4_130}},
        "fit_summary": {"objective_mean": 1.2},
    }
    log_dir = state / "logs"
    log_dir.mkdir()
    (log_dir / "fit__node01.log").write_text(
        "ep 0 batch 0/413 loss 1.25 lr 0.001 seq/s 10.0\n"
        + "GRAD_STATS "
        + json.dumps({"epoch": 0, "steps": 4130, "clip_rate": 0.2})
        + "\n"
        + "VALID "
        + json.dumps(row)
        + "\n",
        encoding="utf-8",
    )
    return root, state, runs, {"pid": 4242, "uuid": "GPU-test-uuid", "task": task}


def _proc_fixture(proc_root: Path, root: Path, task: dict[str, Any]) -> None:
    pid = task["pid"]
    process_dir = proc_root / str(pid)
    process_dir.mkdir(parents=True)
    fields = ["S", "1", str(pid), str(pid), *(["0"] * 15), task["start_ticks"]]
    (process_dir / "stat").write_text(f"{pid} (python) " + " ".join(fields), encoding="utf-8")
    argv = ["/opt/venv/bin/python", str(root / "scripts" / "train_rnn.py"), *task["command"][2:]]
    (process_dir / "cmdline").write_bytes(b"\0".join(part.encode() for part in argv) + b"\0")
    (process_dir / "environ").write_bytes(f"CUDA_VISIBLE_DEVICES={task['gpu_uuid']}\0".encode())
    (process_dir / "statm").write_text("100 50 0 0 0 0 0", encoding="ascii")


def test_native_broker_publishes_paired_metrics_and_verified_resources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, state, runs, identity = _fixture(tmp_path)
    proc_root = tmp_path / "proc"
    _proc_fixture(proc_root, root, identity["task"])
    monkeypatch.setattr(dashboard._process.os, "sysconf", lambda _key: 4_096, raising=False)
    telemetry = (
        [{"pid": 4242, "uuid": identity["uuid"], "memory": 300.0}],
        [{"uuid": identity["uuid"], "util": 80.0, "memory": 1_000.0}],
    )

    first = dashboard.publish_once(root, state, runs, proc_root=proc_root, telemetry=telemetry)
    assert first["status"] == "running"
    assert first["nodes"] == 16
    assert first["events_added"] > 16
    assert not (runs / "tl3c_fin256_s3").exists()
    assert len(list(runs.iterdir())) == 16
    view = runs / "tl3c_fin256_s4"
    events = [json.loads(line) for line in (view / "monitor_events.jsonl").read_text().splitlines()]
    evaluation = next(event for event in events if event["kind"] == "evaluation")
    assert evaluation["metrics"]["train_eval/loss"] == pytest.approx(0.7)
    assert evaluation["metrics"]["valid/loss"] == pytest.approx(0.8)
    assert evaluation["metrics"]["train_eval/score"] == pytest.approx(0.4)
    assert evaluation["metrics"]["train_eval/score/t0"] == pytest.approx(0.35)
    assert evaluation["metrics"]["valid/score"] == pytest.approx(0.3)
    assert evaluation["metrics"]["valid/score/t1"] == pytest.approx(0.35)
    assert evaluation["context"]["valid"]["row_mask"] == "is_scored"
    assert evaluation["context"]["train_eval"]["row_mask"] == "all_after_warmup"
    resource = next(event for event in events if event["kind"] == "resources")
    assert resource["context"]["process_identity_verified"] is True
    assert resource["metrics"]["resource/process_rss_bytes"] > 0
    progress = next(event for event in events if event["kind"] == "train_step")
    assert progress["metrics"]["train/loss"] == pytest.approx(1.25)
    assert "composite per-window fit objective" in progress["context"]["loss_definition"]

    replay = dashboard.publish_once(root, state, runs, proc_root=proc_root, telemetry=telemetry)
    assert replay["events_added"] == 1
    replayed = [
        json.loads(line) for line in (view / "monitor_events.jsonl").read_text().splitlines()
    ]
    assert sum(event["kind"] == "evaluation" for event in replayed) == 1
    assert sum(event["kind"] == "resources" for event in replayed) == 2


def test_native_broker_rejects_mislabelled_train_mask_before_evaluation(tmp_path: Path) -> None:
    root, state, runs, _identity = _fixture(tmp_path)
    log = state / "logs" / "fit__node01.log"
    text = log.read_text(encoding="utf-8")
    record = json.loads(text.split("VALID ", 1)[1])
    record["train_eval_mask"] = "is_scored"
    log.write_text(
        text.split("VALID ", 1)[0] + "VALID " + json.dumps(record) + "\n", encoding="utf-8"
    )

    with pytest.raises(dashboard.BridgeError, match="train-eval membership or mask"):
        dashboard.publish_once(root, state, runs, telemetry=([], []))

    view = runs / "tl3c_fin256_s4"
    assert not (view / "monitor_evaluations.jsonl").read_text(encoding="utf-8").strip()


def test_native_broker_keeps_polling_while_driver_is_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshots = iter(({"status": "running"}, {"status": "completed"}))
    monkeypatch.setattr(dashboard.sys, "platform", "linux")
    monkeypatch.setattr(dashboard, "publish_once", lambda *_args: next(snapshots))
    monkeypatch.setattr(dashboard.time, "sleep", lambda _seconds: None)

    assert (
        dashboard.run_bridge(tmp_path / "root", tmp_path / "state", tmp_path / "runs", 15, 60, 1)
        == 0
    )


def test_native_broker_ignores_process_with_reused_pid(tmp_path: Path) -> None:
    root, state, _runs, identity = _fixture(tmp_path)
    proc_root = tmp_path / "proc"
    _proc_fixture(proc_root, root, identity["task"])
    stat_path = proc_root / str(identity["pid"]) / "stat"
    stat_path.write_text(
        stat_path.read_text(encoding="utf-8").replace("987654", "123"), encoding="utf-8"
    )
    plan = dashboard._read_json(state / "plan.json")

    assert dashboard._owned_process(identity["task"], plan["nodes"][1], root, proc_root) is None
