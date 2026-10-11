from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

CONTROLLER_PATH = Path(__file__).resolve().parents[1] / "controller.py"
SPEC = importlib.util.spec_from_file_location("vast_teacher_controller", CONTROLLER_PATH)
assert SPEC is not None and SPEC.loader is not None
controller = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = controller
SPEC.loader.exec_module(controller)

OWNERSHIP = {"attempt_id": "attempt-test", "instance_id": 94123}


def _configure_collection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    collected = tmp_path / "collected"
    runtime = tmp_path / "runtime"
    monkeypatch.setattr(controller, "COLLECTED_DIR", collected)
    monkeypatch.setattr(controller, "COLLECTION_MANIFEST", collected / "collection_manifest.json")
    monkeypatch.setattr(controller, "COLLECTION_RECEIPT", collected / "collection_receipt.json")
    monkeypatch.setattr(controller, "COLLECTION_VERIFICATION", runtime / "collection_verified.json")
    return collected


def _write_collection(collected: Path) -> list[dict[str, object]]:
    prediction = collected / "cache/pred_vt_fixture_valid.npy"
    prediction.parent.mkdir(parents=True)
    np.save(prediction, np.array([[0.25, -1.5], [2.0, 0.0]], dtype=np.float32), allow_pickle=False)
    artifacts: list[tuple[str, str, str, Path]] = [
        (
            "wnn/cache/pred_vt_fixture_valid.npy",
            "wnn/cache/pred_vt_fixture_valid.npy",
            "cache/pred_vt_fixture_valid.npy",
            prediction,
        ),
        (
            "wnn/cache/pred_vt_fixture_valid.json",
            "wnn/cache/pred_vt_fixture_valid.json",
            "cache/pred_vt_fixture_valid.json",
            prediction.with_suffix(".json"),
        ),
        (
            "wnn/artifacts/runs/vt_fixture/model_ep1.pt",
            "wnn/artifacts/runs/vt_fixture/model_ep1.pt",
            "runs/vt_fixture/model_ep1.pt",
            collected / "runs/vt_fixture/model_ep1.pt",
        ),
        (
            "run/logs/vt_fixture.fit.log",
            "run/logs/vt_fixture.fit.log",
            "logs/vt_fixture.fit.log",
            collected / "logs/vt_fixture.fit.log",
        ),
        ("run/status.json", "run/status.json", "status.json", collected / "status.json"),
        ("run/DONE.json", "run/DONE.json", "DONE.json", collected / "DONE.json"),
        (
            "run/refill/status.json",
            "run/.collection_snapshot/refill/status.json",
            "refill/status.json",
            collected / "refill/status.json",
        ),
        (
            "wnn/artifacts/runs/fin_mu256p0e4_s3/model.pt",
            "wnn/artifacts/runs/fin_mu256p0e4_s3/model.pt",
            "runs/fin_mu256p0e4_s3/model.pt",
            collected / "runs/fin_mu256p0e4_s3/model.pt",
        ),
        (
            "wnn/artifacts/runs/fin_mu256p0e4_s3/training_state.pt",
            "wnn/artifacts/runs/fin_mu256p0e4_s3/training_state.pt",
            "runs/fin_mu256p0e4_s3/training_state.pt",
            collected / "runs/fin_mu256p0e4_s3/training_state.pt",
        ),
        (
            "wnn/artifacts/runs/fin_mu256p0e4_s3/args.json",
            "wnn/artifacts/runs/fin_mu256p0e4_s3/args.json",
            "runs/fin_mu256p0e4_s3/args.json",
            collected / "runs/fin_mu256p0e4_s3/args.json",
        ),
    ]
    artifacts[1][3].write_text('{"shape": [2, 2], "dtype": "float32"}')
    artifacts[2][3].parent.mkdir(parents=True)
    artifacts[2][3].write_bytes(b"small checkpoint fixture")
    artifacts[3][3].parent.mkdir(parents=True)
    artifacts[3][3].write_text("epoch 1 complete\n")
    artifacts[4][3].write_text('{"phase": "done"}')
    artifacts[5][3].write_text('{"result": "success"}')
    artifacts[6][3].parent.mkdir(parents=True)
    artifacts[6][3].write_text('{"_pending": []}')
    artifacts[7][3].parent.mkdir(parents=True)
    artifacts[7][3].write_bytes(b"final model")
    artifacts[8][3].write_bytes(b"optimizer and resume state")
    artifacts[9][3].write_text('{"seed": 3, "recipe": "mu256p0e4"}')

    files = [
        {
            "remote_path": remote_path,
            "transfer_path": transfer_path,
            "size_bytes": artifact_path.stat().st_size,
            "sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
        }
        for remote_path, transfer_path, _, artifact_path in artifacts
    ]
    manifest = {
        "schema": controller.COLLECTION_MANIFEST_SCHEMA,
        "inventory_scope": controller.COLLECTION_INVENTORY_SCOPE,
        "attempt_id": OWNERSHIP["attempt_id"],
        "instance_id": OWNERSHIP["instance_id"],
        "inventory_complete": True,
        "quiescent": True,
        "primary_terminal": "DONE.json",
        "files": files,
        "inventory_sha256": controller._collection_inventory_sha256(files),
    }
    manifest_raw = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    controller.COLLECTION_MANIFEST.write_bytes(manifest_raw)
    collection = {
        "schema": controller.COLLECTION_RECEIPT_SCHEMA,
        "attempt_id": OWNERSHIP["attempt_id"],
        "instance_id": OWNERSHIP["instance_id"],
        "remote_manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "remote_inventory_sha256": manifest["inventory_sha256"],
        "collector_reviewed_complete": True,
        "artifacts": [
            {
                "remote_path": remote_path,
                "local_path": local_path,
                "size_bytes": artifact_path.stat().st_size,
                "sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
            }
            for remote_path, _, local_path, artifact_path in artifacts
        ],
    }
    controller.COLLECTION_RECEIPT.write_text(
        json.dumps(collection, sort_keys=True, separators=(",", ":"))
    )
    return files


def _seed_remote_outputs(remote_root: Path) -> None:
    prediction = remote_root / "wnn/cache/pred_vt_fixture_valid.npy"
    prediction.parent.mkdir(parents=True)
    np.save(prediction, np.array([[0.25, -1.5], [2.0, 0.0]], dtype=np.float32), allow_pickle=False)
    prediction.with_suffix(".json").write_text('{"shape": [2, 2], "dtype": "float32"}')
    run_output = remote_root / "wnn/artifacts/runs/vt_fixture"
    run_output.mkdir(parents=True)
    (run_output / "model_ep1.pt").write_bytes(b"checkpoint")
    (run_output / "provenance.json").write_text('{"seed": 1}')
    (run_output / "optimizer.pt").write_bytes(b"not a model output")
    final_output = remote_root / "wnn/artifacts/runs/fin_mu256p0e4_s3"
    final_output.mkdir(parents=True)
    (final_output / "model.pt").write_bytes(b"final checkpoint")
    (final_output / "training_state.pt").write_bytes(b"optimizer resume state")
    (final_output / "args.json").write_text('{"name": "fin_mu256p0e4_s3"}')
    (final_output / "stats.json").write_text('{"loss": 0.5}')
    (final_output / "init_provenance.json").write_text('{"source": "independent"}')
    cache = remote_root / "wnn/cache"
    (cache / "training_cache.bin").write_bytes(b"ephemeral cache")
    run = remote_root / "run"
    (run / "logs").mkdir(parents=True)
    (run / "logs/vt_fixture.fit.log").write_text("epoch 1 complete\n")
    (run / "status.json").write_text('{"phase": "done"}')
    (run / "DONE.json").write_text('{"result": "success"}')
    (run / "worker.out").write_text("worker terminal output\n")
    (run / "refill.out").write_text("refill terminal output\n")
    (run / "plan.json").write_text('{"private_input": "excluded"}')
    refill = run / "refill"
    (refill / "status.json").parent.mkdir(parents=True)
    (refill / "status.json").write_text('{"_pending": [], "fixture": {"running": false, "rc": 0}}')
    (refill / "pending").mkdir()
    (refill / "running").mkdir()
    (refill / "done").mkdir()
    (refill / "done/fixture.rc0.json").write_text('{"name": "fixture"}')
    (refill / "held").mkdir()
    (refill / "held/capped.json").write_text('{"name": "held"}')


def _run_remote_inventory(
    remote_root: Path, gpu_processes: str = "", refill_daemon: bool = False
) -> subprocess.CompletedProcess[str]:
    fake_bin = remote_root.parent / "fake-bin"
    fake_bin.mkdir(exist_ok=True)
    nvidia_smi = fake_bin / "nvidia-smi"
    nvidia_smi.write_text(f"#!/bin/sh\nprintf '%s' {json.dumps(gpu_processes)}\n")
    nvidia_smi.chmod(0o700)
    pgrep = fake_bin / "pgrep"
    pgrep.write_text("#!/bin/sh\n" + ("echo 4173\nexit 0\n" if refill_daemon else "exit 1\n"))
    pgrep.chmod(0o700)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    return subprocess.run(
        [
            sys.executable,
            "-c",
            controller.REMOTE_COLLECTION_SCRIPT,
            str(OWNERSHIP["attempt_id"]),
            str(OWNERSHIP["instance_id"]),
            str(remote_root),
        ],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def test_remote_manifest_exhaustively_lists_bounded_output_scope(tmp_path: Path) -> None:
    remote_root = tmp_path / "remote"
    _seed_remote_outputs(remote_root)

    result = _run_remote_inventory(remote_root)

    assert result.returncode == 0, result.stderr
    manifest = json.loads((remote_root / "run/collection_manifest.json").read_text())
    assert manifest["inventory_complete"] is True
    assert manifest["quiescent"] is True
    assert manifest["attempt_id"] == OWNERSHIP["attempt_id"]
    assert manifest["instance_id"] == OWNERSHIP["instance_id"]
    paths = {item["remote_path"] for item in manifest["files"]}
    assert {
        "wnn/cache/pred_vt_fixture_valid.npy",
        "wnn/cache/pred_vt_fixture_valid.json",
        "wnn/artifacts/runs/vt_fixture/model_ep1.pt",
        "wnn/artifacts/runs/vt_fixture/provenance.json",
        "wnn/artifacts/runs/fin_mu256p0e4_s3/model.pt",
        "wnn/artifacts/runs/fin_mu256p0e4_s3/training_state.pt",
        "wnn/artifacts/runs/fin_mu256p0e4_s3/args.json",
        "wnn/artifacts/runs/fin_mu256p0e4_s3/stats.json",
        "wnn/artifacts/runs/fin_mu256p0e4_s3/init_provenance.json",
        "run/logs/vt_fixture.fit.log",
        "run/status.json",
        "run/DONE.json",
        "run/refill/status.json",
        "run/refill/done/fixture.rc0.json",
        "run/refill/held/capped.json",
        "run/worker.out",
        "run/refill.out",
    } <= paths
    assert not any(
        "training_cache" in path or "optimizer.pt" in path or path == "run/plan.json"
        for path in paths
    )
    assert (remote_root / "run/collection_manifest.json").is_file()
    refill_status = next(
        item for item in manifest["files"] if item["remote_path"] == "run/refill/status.json"
    )
    assert refill_status["transfer_path"] == "run/.collection_snapshot/refill/status.json"
    assert (remote_root / refill_status["transfer_path"]).read_text() == (
        remote_root / "run/refill/status.json"
    ).read_text()


def test_remote_inventory_refuses_pending_work(tmp_path: Path) -> None:
    remote_root = tmp_path / "remote"
    _seed_remote_outputs(remote_root)
    (remote_root / "run/refill/pending/not-yet-run.json").write_text("{}")

    result = _run_remote_inventory(remote_root)

    assert result.returncode != 0
    assert "not drained" in result.stderr
    assert not (remote_root / "run/collection_manifest.json").exists()


def test_remote_inventory_refuses_active_gpu_after_bootstrap_done(tmp_path: Path) -> None:
    remote_root = tmp_path / "remote"
    _seed_remote_outputs(remote_root)

    result = _run_remote_inventory(remote_root, gpu_processes="4172\n")

    assert result.returncode != 0
    assert "GPU process inventory is unavailable or still active" in result.stderr
    assert not (remote_root / "run/collection_manifest.json").exists()


def test_remote_inventory_refuses_live_idle_refill_daemon(tmp_path: Path) -> None:
    remote_root = tmp_path / "remote"
    _seed_remote_outputs(remote_root)

    result = _run_remote_inventory(remote_root, refill_daemon=True)

    assert result.returncode != 0
    assert "refill daemon" in result.stderr
    assert not (remote_root / "run/collection_manifest.json").exists()


def test_remote_state_parses_done_job_suffix_from_probe() -> None:
    assert controller._remote_state(
        {
            "PENDING": "0 RUNNING 0 DONEJOBS first.rc0.json,",
            "GPUPROCS": "0",
            "REFILL_DAEMON": "0",
            "TERMINAL": "DONE.json,",
        }
    ) == (0, 0, 0, 0, True)


def test_cmd_collect_downloads_exact_inventory_and_writes_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    collected = _configure_collection(tmp_path, monkeypatch)
    remote_root = tmp_path / "remote"
    _seed_remote_outputs(remote_root)
    monkeypatch.setattr(controller, "receipt", lambda: OWNERSHIP)

    def fake_ssh(command: str, timeout: int = 120) -> subprocess.CompletedProcess[str]:
        assert "vast-teacher-terminal-outputs-v2" in command
        return _run_remote_inventory(remote_root)

    def fake_scp(sources: list[str], destination: str, upload: bool, timeout: int = 1800) -> None:
        assert upload is False
        target = Path(destination)
        for source in sources:
            relative = source.removeprefix("/root/")
            target_file = target / Path(relative).name
            shutil.copyfile(remote_root / relative, target_file)

    monkeypatch.setattr(controller, "ssh", fake_ssh)
    monkeypatch.setattr(controller, "scp", fake_scp)

    controller.cmd_collect(Namespace())

    output = json.loads(capsys.readouterr().out)
    assert output["artifacts"] == len(
        json.loads(controller.COLLECTION_MANIFEST.read_text())["files"]
    )
    assert output["collection_status"] == "verified_cached"
    assert controller._collection_status(OWNERSHIP) == (True, "verified_cached")
    assert (collected / "runs/vt_fixture/provenance.json").is_file()
    assert (collected / "runs/fin_mu256p0e4_s3/training_state.pt").is_file()
    assert (collected / "runs/fin_mu256p0e4_s3/args.json").is_file()
    assert (collected / "refill/held/capped.json").is_file()
    assert np.array_equal(
        np.load(collected / "cache/pred_vt_fixture_valid.npy", allow_pickle=False),
        np.array([[0.25, -1.5], [2.0, 0.0]], dtype=np.float32),
    )
    assert (remote_root / "run/collection_manifest.json").is_file()


def test_actual_small_npy_collection_is_verified_once_and_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    collected = _configure_collection(tmp_path, monkeypatch)
    _write_collection(collected)

    ok, status = controller._collection_status(OWNERSHIP)

    assert ok is True
    assert status == "verified"
    assert controller.COLLECTION_VERIFICATION.is_file()

    def unexpected_hash(_: Path) -> str:
        pytest.fail("verified artifacts should not be rehashed on every tick")

    monkeypatch.setattr(controller, "_file_sha256", unexpected_hash)
    ok, status = controller._collection_status(OWNERSHIP)
    assert ok is True
    assert status == "verified_cached"


@pytest.mark.parametrize("missing", ["manifest", "receipt"])
def test_missing_manifest_or_receipt_requires_collection(
    missing: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    collected = _configure_collection(tmp_path, monkeypatch)
    _write_collection(collected)
    target = (
        controller.COLLECTION_MANIFEST if missing == "manifest" else controller.COLLECTION_RECEIPT
    )
    target.unlink()

    ok, status = controller._collection_status(OWNERSHIP)

    assert ok is False
    assert "missing or invalid" in status


def test_changed_local_artifact_invalidates_verified_collection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    collected = _configure_collection(tmp_path, monkeypatch)
    _write_collection(collected)
    assert controller._collection_status(OWNERSHIP)[0] is True
    checkpoint = collected / "runs/vt_fixture/model_ep1.pt"
    original = checkpoint.read_bytes()
    checkpoint.write_bytes(b"x" + original[1:])

    ok, status = controller._collection_status(OWNERSHIP)

    assert ok is False
    assert "wrong size" in status or "hash differs" in status


def test_same_artifact_count_with_wrong_local_name_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    collected = _configure_collection(tmp_path, monkeypatch)
    _write_collection(collected)
    collection = json.loads(controller.COLLECTION_RECEIPT.read_text())
    collection["artifacts"][2]["local_path"] = "runs/vt_fixture/renamed_model_ep1.pt"
    controller.COLLECTION_RECEIPT.write_text(json.dumps(collection))

    ok, status = controller._collection_status(OWNERSHIP)

    assert ok is False
    assert "does not exactly bind" in status


def test_missing_local_checkpoint_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    collected = _configure_collection(tmp_path, monkeypatch)
    _write_collection(collected)
    (collected / "runs/vt_fixture/model_ep1.pt").unlink()

    ok, status = controller._collection_status(OWNERSHIP)

    assert ok is False
    assert "missing or has the wrong size" in status


def test_collection_missing_blocks_tick_teardown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _configure_collection(tmp_path, monkeypatch)
    monkeypatch.setattr(controller, "receipt", lambda: OWNERSHIP)
    monkeypatch.setattr(
        controller,
        "probe",
        lambda: {
            "PENDING": "0 RUNNING 0",
            "GPUPROCS": "0",
            "REFILL_DAEMON": "0",
            "TERMINAL": "DONE.json",
        },
    )
    monkeypatch.setattr(controller, "spend", lambda: (70.0, 70.0))
    monkeypatch.setattr(controller, "ready_items", lambda _: [])
    monkeypatch.setattr(
        controller, "cmd_collect", lambda _: (_ for _ in ()).throw(SystemExit("fixture missing"))
    )
    destroyed: list[bool] = []
    monkeypatch.setattr(controller, "cmd_destroy", lambda _: destroyed.append(True))

    controller.cmd_tick(Namespace(no_pull=True, cap=70.0))

    output = capsys.readouterr().out
    assert '"collection_required": true' in output
    assert '"drained": false' in output
    assert destroyed == []


def test_direct_destroy_fails_closed_without_collection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure_collection(tmp_path, monkeypatch)
    monkeypatch.setattr(controller, "receipt", lambda: OWNERSHIP)

    def unexpected_provider_call(*args: str, **kwargs: object) -> str:
        pytest.fail("destroy must not reach the provider without collection evidence")

    monkeypatch.setattr(controller, "vast", unexpected_provider_call)
    with pytest.raises(SystemExit, match="collection required"):
        controller.cmd_destroy(Namespace())


def test_direct_destroy_reprobes_remote_before_provider_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    collected = _configure_collection(tmp_path, monkeypatch)
    _write_collection(collected)
    monkeypatch.setattr(controller, "receipt", lambda: OWNERSHIP)
    monkeypatch.setattr(
        controller,
        "probe",
        lambda: {
            "PENDING": "1 RUNNING 0",
            "GPUPROCS": "0",
            "REFILL_DAEMON": "0",
            "TERMINAL": "DONE.json,",
        },
    )

    def unexpected_provider_call(*args: str, **kwargs: object) -> str:
        pytest.fail("fresh quiescence must block destroy while refill work is pending")

    monkeypatch.setattr(controller, "vast", unexpected_provider_call)
    with pytest.raises(SystemExit, match="remote quiescence required"):
        controller.cmd_destroy(Namespace())


def test_tick_feeds_ready_backlog_and_invalidates_old_collection_without_destroy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    collected = _configure_collection(tmp_path, monkeypatch)
    _write_collection(collected)
    monkeypatch.setattr(controller, "receipt", lambda: OWNERSHIP)
    monkeypatch.setattr(
        controller,
        "probe",
        lambda: {
            "PENDING": "0 RUNNING 0",
            "GPUPROCS": "0",
            "REFILL_DAEMON": "0",
            "TERMINAL": "DONE.json,",
        },
    )
    monkeypatch.setattr(controller, "spend", lambda: (1.0, 1.0))
    monkeypatch.setattr(controller, "ready_items", lambda _: [{"id": "ready"}])
    fed: list[bool] = []
    destroyed: list[bool] = []
    monkeypatch.setattr(controller, "cmd_feed", lambda _: fed.append(True) or True)
    monkeypatch.setattr(controller, "cmd_destroy", lambda _: destroyed.append(True))

    controller.cmd_tick(Namespace(no_pull=True, cap=70.0))

    assert fed == [True]
    assert destroyed == []
    assert not controller.COLLECTION_RECEIPT.exists()
    assert not controller.COLLECTION_VERIFICATION.exists()


def test_tick_stops_idle_refill_then_collects_reprobes_and_destroys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _configure_collection(tmp_path, monkeypatch)
    monkeypatch.setattr(controller, "receipt", lambda: OWNERSHIP)
    states = [
        {"PENDING": "0 RUNNING 0", "GPUPROCS": "0", "REFILL_DAEMON": "1", "TERMINAL": "DONE.json,"},
        {"PENDING": "0 RUNNING 0", "GPUPROCS": "0", "REFILL_DAEMON": "0", "TERMINAL": "DONE.json,"},
        {"PENDING": "0 RUNNING 0", "GPUPROCS": "0", "REFILL_DAEMON": "0", "TERMINAL": "DONE.json,"},
    ]
    probes: list[bool] = []

    def fake_probe() -> dict[str, str]:
        probes.append(True)
        return (
            states.pop(0)
            if states
            else {
                "PENDING": "0 RUNNING 0",
                "GPUPROCS": "0",
                "REFILL_DAEMON": "0",
                "TERMINAL": "DONE.json,",
            }
        )

    monkeypatch.setattr(controller, "probe", fake_probe)
    monkeypatch.setattr(controller, "spend", lambda: (70.0, 70.0))
    monkeypatch.setattr(controller, "ready_items", lambda _: [])
    monkeypatch.setattr(controller.time, "sleep", lambda _: None)
    stop_calls: list[str] = []
    monkeypatch.setattr(
        controller,
        "ssh",
        lambda command, **kwargs: (
            stop_calls.append(command) or subprocess.CompletedProcess([], 0, "", "")
        ),
    )
    collect_calls: list[bool] = []
    status_calls: list[bool] = []

    def fake_collection_status(_: dict[str, object]) -> tuple[bool, str]:
        status_calls.append(True)
        return (False, "missing") if len(status_calls) == 1 else (True, "verified")

    monkeypatch.setattr(controller, "_collection_status", fake_collection_status)
    monkeypatch.setattr(controller, "cmd_collect", lambda _: collect_calls.append(True))
    destroyed: list[bool] = []
    monkeypatch.setattr(controller, "cmd_destroy", lambda _: destroyed.append(True))

    controller.cmd_tick(Namespace(no_pull=True, cap=70.0))

    assert stop_calls == ["touch /root/run/refill/STOP"]
    assert len(collect_calls) == 1
    assert len(probes) >= 3  # initial, post-STOP, and post-collection freshness check
    assert destroyed == [True]
    assert '"drained": true' in capsys.readouterr().out


def test_create_receipt_drops_api_key_and_unknown_response_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "ownership.json"
    monkeypatch.setattr(controller, "RECEIPT", target)
    monkeypatch.setattr(controller, "now", lambda: "2026-10-11T00:00:00+00:00")
    monkeypatch.setattr(
        controller,
        "vast",
        lambda *args, **kwargs: json.dumps(
            {
                "success": True,
                "new_contract": 94123,
                "instance_api_key": "sensitive-key-value",
                "account": {"token": "nested-sensitive-value"},
            }
        ),
    )

    controller.cmd_create(
        Namespace(offer=800, attempt="attempt-test", image="image", disk=20, hard_stop="later")
    )

    saved = json.loads(target.read_text())
    assert saved["create_response"] == {"success": True, "new_contract": 94123}
    assert "sensitive-key-value" not in target.read_text()
    assert "nested-sensitive-value" not in target.read_text()
