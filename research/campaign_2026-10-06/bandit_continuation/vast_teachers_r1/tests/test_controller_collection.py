from __future__ import annotations

import hashlib
import importlib.util
import json
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
    artifacts: list[tuple[str, str, Path]] = [
        (
            "wnn/cache/pred_vt_fixture_valid.npy",
            "cache/pred_vt_fixture_valid.npy",
            prediction,
        ),
        (
            "wnn/cache/pred_vt_fixture_valid.json",
            "cache/pred_vt_fixture_valid.json",
            prediction.with_suffix(".json"),
        ),
        (
            "wnn/artifacts/runs/vt_fixture/model_ep1.pt",
            "runs/vt_fixture/model_ep1.pt",
            collected / "runs/vt_fixture/model_ep1.pt",
        ),
        (
            "run/logs/vt_fixture.fit.log",
            "logs/vt_fixture.fit.log",
            collected / "logs/vt_fixture.fit.log",
        ),
    ]
    artifacts[1][2].write_text('{"shape": [2, 2], "dtype": "float32"}')
    artifacts[2][2].parent.mkdir(parents=True)
    artifacts[2][2].write_bytes(b"small checkpoint fixture")
    artifacts[3][2].parent.mkdir(parents=True)
    artifacts[3][2].write_text("epoch 1 complete\n")

    files = [
        {
            "remote_path": remote_path,
            "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for remote_path, _, path in artifacts
    ]
    manifest = {
        "schema": controller.COLLECTION_MANIFEST_SCHEMA,
        "inventory_complete": True,
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
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for remote_path, local_path, path in artifacts
        ],
    }
    controller.COLLECTION_RECEIPT.write_text(
        json.dumps(collection, sort_keys=True, separators=(",", ":"))
    )
    return files


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
        lambda: {"PENDING": "0 RUNNING 0", "GPUPROCS": "0", "TERMINAL": "DONE.json"},
    )
    monkeypatch.setattr(controller, "spend", lambda: (70.0, 70.0))
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
