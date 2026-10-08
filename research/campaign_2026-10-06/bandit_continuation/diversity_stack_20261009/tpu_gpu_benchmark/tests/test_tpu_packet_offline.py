from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[4]
sys.path.insert(0, str(HERE))
packet_builder = importlib.import_module("seal_tpu_packet")
seal = packet_builder.seal
runtime = importlib.import_module("tpu_packet_runtime")
CACHE_KEY = runtime.CACHE_KEY
CACHE_MANIFEST_SHA = runtime.CACHE_MANIFEST_SHA
MEMBERSHIP_SHA = runtime.MEMBERSHIP_SHA
REGISTRY_SHA = runtime.REGISTRY_SHA
SPLIT_SHA = runtime.SPLIT_SHA
check_admission = runtime.check_admission
digest = runtime.digest
verify_shared_contract = runtime.verify_shared_contract


def test_actual_cpu_producer_metadata_and_binary_accounting() -> None:
    """Use archived real producer metadata, distinguishing NPY and physical reads."""
    fixture = HERE / "fixtures/actual_cpu_stage_r1"
    manifest_path = fixture / "manifest.json"
    assert digest(manifest_path) == CACHE_MANIFEST_SHA
    manifest = json.loads(manifest_path.read_text())
    assert manifest["cache_key"] == CACHE_KEY
    assert manifest["membership_artifact"]["sha256"] == MEMBERSHIP_SHA
    assert manifest["split_manifest_sha256"] == SPLIT_SHA
    assert manifest["member_counts"] == {"fit_train": 350, "fit_valid": 28, "evaluation_valid": 468}
    assert len(manifest["arrays"]) == 6
    for name, row in manifest["arrays"].items():
        element_bytes = 1 if name.endswith("scored") else 4
        payload_bytes = element_bytes
        for dimension in row["shape"]:
            payload_bytes *= dimension
        assert row["size_bytes"] == payload_bytes + 128
    assert sum(row["size_bytes"] for row in manifest["arrays"].values()) == 7_732_440_768
    receipt = json.loads((fixture / "reader_validation.json").read_text())
    assert receipt["manifest_sha256"] == CACHE_MANIFEST_SHA
    assert set(receipt["arrays"]) == set(manifest["arrays"])
    for name, row in manifest["arrays"].items():
        assert receipt["arrays"][name]["sha256"] == row["sha256"]
        assert receipt["arrays"][name]["bytes"] == row["size_bytes"]


def test_sealed_actual_entrypoint_runs_offline_and_has_pending_authority(tmp_path: Path) -> None:
    packet_dir = tmp_path / "packet"
    packet = seal(ROOT, packet_dir, None)
    metadata = json.loads((packet_dir / "kernel/kernel-metadata.json").read_text())
    assert packet["bootstrap_bytes"] < 1_000_000
    assert metadata["kernel_sources"] == [
        "vladkochetov07/wnn-parsed-cache-source-stage-20261009-v1",
        "vladkochetov07/wnn-loss-pair-gpu-20261009-v2",
    ]
    env = os.environ.copy()
    env["KAGGLE_WORKING"] = str(tmp_path / "working")
    env["WNN_TPU_OFFLINE"] = "1"
    subprocess.run(
        [sys.executable, str(packet_dir / "kernel" / metadata["code_file"])],
        env=env,
        check=True,
        timeout=30,
    )
    offline = json.loads((tmp_path / "working/out/offline_source_receipt.json").read_text())
    assert offline["status"] == "pass" and offline["fit_admitted"] is False
    assert offline["source_manifest_sha256"] == packet["source_manifest_sha256"]
    # Exercise imports from the exact extracted closure, without a model fit.
    env["JAX_PLATFORMS"] = "cpu"
    for entrypoint in ("tpu_worker.py", "run_device_parity.py"):
        result = subprocess.run(
            [
                sys.executable,
                str(tmp_path / "working/tpu_source" / packet_builder.PREFIX / entrypoint),
                "--help",
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout
    with pytest.raises(ValueError, match="pending"):
        check_admission(
            json.loads((packet_dir / "root_admission.json").read_text()),
            packet["source_manifest_sha256"],
        )


def test_shared_contract_accepts_role_triples_and_rejects_mutation(tmp_path: Path) -> None:
    """Schema adversarial check; actual GPU writer fixture remains a separate gate."""
    artifacts = {}
    for role in (
        "initial_state",
        "normalization",
        "normalization_metadata",
        "sample_orders",
        "onecycle_trace",
        "profile_inputs",
        "profile_inputs_receipt",
    ):
        path = tmp_path / f"{role}.fixture"
        path.write_bytes(role.encode())
        artifacts[role] = {
            "path": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": digest(path),
        }
    contract = {
        "schema": "wnn.loss-pair.training-contract.v1",
        "format_version": 1,
        "state": "complete",
        "cache": {
            "manifest_sha256": CACHE_MANIFEST_SHA,
            "cache_key": CACHE_KEY,
            "membership_sha256": MEMBERSHIP_SHA,
            "split_manifest_sha256": SPLIT_SHA,
            "portable_registry_sha256": REGISTRY_SHA,
            "fold": 0,
        },
        "artifacts": artifacts,
    }
    path = tmp_path / "training_contract.json"
    path.write_text(json.dumps(contract))
    admission = {"gpu_training_contract_sha256": digest(path)}
    _, resolved = verify_shared_contract(path, admission)
    assert set(resolved) == set(artifacts)
    resolved["initial_state"].write_bytes(b"mutated")
    with pytest.raises(ValueError, match="changed"):
        verify_shared_contract(path, admission)


def test_actual_profile_receipt_requires_exact_batch_cache_and_normalization() -> None:
    receipt = json.loads((HERE / "fixtures/actual_gpu_writer_profile_receipt_r1.json").read_text())
    ids = receipt["sequence_ids"]
    normalization_sha = receipt["normalization_sha256"]
    runtime.check_profile_binding(receipt, ids, normalization_sha)
    for key, value in (
        ("cache_key", "wrong-cache"),
        ("sequence_ids", list(reversed(ids))),
        ("normalization_sha256", "wrong-normalization"),
    ):
        with pytest.raises(ValueError, match="first-batch order differs"):
            runtime.check_profile_binding({**receipt, key: value}, ids, normalization_sha)
