"""Focused V5R2 packet tests; no training data or GPU execution is performed."""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

PACKET = Path(__file__).resolve().parents[1]
REPO = next(parent for parent in PACKET.parents if (parent / "AGENTS.md").is_file())
sys.path.insert(0, str(PACKET))

import build_fit_archive  # noqa: E402
import build_fit_packet  # noqa: E402
import capacity_fit_worker  # noqa: E402
import capacity_worker  # noqa: E402


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _extract_archive(destination: Path) -> Path:
    capacity_worker._safe_extract_source(PACKET, destination)
    return destination


def test_generated_outer_driver_is_valid_python_and_flushes_allowlisted_records() -> None:
    manifest = json.loads((PACKET / "source_dataset_upload/assets-manifest.json").read_text())
    source = build_fit_packet._outer_kernel_source(
        hashlib.sha256((PACKET / "source_dataset_upload/assets-manifest.json").read_bytes()).hexdigest(),
        [entry["path"] for entry in manifest["files"]],
    )
    ast.parse(source)
    assert "'{\"event\": \"campaign_resource_heartbeat\"'" in source
    assert "selectors.EVENT_READ" in source
    assert "log_stream.flush()" in source


def test_content_addressed_source_dataset_is_complete_and_driver_is_bound() -> None:
    upload_root = PACKET / "source_dataset_upload"
    manifest_path = upload_root / "assets-manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    assert manifest["private"] is True
    paths = [entry["path"] for entry in manifest["files"]]
    assert len(paths) == len(set(paths))
    for entry in manifest["files"]:
        assert entry["filename"] == f"{entry['sha256']}.wnn"
        source = upload_root / entry["filename"]
        assert source.is_file() and not source.is_symlink()
        assert source.stat().st_size == entry["size"]
        assert capacity_worker.sha256_file(source) == entry["sha256"]
    driver = (PACKET / "kernel_upload/kernel.py").read_text()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    assert f"EXPECTED_MANIFEST_SHA256 = '{manifest_sha}'" in driver
    build_receipt = json.loads((PACKET / "PACKET_BUILD_RECEIPT.json").read_text())
    assert build_receipt["source_dataset_manifest_sha256"] == manifest_sha
    assert build_receipt["kernel_driver_sha256"] == capacity_worker.sha256_file(
        PACKET / "kernel_upload/kernel.py"
    )
    recipe = json.loads((PACKET / "GRU512_768_MUON_FIT_RECIPE_V5R2.json").read_text())
    assert "fit-only" in recipe["status"]
    assert recipe["training"]["initial_batch"] == 8
    assert recipe["training"]["fallback_batch"] is None
    assert "no profile rerun" in recipe["training"]["batch_selection_rule"]
    assert "fit-only" in recipe["runtime"]["initial_execution_scope"]


def test_pair_cpu_allocation_matches_four_cpu_dual_t4_host() -> None:
    recipe = json.loads((PACKET / "GRU512_768_MUON_FIT_RECIPE_V5R2.json").read_text())
    assert recipe["runtime"]["cpu_threads"] == 2
    assert recipe["runtime"]["aggregate_cpu_threads"] == 4
    assert "V4 profile timings used 4 CPU threads per arm" in recipe["runtime"]["cpu_allocation_change"]
    specs = capacity_fit_worker._capacity_preflight_specs(["muon512", "muon768"])
    assert [spec["limits"]["cpus"] for spec in specs] == [2, 2]
    assert capacity_worker.CPU_THREADS_PER_ARM == 2
    env = capacity_worker._child_env(
        PACKET, "muon512", "fit", PACKET / "receipts", PACKET / "identity.json"
    )
    assert (env["WNN_THREADS"], env["OMP_NUM_THREADS"], env["MKL_NUM_THREADS"]) == (
        "2",
        "2",
        "2",
    )


def test_fit_authorization_requires_both_widths(tmp_path: Path) -> None:
    recipe_path = tmp_path / "GRU512_768_MUON_FIT_RECIPE_V5R2.json"
    (tmp_path / "source").mkdir()
    archive_path = tmp_path / "source/source_archive.tar"
    manifest_path = tmp_path / "source/source_archive_manifest.json"
    review_path = tmp_path / "ROOT_FIT_SOURCE_REVIEW.json"
    for path, value in (
        (recipe_path, b"recipe"),
        (archive_path, b"archive"),
        (manifest_path, b"manifest"),
        (review_path, b"approved review"),
    ):
        path.write_bytes(value)
    profile_evidence = {
        "manifest_sha256": "c" * 64,
        "profiles": {
            "muon512": {"step_wall_ms": 1845.01},
            "muon768": {"step_wall_ms": 2861.57},
        },
    }
    recipe = {"packet_id": capacity_fit_worker.PACKET_ID}
    review = {"packet_id": capacity_fit_worker.PACKET_ID, "status": "approved"}
    projected = {
        arm: round(row["step_wall_ms"] / 1000 * 1492 * 4 * 1.20, 3)
        for arm, row in profile_evidence["profiles"].items()
    }
    authorization = {
        "schema": "wnn-gru-capacity-fit-authorization-v1",
        "packet_id": capacity_fit_worker.PACKET_ID,
        "fit_authorized": True,
        "fit_arm_ids": ["muon512", "muon768"],
        "selected_batch": 8,
        "max_worker_seconds": 26_000,
        "profile_projection_seconds": projected,
        "recipe_sha256": hashlib.sha256(recipe_path.read_bytes()).hexdigest(),
        "source_archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        "source_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "profile_evidence_manifest_sha256": profile_evidence["manifest_sha256"],
        "root_review_sha256": hashlib.sha256(review_path.read_bytes()).hexdigest(),
        "root_source_review_sha256": hashlib.sha256(review_path.read_bytes()).hexdigest(),
    }
    assert capacity_fit_worker._validate_authorization(
        tmp_path, recipe, authorization, review, profile_evidence
    ) == (["muon512", "muon768"], 26_000)
    authorization["fit_arm_ids"] = ["muon512"]
    with pytest.raises(PermissionError, match="complete matched Muon512/Muon768 pair"):
        capacity_fit_worker._validate_authorization(
            tmp_path, recipe, authorization, review, profile_evidence
        )


def test_reused_v4_profile_receipts_bind_executed_precision_and_width_argv() -> None:
    evidence = capacity_fit_worker._verify_profile_evidence(PACKET)
    assert set(evidence["profiles"]) == {"muon512", "muon768"}
    for arm_id, hidden in (("muon512", 512), ("muon768", 768)):
        profile = evidence["profiles"][arm_id]
        assert profile["model_precision"] == "fp16"
        assert profile["amp_init_scale"] == 4096.0
        assert profile["grad_scaler_enabled"] is True
        assert profile["orthogonalization_dtype"] == "float32"
        assert profile["native_bf16_selected"] is False
        assert profile["device_capability"] == [7, 5]
        assert profile["optimizer_attempts"] == 230
        assert (arm_id == "muon512") == (hidden == 512)


def test_runner_patch_emits_one_uuid_binding_and_frozen_archive_compiles(
    monkeypatch, tmp_path: Path
) -> None:
    base_runner = build_fit_archive._frozen_source(REPO, "scripts/kaggle_gpu_runner.py")
    patched = build_fit_archive._patch_runner(base_runner)
    ast.parse(patched)
    assert patched.count('"WNN_PHYSICAL_GPU_UUID": str(device["uuid"])') == 1
    assert patched.count('"WNN_PHYSICAL_GPU_INDEX": str(device["index"])') == 1

    extracted = _extract_archive(tmp_path / "source")
    monkeypatch.syspath_prepend(str(extracted / "scripts"))
    runner = _load_module("v5_archived_kaggle_gpu_runner", extracted / "scripts/kaggle_gpu_runner.py")
    available_cpus = sorted(os.sched_getaffinity(0))
    assert len(available_cpus) >= 4
    host = {
        "capacity_bytes": 30 * 1024**3,
        "available_bytes": 30 * 1024**3,
        "affinity": available_cpus[:4],
    }
    devices = [
        {"index": 0, "uuid": "GPU-A", "name": "Tesla T4", "total_bytes": 16 * 1024**3, "used_bytes": 0},
        {"index": 1, "uuid": "GPU-B", "name": "Tesla T4", "total_bytes": 16 * 1024**3, "used_bytes": 0},
    ]
    runner.host_inventory = lambda: host
    runner.gpu_inventory = lambda: devices
    pair_specs = [
        {"limits": {"cpus": 2, "ram_bytes": 8 * 1024**3, "gpu_stop_bytes": 12 * 1024**3}},
        {"limits": {"cpus": 2, "ram_bytes": 8 * 1024**3, "gpu_stop_bytes": 12 * 1024**3}},
    ]
    assert runner.admit_specs(
        pair_specs,
        host,
        devices,
        maximum_preexisting_gpu_used_bytes=512 * 1024**2,
    ) == 2
    host_three_cpu = {**host, "affinity": host["affinity"][:3]}
    assert runner.admit_specs(
        pair_specs,
        host_three_cpu,
        devices,
        maximum_preexisting_gpu_used_bytes=512 * 1024**2,
    ) == 1

    script = extracted / "scripts/probe.py"
    script.write_text(
        "import json, os\n"
        "print('DEVICE_BINDING ' + json.dumps({**{k: os.environ[k] for k in "
        "('CUDA_VISIBLE_DEVICES','WNN_PHYSICAL_GPU_UUID','WNN_PHYSICAL_GPU_INDEX')}, "
        "'affinity_count': len(os.sched_getaffinity(0)), 'threads': [os.environ[k] for k in "
        "('WNN_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')]}), flush=True)\n",
        encoding="utf-8",
    )
    script_hash = hashlib.sha256(script.read_bytes()).hexdigest()
    specs = {}
    spec_paths = []
    for arm_id in ("muon512", "muon768"):
        spec_path = tmp_path / f"{arm_id}.json"
        spec_paths.append(spec_path)
        specs[str(spec_path)] = {
            "name": arm_id,
            "argv": [sys.executable, "scripts/probe.py"],
            "code": [{"path": "scripts/probe.py", "size": script.stat().st_size, "sha256": script_hash}],
            "inputs": [],
            "environment": {},
            "limits": {
                "cpus": 2,
                "ram_bytes": 8 * 1024**3,
                "gpu_memory_bytes": 6 * 1024**3,
                "gpu_stop_bytes": 12 * 1024**3,
                "timeout_seconds": 30,
            },
            "artifacts": [],
        }
    runner.read_spec = lambda path: copy.deepcopy(specs[str(path)])
    output = tmp_path / "runner-output"
    runner.run(
        extracted,
        spec_paths,
        output,
        stagger_seconds=0,
        required_concurrency=2,
        maximum_preexisting_gpu_used_bytes=512 * 1024**2,
    )
    for arm_id, uuid, index in (("muon512", "GPU-A", 0), ("muon768", "GPU-B", 1)):
        log = (output / arm_id / "worker.log").read_text()
        rows = [line for line in log.splitlines() if line.startswith("DEVICE_BINDING ")]
        assert len(rows) == 1
        binding = json.loads(rows[0].removeprefix("DEVICE_BINDING "))
        assert binding == {
            "CUDA_VISIBLE_DEVICES": uuid,
            "WNN_PHYSICAL_GPU_UUID": uuid,
            "WNN_PHYSICAL_GPU_INDEX": str(index),
            "affinity_count": 2,
            "threads": ["2", "2", "2"],
        }
        result = json.loads((output / arm_id / "result.json").read_text())
        assert result["status"] == "complete"


def test_child_environment_sampler_and_run_child_keep_device_and_timing(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-B")
    monkeypatch.setenv("WNN_PHYSICAL_GPU_UUID", "GPU-B")
    monkeypatch.setenv("WNN_PHYSICAL_GPU_INDEX", "1")
    monkeypatch.setattr(capacity_worker, "FINALIZATION_RESERVE_SECONDS", 0)
    env = capacity_worker._child_env(
        tmp_path, "muon768", "fit", tmp_path / "receipts", tmp_path / "identity.json"
    )
    assert (env["CUDA_VISIBLE_DEVICES"], env["WNN_PHYSICAL_GPU_UUID"], env["WNN_PHYSICAL_GPU_INDEX"]) == (
        "GPU-B",
        "GPU-B",
        "1",
    )

    calls = []

    def fake_nvidia_smi(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout="1, 100, 15000, 16000\n")

    monkeypatch.setattr(capacity_worker.subprocess, "run", fake_nvidia_smi)
    sampler = capacity_worker.GpuMemorySampler(interval_seconds=0.01)
    sampler.start()
    deadline = time.monotonic() + 1
    while not sampler.samples and time.monotonic() < deadline:
        time.sleep(0.01)
    receipt = sampler.stop()
    assert calls and calls[0][1] == "--id=GPU-B"
    assert receipt["gpu_peak_used_mb"] == 100

    packet_root = tmp_path / "packet"
    packet_root.mkdir()
    (packet_root / "train_entry.py").write_text(
        "print('GRU_CAPACITY_CHILD_STAGE {\"stage\":\"trainer_import_done\"}', flush=True)\n",
        encoding="utf-8",
    )
    worker_started = time.monotonic()
    result = capacity_worker._run_child(
        packet_root=packet_root,
        project_root=tmp_path,
        receipt_root=tmp_path / "receipts",
        input_identity=tmp_path / "identity.json",
        arm="muon768",
        phase="fit",
        run_name="unit-probe",
        argv=[],
        worker_started=worker_started,
        worker_deadline=worker_started + 30,
    )
    assert result["status"] == "complete"
    assert result["markers"]["child_stage"] == "trainer_import_done"
    assert result["phase_timings"]["child_process_seconds"] >= 0
    assert result["worker_elapsed_seconds"] >= 0


def test_arm_startup_binds_cuda_logical_zero_to_assigned_physical_t4(monkeypatch, capsys) -> None:
    import capacity_arm_worker

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-B")
    monkeypatch.setenv("WNN_PHYSICAL_GPU_UUID", "GPU-B")
    monkeypatch.setenv("WNN_PHYSICAL_GPU_INDEX", "1")
    calls = []

    def fake_nvidia_smi(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout="1, GPU-B, Tesla T4, 16384, 15000\n")

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 1

        @staticmethod
        def get_device_properties(index):
            assert index == 0
            return SimpleNamespace(major=7, minor=5, name="Tesla T4")

        @staticmethod
        def mem_get_info(index):
            assert index == 0
            return 15 * 1024**3, 16 * 1024**3

    monkeypatch.setattr(capacity_arm_worker.subprocess, "run", fake_nvidia_smi)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=FakeCuda))
    receipt = capacity_arm_worker._assert_assigned_device()
    assert calls[0][1] == "--id=GPU-B"
    assert receipt["physical_index"] == 1
    assert receipt["physical_uuid"] == "GPU-B"
    assert receipt["visible_device_count"] == 1
    assert "GRU_CAPACITY_DEVICE" in capsys.readouterr().out


def test_exact_fit_profile_and_prediction_argv_parse_in_frozen_archive(tmp_path: Path) -> None:
    extracted = _extract_archive(tmp_path / "source")
    scripts = extracted / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        arms = [
            {"id": "muon512", "hidden": 512, "optimizer": "muon", "run_name": "v5-512"},
            {"id": "muon768", "hidden": 768, "optimizer": "muon", "run_name": "v5-768"},
        ]
        split = extracted / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
        argv_sets = []
        for arm in arms:
            argv_sets.append(capacity_worker._profile_argv(arm, 8, split))
            argv_sets.append(capacity_worker._fit_argv(arm, 8, split))
        predictor_argv = [
            capacity_worker._predict_argv(arm["run_name"], split) for arm in arms
        ]
    finally:
        sys.path.remove(str(scripts))

    parser_probe = r'''
import argparse, json, sys
class Parsed(Exception): pass
sys.path[:0] = [sys.argv[1] + "/scripts", sys.argv[1] + "/src"]
import train_rnn
original = argparse.ArgumentParser.parse_args
captured = []
def stop_after_parse(self, *args, **kwargs):
    namespace = original(self, *args, **kwargs)
    captured.append({"name": namespace.name, "hidden": namespace.hidden,
                     "batch": namespace.batch, "train_source": namespace.train_source,
                     "precision": namespace.precision, "amp_init_scale": namespace.amp_init_scale,
                     "profile_steps": namespace.profile_steps})
    raise Parsed
argparse.ArgumentParser.parse_args = stop_after_parse
for argv in json.loads(sys.argv[2]):
    sys.argv = [sys.argv[1] + "/scripts/train_rnn.py", *argv]
    try: train_rnn.main()
    except Parsed: pass
print("V5R2_PARSED " + json.dumps(captured))
'''
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "PYTHONPATH": os.pathsep.join((str(extracted / "scripts"), str(extracted / "src"))),
    }
    training = subprocess.run(
        [sys.executable, "-c", parser_probe, str(extracted), json.dumps(argv_sets)],
        cwd=extracted,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert training.returncode == 0, training.stderr[-3000:]
    parsed_line = next(line for line in training.stdout.splitlines() if line.startswith("V5R2_PARSED "))
    rows = json.loads(parsed_line.removeprefix("V5R2_PARSED "))
    assert [row["hidden"] for row in rows] == [512, 512, 768, 768]
    assert all(row["batch"] == 8 and row["train_source"] == "train+valid" for row in rows)
    assert all(row["precision"] == "fp16" and row["amp_init_scale"] == 4096.0 for row in rows)
    assert [row["profile_steps"] for row in rows] == [20, 0, 20, 0]

    predictor_probe = r'''
import argparse, json, sys
class Parsed(Exception): pass
sys.path[:0] = [sys.argv[1] + "/scripts", sys.argv[1] + "/src"]
import predict_run
original = argparse.ArgumentParser.parse_args
captured = []
def stop_after_parse(self, *args, **kwargs):
    namespace = original(self, *args, **kwargs)
    captured.append({"name": namespace.name, "split": namespace.split,
                     "epoch": namespace.epoch, "inference_precision": namespace.inference_precision})
    raise Parsed
argparse.ArgumentParser.parse_args = stop_after_parse
for argv in json.loads(sys.argv[2]):
    sys.argv = [sys.argv[1] + "/scripts/predict_run.py", *argv]
    try: predict_run.main()
    except Parsed: pass
print("V5R2_PARSED " + json.dumps(captured))
'''
    prediction = subprocess.run(
        [sys.executable, "-c", predictor_probe, str(extracted), json.dumps(predictor_argv)],
        cwd=extracted,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert prediction.returncode == 0, prediction.stderr[-3000:]
    parsed_line = next(line for line in prediction.stdout.splitlines() if line.startswith("V5R2_PARSED "))
    rows = json.loads(parsed_line.removeprefix("V5R2_PARSED "))
    assert [row["name"] for row in rows] == ["v5-512", "v5-768"]
    assert all(row["split"] == "valid" and row["epoch"] == 3 for row in rows)
    assert all(row["inference_precision"] == "fp32" for row in rows)


def test_v5r2_runner_counts_page_cache_as_available(tmp_path, monkeypatch):
    """V5R1 failed admission with 28 GB of reclaimable hashing cache in memory.current."""
    extracted = _extract_archive(tmp_path / "project")
    runner = _load_module("v5r2_runner_memory", extracted / "scripts/kaggle_gpu_runner.py")
    cgroup = tmp_path / "cgroup"
    cgroup.mkdir()
    gib = 1024**3
    (cgroup / "memory.max").write_text(str(30 * gib))
    (cgroup / "memory.current").write_text(str(29 * gib))
    (cgroup / "memory.stat").write_text(f"anon {1 * gib}\nfile {28 * gib}\n")
    real_path = runner.Path

    def fake_path(value, *rest):
        text = str(value)
        if text.startswith("/sys/fs/cgroup/"):
            return real_path(cgroup / text.rsplit("/", 1)[1])
        return real_path(value, *rest)

    monkeypatch.setattr(runner, "Path", fake_path)
    monkeypatch.setattr(runner.psutil, "virtual_memory",
                        lambda: SimpleNamespace(total=32 * gib, available=31 * gib))
    host = runner.host_inventory()
    assert host["capacity_bytes"] == 30 * gib
    assert host["available_bytes"] == 29 * gib  # 30 GiB limit minus 1 GiB unreclaimable
