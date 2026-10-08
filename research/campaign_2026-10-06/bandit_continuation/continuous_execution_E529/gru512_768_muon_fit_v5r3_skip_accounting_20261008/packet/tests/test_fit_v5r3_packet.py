"""Focused V5R3 packet tests; no training data or GPU execution is performed."""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tarfile
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
        hashlib.sha256(
            (PACKET / "source_dataset_upload/assets-manifest.json").read_bytes()
        ).hexdigest(),
        [entry["path"] for entry in manifest["files"]],
    )
    ast.parse(source)
    assert '\'{"event": "campaign_resource_heartbeat"\'' in source
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
    recipe = json.loads((PACKET / "GRU512_768_MUON_FIT_RECIPE_V5R3.json").read_text())
    assert "fit-only" in recipe["status"]
    assert recipe["training"]["initial_batch"] == 8
    assert recipe["training"]["fallback_batch"] is None
    assert "no profile rerun" in recipe["training"]["batch_selection_rule"]
    assert "nonnegative GradScaler skips" in recipe["training"]["amp_skip_gate"]
    assert (
        "equal attempt budgets do not imply equal realized optimizer updates"
        in recipe["amp_skip_interpretation"]["cross_arm_interpretation"]
    )
    assert recipe["parent_packet"] == "E529_H232_GRU512_768_MUON_SEED9_FIT_V5R2_20261008"
    assert recipe["source"]["parent_source_commit"] == "b98ed5dfec9052f4a8ae310b31b531ce26cae96f"
    assert "fit-only" in recipe["runtime"]["initial_execution_scope"]
    assert recipe["input_cache_repair"]["effect_claim"].endswith(
        "does not establish that pages were evicted"
    )
    assert recipe["comparison_scope"]["selection_policy"].startswith(
        "valid local/cloud cross-validation is the primary"
    )
    assert "producer_evidence/v5r1/repair_evidence_manifest.json" in paths


def test_pair_cpu_allocation_matches_four_cpu_dual_t4_host() -> None:
    recipe = json.loads((PACKET / "GRU512_768_MUON_FIT_RECIPE_V5R3.json").read_text())
    assert recipe["runtime"]["cpu_threads"] == 2
    assert recipe["runtime"]["aggregate_cpu_threads"] == 4
    assert (
        "V4 profile timings used 4 CPU threads per arm"
        in recipe["runtime"]["cpu_allocation_change"]
    )
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


def test_memory_snapshot_reads_current_cgroup_membership() -> None:
    snapshot = capacity_worker._memory_snapshot()
    assert isinstance(snapshot["memory_current_bytes"], int)
    assert snapshot["memory_current_bytes"] >= 0
    assert isinstance(snapshot["system_available_bytes"], int)
    assert snapshot["system_available_bytes"] > 0
    maximum = snapshot["memory_max_bytes"]
    if maximum is not None:
        assert maximum > 0
        assert isinstance(snapshot["cgroup_available_bytes"], int)


def test_arm_specs_collect_the_single_input_and_page_cache_receipts(
    monkeypatch, tmp_path: Path
) -> None:
    project_root = tmp_path / "project"
    scripts = project_root / "scripts"
    scripts.mkdir(parents=True)
    portable = project_root / "research/campaign_2026-10-04/portable_inputs.json"
    portable.parent.mkdir(parents=True)
    portable.write_text(
        json.dumps(
            {
                "sources": {
                    logical: {"sha256": "a" * 64} for logical in capacity_fit_worker.INPUT_NAMES
                }
            }
        )
    )
    for logical in capacity_fit_worker.INPUT_NAMES:
        source = project_root / logical
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b"x")
    context_path = project_root / "fit_context.json"
    context_path.write_text(json.dumps({"source_hashes": {}}))
    binding_path = project_root / "fit_input_binding.json"
    binding_path.write_text("{}")
    monkeypatch.setitem(
        sys.modules,
        "kaggle_common",
        SimpleNamespace(canonical_sha256=lambda value: "c" * 64),
    )
    spec = capacity_fit_worker._make_arm_spec(
        project_root,
        "muon512",
        15_500,
        binding_path,
        "b" * 64,
        context_path,
        hashlib.sha256(context_path.read_bytes()).hexdigest(),
    )
    assert "input_identity_receipt.json" in spec["artifacts"]
    assert "input_cache_advice" in spec["artifacts"]


def test_fit_epoch_counter_contract_accepts_zero_and_rare_amp_skips() -> None:
    zero_skips = {
        "optimizer_attempts": 230,
        "optimizer_completed_updates": 230,
        "amp_skipped_updates": 0,
        "completed_batches": 23,
    }
    assert (
        capacity_worker._validate_fit_epoch_update_counters(
            zero_skips, expected_attempts=230, expected_batches=23
        )["interpretation"]
        == "zero_amp_skips_observed_and_budget_identity_valid"
    )

    producer_shaped_skips = {
        "optimizer_attempts": 62_320,
        "optimizer_completed_updates": 62_293,
        "amp_skipped_updates": 27,
        "completed_batches": 6_232,
    }
    accepted = capacity_worker._validate_fit_epoch_update_counters(
        producer_shaped_skips,
        expected_attempts=62_320,
        expected_batches=6_232,
    )
    assert accepted["status"] == "pass"
    assert accepted["optimizer_completed_updates"] == 62_293
    assert accepted["amp_skipped_updates"] == 27
    assert accepted["interpretation"] == "nonzero_amp_skips_recorded_and_budget_identity_valid"


@pytest.mark.parametrize(
    ("updates", "expected_attempts", "expected_batches"),
    [
        (
            {"optimizer_completed_updates": 2, "amp_skipped_updates": 0, "completed_batches": 1},
            2,
            1,
        ),
        ({"optimizer_attempts": 2, "optimizer_completed_updates": 1, "completed_batches": 1}, 2, 1),
        (
            {"optimizer_attempts": 2, "optimizer_completed_updates": 2, "amp_skipped_updates": 0},
            2,
            1,
        ),
        (
            {
                "optimizer_attempts": 2,
                "optimizer_completed_updates": 1,
                "amp_skipped_updates": 0,
                "completed_batches": 1,
            },
            2,
            1,
        ),
        (
            {
                "optimizer_attempts": 2,
                "optimizer_completed_updates": -1,
                "amp_skipped_updates": 3,
                "completed_batches": 1,
            },
            2,
            1,
        ),
        (
            {
                "optimizer_attempts": 2,
                "optimizer_completed_updates": 2,
                "amp_skipped_updates": -1,
                "completed_batches": 1,
            },
            2,
            1,
        ),
        (
            {
                "optimizer_attempts": 2,
                "optimizer_completed_updates": 0,
                "amp_skipped_updates": 2,
                "completed_batches": 1,
            },
            2,
            1,
        ),
        (
            {
                "optimizer_attempts": 2,
                "optimizer_completed_updates": 1,
                "amp_skipped_updates": 1,
                "completed_batches": 1,
            },
            3,
            1,
        ),
    ],
)
def test_fit_epoch_counter_contract_rejects_missing_or_contradictory_values(
    updates: dict[str, object], expected_attempts: int, expected_batches: int
) -> None:
    with pytest.raises(ValueError):
        capacity_worker._validate_fit_epoch_update_counters(
            updates,
            expected_attempts=expected_attempts,
            expected_batches=expected_batches,
        )


def test_periodic_fit_loss_samples_are_finite_complete_and_ordered() -> None:
    samples = [
        {
            "epoch": epoch,
            "batch": batch,
            "total_batches": 41,
            "loss_text": "0.1250",
        }
        for epoch in range(4)
        for batch in range(0, 41, 20)
    ]
    receipt = capacity_worker._validate_fit_loss_samples(
        samples, expected_epochs=4, expected_batches=41
    )
    assert receipt["status"] == "pass"
    assert receipt["sample_count"] == 12
    assert receipt["samples_per_epoch"] == 3
    assert "not a per-optimizer-step" in receipt["scope"]

    nonfinite = [dict(sample) for sample in samples]
    nonfinite[0]["loss_text"] = "nan"
    with pytest.raises(ValueError, match="nonfinite"):
        capacity_worker._validate_fit_loss_samples(
            nonfinite, expected_epochs=4, expected_batches=41
        )
    with pytest.raises(ValueError, match="incomplete"):
        capacity_worker._validate_fit_loss_samples(
            samples[:-1], expected_epochs=4, expected_batches=41
        )


def test_fit_progress_parser_retains_only_sampled_loss_telemetry() -> None:
    parsed = capacity_worker._parse_child_line(
        "ep 2 batch 40/41 loss 0.1250 lr 1.00e-3 seq/s 12.3",
        "muon512",
        "fit",
        time.monotonic(),
    )
    assert parsed["fit_loss_sample"] == {
        "epoch": 2,
        "batch": 40,
        "total_batches": 41,
        "loss_text": "0.1250",
    }


def test_fit_authorization_requires_both_widths(tmp_path: Path) -> None:
    recipe_path = tmp_path / "GRU512_768_MUON_FIT_RECIPE_V5R3.json"
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


def test_source_archive_is_exactly_reused_from_v5r2(tmp_path: Path) -> None:
    base_archive = REPO / build_fit_archive.V5R2_PACKET_RELATIVE / "source/source_archive.tar"
    base_manifest = (
        REPO / build_fit_archive.V5R2_PACKET_RELATIVE / "source/source_archive_manifest.json"
    )
    assert (
        hashlib.sha256(base_archive.read_bytes()).hexdigest()
        == build_fit_archive.V5R2_ARCHIVE_SHA256
    )
    assert (
        hashlib.sha256(base_manifest.read_bytes()).hexdigest()
        == build_fit_archive.V5R2_MANIFEST_SHA256
    )

    packet_root = tmp_path / "packet"
    produced, manifest_path = build_fit_archive._archive_source(packet_root, REPO)
    assert produced.read_bytes() == base_archive.read_bytes()
    assert manifest_path.read_bytes() == base_manifest.read_bytes()
    assert (
        packet_root / "producer_evidence/v5r2/source_archive.tar"
    ).read_bytes() == base_archive.read_bytes()
    assert (
        packet_root / "producer_evidence/v5r2/source_archive_manifest.json"
    ).read_bytes() == base_manifest.read_bytes()
    assert (packet_root / "producer_evidence/v5r2/page_cache_probe_20261008_r1.json").is_file()

    with tarfile.open(produced, "r") as archive:
        members = archive.getmembers()
    assert all(member.isfile() for member in members)
    assert {member.name for member in members} == {
        row["path"] for row in json.loads(base_manifest.read_text())["members"]
    }


def test_runner_inventory_and_ram_gate_use_post_hash_memory_fields(
    monkeypatch, tmp_path: Path
) -> None:
    extracted = _extract_archive(tmp_path / "source")
    monkeypatch.syspath_prepend(str(extracted / "scripts"))
    runner = _load_module(
        "v5_archived_kaggle_gpu_runner", extracted / "scripts/kaggle_gpu_runner.py"
    )
    host = runner.host_inventory()
    assert isinstance(host["system_available_bytes"], int)
    assert host["available_bytes"] <= host["capacity_bytes"]
    assert "cgroup_current_bytes" in host
    assert "cgroup_max_bytes" in host
    assert "cgroup_available_bytes" in host

    class SyntheticPath:
        def __init__(self, value: str) -> None:
            self.value = value

        def exists(self) -> bool:
            return True

        def read_text(self) -> str:
            return {
                "/sys/fs/cgroup/memory.max": str(30 * 1024**3),
                "/sys/fs/cgroup/memory.current": str(10 * 1024**3),
            }[self.value]

    with monkeypatch.context() as scoped:
        scoped.setattr(
            runner.psutil,
            "virtual_memory",
            lambda: SimpleNamespace(total=20 * 1024**3, available=12 * 1024**3),
        )
        scoped.setattr(runner, "Path", SyntheticPath)
        measured = runner.host_inventory()
    assert measured["capacity_bytes"] == 20 * 1024**3
    assert measured["system_available_bytes"] == 12 * 1024**3
    assert measured["cgroup_current_bytes"] == 10 * 1024**3
    assert measured["cgroup_max_bytes"] == 30 * 1024**3
    assert measured["cgroup_available_bytes"] == 20 * 1024**3
    assert measured["available_bytes"] == 10 * 1024**3

    available_cpus = sorted(os.sched_getaffinity(0))
    assert len(available_cpus) >= 4
    healthy_host = {
        "capacity_bytes": 30 * 1024**3,
        "available_bytes": 30 * 1024**3,
        "system_available_bytes": 30 * 1024**3,
        "cgroup_current_bytes": 0,
        "cgroup_max_bytes": 30 * 1024**3,
        "cgroup_available_bytes": 30 * 1024**3,
        "affinity": available_cpus[:4],
    }
    devices = [
        {
            "index": 0,
            "uuid": "GPU-A",
            "name": "Tesla T4",
            "total_bytes": 16 * 1024**3,
            "used_bytes": 0,
        },
        {
            "index": 1,
            "uuid": "GPU-B",
            "name": "Tesla T4",
            "total_bytes": 16 * 1024**3,
            "used_bytes": 0,
        },
    ]
    runner.gpu_inventory = lambda: devices
    pair_specs = [
        {"limits": {"cpus": 2, "ram_bytes": 8 * 1024**3, "gpu_stop_bytes": 12 * 1024**3}},
        {"limits": {"cpus": 2, "ram_bytes": 8 * 1024**3, "gpu_stop_bytes": 12 * 1024**3}},
    ]
    assert (
        runner.admit_specs(
            pair_specs,
            healthy_host,
            devices,
            maximum_preexisting_gpu_used_bytes=512 * 1024**2,
        )
        == 2
    )
    host_three_cpu = {**healthy_host, "affinity": healthy_host["affinity"][:3]}
    assert (
        runner.admit_specs(
            pair_specs,
            host_three_cpu,
            devices,
            maximum_preexisting_gpu_used_bytes=512 * 1024**2,
        )
        == 1
    )

    # Exact late V5R1 inventory: the unchanged 20% host reserve makes this a pre-launch rejection.
    v5r1_late_host = {
        **healthy_host,
        "capacity_bytes": 32_212_254_720,
        "available_bytes": 4_224_753_664,
        "cgroup_current_bytes": 27_987_501_056,
        "cgroup_max_bytes": 32_212_254_720,
        "cgroup_available_bytes": 4_224_753_664,
    }
    with pytest.raises(ValueError, match="Worker reservation exceeds host operational budget"):
        runner.admit_specs(
            pair_specs,
            v5r1_late_host,
            devices,
            maximum_preexisting_gpu_used_bytes=512 * 1024**2,
        )

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
            "code": [
                {"path": "scripts/probe.py", "size": script.stat().st_size, "sha256": script_hash}
            ],
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
    runner.host_inventory = lambda: healthy_host
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


def test_full_file_hash_advises_bounded_windows_without_changing_input(
    monkeypatch, tmp_path: Path
) -> None:
    path = tmp_path / "large-immutable-input.bin"
    block = bytes(index % 251 for index in range(1 << 20))
    digest = hashlib.sha256()
    total_bytes = 192 * 1024**2
    with path.open("wb") as target:
        for _ in range(total_bytes // len(block)):
            target.write(block)
            digest.update(block)
    expected_sha256 = digest.hexdigest()
    before = capacity_worker.FileIdentity.from_stat(path.stat())
    assert os.path.isfile(path) and os.path.getsize(path) == total_bytes

    original_fadvise = os.posix_fadvise
    calls: list[tuple[int, int]] = []

    def recording_fadvise(file_descriptor: int, offset: int, length: int, advice: int):
        calls.append((offset, length))
        return original_fadvise(file_descriptor, offset, length, advice)

    monkeypatch.setattr(capacity_worker.os, "posix_fadvise", recording_fadvise)
    receipt_path = tmp_path / "input_cache_advice.json"
    handoff = capacity_worker.VerifiedDigestHandoff((path,))
    adviser = capacity_worker.BoundedPageCacheAdviser(
        path,
        "datasets/train.parquet",
        receipt_path,
        expected_sha256,
        total_bytes,
    )
    observed = handoff.hash_file(path, on_progress=adviser.on_progress)
    adviser.complete(observed)
    handoff.authorize(
        observed,
        expected_sha256=expected_sha256,
        registry_sha256=expected_sha256,
    )

    after = capacity_worker.FileIdentity.from_stat(path.stat())
    record = json.loads(receipt_path.read_text())
    page_size = os.sysconf("SC_PAGE_SIZE")
    assert observed.sha256 == expected_sha256
    assert observed.bytes_hashed == total_bytes
    assert before == after == observed.identity
    assert len(calls) == total_bytes // capacity_worker.PAGE_CACHE_ADVICE_WINDOW_BYTES
    assert all(offset % page_size == 0 for offset, _ in calls)
    assert all(length % page_size == 0 for _, length in calls)
    assert all(length <= capacity_worker.PAGE_CACHE_ADVICE_WINDOW_BYTES for _, length in calls)
    assert sum(length for _, length in calls) == record["bytes_advised"] == total_bytes
    assert record["status"] == "verified"
    assert record["digest_matches_expected"] is True
    assert record["advice_semantics"].endswith("does not prove eviction")
    assert record["memory_samples"][-1]["bytes_hashed"] == total_bytes


def test_child_environment_sampler_and_run_child_keep_device_and_timing(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-B")
    monkeypatch.setenv("WNN_PHYSICAL_GPU_UUID", "GPU-B")
    monkeypatch.setenv("WNN_PHYSICAL_GPU_INDEX", "1")
    monkeypatch.setattr(capacity_worker, "FINALIZATION_RESERVE_SECONDS", 0)
    env = capacity_worker._child_env(
        tmp_path, "muon768", "fit", tmp_path / "receipts", tmp_path / "identity.json"
    )
    assert (
        env["CUDA_VISIBLE_DEVICES"],
        env["WNN_PHYSICAL_GPU_UUID"],
        env["WNN_PHYSICAL_GPU_INDEX"],
    ) == (
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
        'print(\'GRU_CAPACITY_CHILD_STAGE {"stage":"trainer_import_done"}\', flush=True)\n',
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
        predictor_argv = [capacity_worker._predict_argv(arm["run_name"], split) for arm in arms]
    finally:
        sys.path.remove(str(scripts))

    parser_probe = r"""
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
print("V5R3_PARSED " + json.dumps(captured))
"""
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
    parsed_line = next(
        line for line in training.stdout.splitlines() if line.startswith("V5R3_PARSED ")
    )
    rows = json.loads(parsed_line.removeprefix("V5R3_PARSED "))
    assert [row["hidden"] for row in rows] == [512, 512, 768, 768]
    assert all(row["batch"] == 8 and row["train_source"] == "train+valid" for row in rows)
    assert all(row["precision"] == "fp16" and row["amp_init_scale"] == 4096.0 for row in rows)
    assert [row["profile_steps"] for row in rows] == [20, 0, 20, 0]

    predictor_probe = r"""
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
print("V5R3_PARSED " + json.dumps(captured))
"""
    prediction = subprocess.run(
        [sys.executable, "-c", predictor_probe, str(extracted), json.dumps(predictor_argv)],
        cwd=extracted,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert prediction.returncode == 0, prediction.stderr[-3000:]
    parsed_line = next(
        line for line in prediction.stdout.splitlines() if line.startswith("V5R3_PARSED ")
    )
    rows = json.loads(parsed_line.removeprefix("V5R3_PARSED "))
    assert [row["name"] for row in rows] == ["v5-512", "v5-768"]
    assert all(row["split"] == "valid" and row["epoch"] == 3 for row in rows)
    assert all(row["inference_precision"] == "fp32" for row in rows)
