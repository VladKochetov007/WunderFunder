from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import tarfile
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "research/campaign_2026-10-13/strategy_execution_r1/first_cohort_f3_teachers_r2"
R1 = ROOT / "research/campaign_2026-10-13/strategy_execution_r1/first_cohort_f3_teachers_r1"
BUNDLE = PACKET / "bundle_f3_teacher_seed3_r2"


def test_sealed_r2_bootstrap_uses_nested_mounts_and_actual_driver(tmp_path, monkeypatch):
    manifest = json.loads((BUNDLE / "bundle-manifest.json").read_text())
    metadata = json.loads((BUNDLE / "kernel/kernel-metadata.json").read_text())
    binding = json.loads((PACKET / "ENTRYPOINT_BINDING_R2.json").read_text())
    kernel = BUNDLE / "kernel/kernel.py"

    assert metadata["id"] == "vladkochetov07/wnn-strategy-f3-t3-seed3-20261013-r2-runner"
    assert metadata["title"] == "wnn-strategy-f3-t3-seed3-20261013-r2 runner"
    assert binding["title_derived_slug"] == manifest["dataset"].split("/")[1]
    assert binding["kernel_sha256"] == hashlib.sha256(kernel.read_bytes()).hexdigest()
    assert binding["archive_sha256"] == manifest["archive_sha256"]

    driver_path = f"{PACKET.relative_to(ROOT).as_posix()}/kernel_support/f3_teacher_driver_r2.py"
    with tarfile.open(BUNDLE / "bundle.wnn", "r:gz") as archive:
        assert driver_path in archive.getnames()

    input_root = tmp_path / "kaggle/input"
    working_root = tmp_path / "kaggle/working"
    working_root.mkdir(parents=True)
    bundle_mount = (
        input_root / "nested-bundle-owner" / "vladkochetov07" / manifest["dataset"].split("/")[1]
    )
    bundle_mount.mkdir(parents=True)
    shutil.copy2(BUNDLE / "bundle.wnn", bundle_mount / "bundle.wnn")
    shutil.copy2(BUNDLE / "bundle-manifest.json", bundle_mount / "bundle-manifest.json")
    nested_mounts = {}
    for logical, mapping in manifest["external_mounts"].items():
        if isinstance(mapping, dict):
            dataset, filename = mapping["dataset"], mapping["filename"]
        else:
            dataset, filename = "vladkochetov07/wunder-funder", mapping
        owner, slug = dataset.split("/")
        mount = input_root / "nested-assets" / owner / slug
        mount.mkdir(parents=True, exist_ok=True)
        (mount / filename).write_bytes(b"nested offline mount fixture")
        nested_mounts[logical] = mount / filename

    spec = importlib.util.spec_from_file_location("sealed_f3_kernel_r2", kernel)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    real_path = Path

    def mapped_path(value):
        text = os.fspath(value)
        if text.startswith("/kaggle/"):
            return real_path(tmp_path / "kaggle" / text.removeprefix("/kaggle/"))
        return real_path(value)

    calls = []

    class CompletedProcess:
        def wait(self, timeout=None):
            assert timeout is not None and timeout > 0
            return 0

    def fake_popen(command, **kwargs):
        calls.append((command, kwargs))
        mounted_root = kwargs["cwd"]
        for logical, expected_target in nested_mounts.items():
            path = mounted_root / logical
            assert path.is_symlink()
            assert path.resolve() == expected_target
        return CompletedProcess()

    monkeypatch.setattr(module, "Path", mapped_path)
    monkeypatch.setattr(module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        module.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=50 * 1024**3),
    )
    module.main()

    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command[0] == module.sys.executable
    assert command[1].endswith(driver_path)
    assert not any(argument == "--spec" for argument in command)
    assert kwargs["env"]["WNN_PROJECT_ROOT"] == str(tmp_path / "kaggle/working/experiment")
    started = float(kwargs["env"]["WNN_KERNEL_STARTED_MONOTONIC"])
    deadline = float(kwargs["env"]["WNN_KERNEL_DEADLINE_MONOTONIC"])
    assert deadline - started == module.KERNEL_PROCESS_WALL_SECONDS_MAX
    extracted = tmp_path / "kaggle/working/experiment"
    assert (extracted / driver_path).is_file()
    assert all(not (extracted / logical).is_symlink() for logical in manifest["external_mounts"])


def test_r2_recipe_bindings_split_loader_and_prediction_geometry(tmp_path):
    specs = {path.stem: json.loads(path.read_text()) for path in (PACKET / "jobs").glob("*.json")}
    fit_name = "fit_clean_t3_f3_muonall_h384_e3_s3_r2"
    fit = specs[fit_name]
    argv = fit["argv"]
    environment = fit["environment"]
    assert environment["WNN_PROJECT_ROOT"] == "/kaggle/working/experiment"
    assert argv[argv.index("--seed") + 1] == "3"
    assert argv[argv.index("--valid-fold") + 1] == "3"
    assert (
        argv[argv.index("--split-manifest") + 1]
        == "research/campaign_2026-09-29/artifacts/splits/fold3.json"
    )
    assert argv[argv.index("--checkpoint-epoch") + 1] == "2"
    assert argv[argv.index("--epochs") + 1] == "3"
    assert argv[argv.index("--muon-scope") + 1] == "all"
    assert "--resume" not in argv
    assert "--init" not in argv

    for name, spec in specs.items():
        assert spec["environment"]["WNN_PROJECT_ROOT"] == "/kaggle/working/experiment"
        if name.startswith("export_"):
            assert "--campaign-split-manifest" in spec["argv"]
            assert spec["argv"][spec["argv"].index("--inference-precision") + 1] == "fp32"
            assert spec["argv"][spec["argv"].index("--epoch") + 1] == "2"

    split_path = ROOT / "research/campaign_2026-09-29/artifacts/splits/fold3.json"
    split = json.loads(split_path.read_text())
    input_binding = json.loads((R1 / "INPUT_BINDING_R1.json").read_text())
    assert hashlib.sha256(split_path.read_bytes()).hexdigest() == input_binding["split"]["sha256"]
    assert split["fold"] == 3
    assert split["counts"]["fit_train"] == 10539
    assert split["counts"]["fit_valid"] == 1368
    assert split["counts"]["evaluation_valid"] == 505

    split_source = R1 / "source_snapshot/src/wnn/splits.py"
    split_spec = importlib.util.spec_from_file_location("f3_actual_split_loader", split_source)
    assert split_spec is not None and split_spec.loader is not None
    split_module = importlib.util.module_from_spec(split_spec)
    split_spec.loader.exec_module(split_module)

    cache = tmp_path / "cache"
    cache.mkdir()
    chains_path = cache / "chains.npz"
    folds_path = cache / "chain_folds.npz"
    np.savez(
        chains_path,
        next=np.array([-1, -1, -1], dtype=np.int64),
        split=np.array(["train", "valid", "valid"]),
        row_group=np.array([0, 0, 1], dtype=np.int64),
    )
    np.savez(
        folds_path,
        split=np.array(["valid", "valid"]),
        row_group=np.array([0, 1], dtype=np.int64),
        fold=np.array([0, 3], dtype=np.int64),
    )

    def source_identity(path: Path) -> dict:
        stat = path.stat()
        return {
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    fixture = {
        "version": 1,
        "fold": 3,
        "n_folds": 4,
        "sources": {
            "cache/chains.npz": source_identity(chains_path),
            "cache/chain_folds.npz": source_identity(folds_path),
        },
        "sequences": {
            "train": [{"row_group": 0, "component": 0}],
            "valid": [
                {"row_group": 0, "component": 1},
                {"row_group": 1, "component": 2},
            ],
        },
        "fit": {"train": [0], "valid": [0]},
        "evaluation": {"valid": [1]},
    }
    manifest = tmp_path / "fold3.json"
    manifest.write_text(json.dumps(fixture))
    loaded = split_module.load_campaign_split(manifest, tmp_path)
    assert loaded.fold == 3
    assert loaded.train_groups == (0,)
    assert loaded.valid_groups == (0,)
    assert loaded.evaluation_groups == (1,)

    plan = json.loads((PACKET / "OUTPUT_COLLECTION_PLAN_R2.json").read_text())
    geometry = plan["prediction_geometry"]
    assert geometry["artifacts"]["fit_train"]["raw_npy_payload_bytes"] == 1686240000
    assert geometry["artifacts"]["fit_train"]["maximum_npy_file_bytes"] == 1686244096
    assert geometry["artifacts"]["fit_valid"]["raw_npy_payload_bytes"] == 218880000
    assert geometry["artifacts"]["fit_valid"]["maximum_npy_file_bytes"] == 218884096
    assert geometry["artifacts"]["full_valid"]["raw_npy_payload_bytes"] == 299680000
    assert geometry["artifacts"]["full_valid"]["maximum_npy_file_bytes"] == 299684096
    assert (
        sum(item["raw_npy_payload_bytes"] for item in geometry["artifacts"].values()) == 2204800000
    )
    assert geometry["raw_payload_bytes_for_three_seeds"] == 6614400000


def test_r2_quota_guard_reserves_full_tranche_with_fresh_snapshot(tmp_path):
    driver_path = PACKET / "kernel_support/f3_teacher_driver_r2.py"
    spec = importlib.util.spec_from_file_location("f3_teacher_driver_r2_test", driver_path)
    assert spec is not None and spec.loader is not None
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)

    packet_root = tmp_path / driver.PACKET
    packet_root.mkdir(parents=True)
    guard = json.loads((PACKET / "KAGGLE_QUOTA_GUARD_R2.json").read_text())
    guard["quota_snapshot"]["checked_utc"] = datetime.now(UTC).isoformat()
    (packet_root / "KAGGLE_QUOTA_GUARD_R2.json").write_text(json.dumps(guard))
    helpers = SimpleNamespace(remaining_seconds=lambda deadline: deadline - 100.0)
    admission = driver.quota_admission(tmp_path, helpers, 100.0, 43300.0)
    assert admission["quota_derived_wall_window_seconds"] == 17019
    assert admission["maximum_planned_stage_plus_finalization_seconds"] == 15700
    assert admission["effective_deadline_monotonic"] == 17119.0

    guard["quota_snapshot"]["checked_utc"] = "2000-01-01T00:00:00+00:00"
    (packet_root / "KAGGLE_QUOTA_GUARD_R2.json").write_text(json.dumps(guard))
    with pytest.raises(RuntimeError, match="quota snapshot is stale"):
        driver.quota_admission(tmp_path, helpers, 100.0, 43300.0)
