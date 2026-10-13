from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import tarfile
from pathlib import Path

PACKET = Path("research/campaign_2026-10-13/strategy_execution_r1/first_cohort_f3_teachers_r3")
BUNDLE = PACKET / "bundle_f3_teacher_seed3_r3"


def test_r3_sealed_bootstrap_dispatches_profile_reuse_driver_offline(tmp_path, monkeypatch):
    manifest = json.loads((BUNDLE / "bundle-manifest.json").read_text())
    metadata = json.loads((BUNDLE / "kernel/kernel-metadata.json").read_text())
    binding = json.loads((PACKET / "ENTRYPOINT_BINDING_R3.json").read_text())
    kernel = BUNDLE / "kernel/kernel.py"

    assert metadata["id"] == "vladkochetov07/wnn-strategy-f3-t3-seed3-20261013-r3-runner"
    assert metadata["title"] == "wnn-strategy-f3-t3-seed3-20261013-r3 runner"
    assert binding["title_derived_slug"] == manifest["dataset"].split("/")[1]
    assert binding["kernel_sha256"] == hashlib.sha256(kernel.read_bytes()).hexdigest()
    assert manifest["archive_sha256"] == binding["archive_sha256"]

    archive_path = BUNDLE / "bundle.wnn"
    driver_path = f"{PACKET.as_posix()}/kernel_support/f3_teacher_driver_r3.py"
    reuse_path = f"{PACKET.as_posix()}/kernel_support/profile_reuse.py"
    with tarfile.open(archive_path, "r:gz") as archive:
        names = archive.getnames()
        assert driver_path in names
        assert reuse_path in names
        assert f"{PACKET.as_posix()}/PROFILE_REUSE_BINDING_R3.json" in names
        assert f"{PACKET.as_posix()}/profile_reuse_r2/profile_worker_r2.log" in names

    job_specs = sorted((PACKET / "jobs").glob("*.json"))
    assert len(job_specs) == 2
    names = {json.loads(path.read_text())["name"] for path in job_specs}
    assert names == {
        "fit_clean_t3_f3_muonall_h384_e3_s3_r3",
        "export_clean_t3_f3_muonall_h384_e3_s3_r3_fit_train",
    }
    assert all(
        json.loads(path.read_text())["limits"]["ram_bytes"] == 6_000_000_000 for path in job_specs
    )
    export_spec = next(
        json.loads(path.read_text()) for path in job_specs if path.name.startswith("export_")
    )
    assert len(export_spec["artifacts"]) == 2

    input_root = tmp_path / "kaggle/input"
    working_root = tmp_path / "kaggle/working"
    working_root.mkdir(parents=True)
    bundle_mount = input_root / manifest["dataset"].split("/")[1]
    bundle_mount.mkdir(parents=True)
    shutil.copy2(archive_path, bundle_mount / "bundle.wnn")
    shutil.copy2(BUNDLE / "bundle-manifest.json", bundle_mount / "bundle-manifest.json")
    for mapping in manifest["external_mounts"].values():
        if isinstance(mapping, dict):
            dataset, filename = mapping["dataset"], mapping["filename"]
        else:
            dataset, filename = "vladkochetov07/wunder-funder", mapping
        mount = input_root / dataset.split("/")[1]
        target = mount / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"offline mount fixture")

    spec = importlib.util.spec_from_file_location("sealed_f3_r3_kernel", kernel)
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
            return 0

    def fake_popen(command, **kwargs):
        calls.append((command, kwargs))
        return CompletedProcess()

    monkeypatch.setattr(module, "Path", mapped_path)
    monkeypatch.setattr(module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        module.shutil,
        "disk_usage",
        lambda _path: type("Usage", (), {"free": 50 * 1024**3})(),
    )
    module.main()

    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command[1].endswith(driver_path)
    assert "--spec" not in command
    assert kwargs["env"]["WNN_KERNEL_DEADLINE_MONOTONIC"]
    assert kwargs["env"]["WNN_PROJECT_ROOT"].endswith("/experiment")
    extracted = tmp_path / "kaggle/working/experiment"
    assert (extracted / PACKET / "kernel_support/profile_reuse.py").is_file()
    assert all(not (extracted / logical).is_symlink() for logical in manifest["external_mounts"])
