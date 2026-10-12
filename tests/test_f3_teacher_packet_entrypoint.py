from __future__ import annotations

import importlib.util
import json
import os
import shutil
import tarfile
from pathlib import Path


PACKET = Path("research/campaign_2026-10-13/strategy_execution_r1/first_cohort_f3_teachers_r1")
BUNDLE = PACKET / "bundle_f3_teacher_seed3_r1"


def test_sealed_kernel_entrypoint_dispatches_profile_guarded_driver_offline(tmp_path, monkeypatch):
    manifest = json.loads((BUNDLE / "bundle-manifest.json").read_text())
    metadata = json.loads((BUNDLE / "kernel/kernel-metadata.json").read_text())
    binding = json.loads((PACKET / "ENTRYPOINT_BINDING_R1.json").read_text())
    kernel = BUNDLE / "kernel/kernel.py"

    assert metadata["id"] == "vladkochetov07/wnn-strategy-f3-t3-seed3-20261013-r1-runner"
    assert metadata["title"] == "wnn-strategy-f3-t3-seed3-20261013-r1 runner"
    assert binding["title_derived_slug"] == manifest["dataset"].split("/")[1]
    assert binding["kernel_sha256"] == __import__("hashlib").sha256(kernel.read_bytes()).hexdigest()
    assert manifest["archive_sha256"] == binding["archive_sha256"]

    archive_path = BUNDLE / "bundle.wnn"
    with tarfile.open(archive_path, "r:gz") as archive:
        assert "research/campaign_2026-10-13/strategy_execution_r1/first_cohort_f3_teachers_r1/kernel_support/f3_teacher_driver.py" in archive.getnames()

    input_root = tmp_path / "kaggle/input"
    working_root = tmp_path / "kaggle/working"
    working_root.mkdir(parents=True)
    bundle_mount = input_root / manifest["dataset"].split("/")[1]
    bundle_mount.mkdir(parents=True)
    shutil.copy2(archive_path, bundle_mount / "bundle.wnn")
    shutil.copy2(BUNDLE / "bundle-manifest.json", bundle_mount / "bundle-manifest.json")
    for logical, mapping in manifest["external_mounts"].items():
        if isinstance(mapping, dict):
            dataset, filename = mapping["dataset"], mapping["filename"]
        else:
            dataset, filename = "vladkochetov07/wunder-funder", mapping
        mount = input_root / dataset.split("/")[1]
        mount.mkdir(parents=True, exist_ok=True)
        (mount / filename).write_bytes(b"offline mount fixture")

    spec = importlib.util.spec_from_file_location("sealed_f3_kernel", kernel)
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
    monkeypatch.setattr(module.shutil, "disk_usage", lambda _path: type("Usage", (), {"free": 50 * 1024**3})())
    module.main()

    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command[1].endswith("first_cohort_f3_teachers_r1/kernel_support/f3_teacher_driver.py")
    assert not any(argument == "--spec" for argument in command)
    assert kwargs["env"]["WNN_KERNEL_DEADLINE_MONOTONIC"]
    extracted = tmp_path / "kaggle/working/experiment"
    assert (extracted / PACKET / "kernel_support/f3_teacher_driver.py").is_file()
    assert all(not (extracted / logical).is_symlink() for logical in manifest["external_mounts"])
