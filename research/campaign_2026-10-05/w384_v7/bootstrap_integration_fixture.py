"""Exercise the assembled v7 bootstrap with a real archive and intercepted phase launch."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
import shutil
import tarfile
import tempfile
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
BUNDLE = HERE / "bundle"
KERNEL_PATH = BUNDLE / "kernel/kernel.py"
spec = importlib.util.spec_from_file_location("w384_v7_kernel_fixture", KERNEL_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError("Could not import assembled v7 kernel")
kernel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kernel)


def make_sandbox(base: Path, bundle: Path) -> tuple[Path, Path, Path]:
    input_root = base / "input"
    working = base / "working"
    working.mkdir(parents=True)
    provider_bundle = input_root / "owner" / kernel.BUNDLE_SLUG
    provider_bundle.mkdir(parents=True)
    shutil.copyfile(bundle / "bundle-manifest.json", provider_bundle / "bundle-manifest.json")
    shutil.copyfile(bundle / "bundle.wnn", provider_bundle / "bundle.wnn")
    manifest = json.loads((provider_bundle / "bundle-manifest.json").read_text())
    for logical, descriptor in manifest["external_mounts"].items():
        if isinstance(descriptor, dict):
            dataset, filename = descriptor["dataset"], descriptor["filename"]
        else:
            dataset, filename = "vladkochetov07/wunder-funder", descriptor
        owner, slug = dataset.split("/")
        source = input_root / owner / slug / filename
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b"external-fixture:" + logical.encode())
    return input_root, working, provider_bundle


def run_bootstrap(bundle: Path, *, collision: bool = False) -> tuple[bool, list[list[str]]]:
    with tempfile.TemporaryDirectory(prefix="w384-v7-bootstrap-") as tmp:
        base = Path(tmp)
        staged_bundle = base / "bundle"
        staged_bundle.mkdir()
        manifest = json.loads((bundle / "bundle-manifest.json").read_text())
        shutil.copyfile(bundle / "bundle-manifest.json", staged_bundle / "bundle-manifest.json")
        shutil.copyfile(bundle / "bundle.wnn", staged_bundle / "bundle.wnn")
        expected_archive = kernel.EXPECTED_ARCHIVE_SHA256
        if collision:
            first_external = next(iter(manifest["external_mounts"]))
            original = staged_bundle / "bundle.wnn"
            altered = staged_bundle / "bundle.with-collision.wnn"
            with tarfile.open(original, "r:gz") as src, tarfile.open(altered, "w:gz") as dst:
                for member in src.getmembers():
                    payload = src.extractfile(member)
                    if payload is None:
                        raise AssertionError(f"Unexpected non-file archive member {member.name}")
                    dst.addfile(member, payload)
                info = tarfile.TarInfo(first_external)
                payload = b"deliberate-archive-collision"
                info.size = len(payload)
                dst.addfile(info, io.BytesIO(payload))
            altered.replace(original)
            digest = hashlib.sha256(original.read_bytes()).hexdigest()
            manifest["archive_sha256"] = digest
            (staged_bundle / "bundle-manifest.json").write_text(json.dumps(manifest))
            expected_archive = digest
        input_root, working, provider_bundle = make_sandbox(base, staged_bundle)

        def mapped_path(value: object) -> Path:
            raw = str(value)
            if raw == "/kaggle/input":
                return input_root
            if raw == "/kaggle/working/experiment":
                return working / "experiment"
            return Path(value)

        calls: list[list[str]] = []
        def intercepted(command: list[str], check: bool = False) -> None:
            assert check is True
            assert Path(command[1]).is_file()
            experiment = working / "experiment"
            assert all((experiment / logical).is_symlink() for logical in manifest["external_mounts"])
            calls.append(command)
            return None

        original_path = kernel.Path
        original_digest = kernel.EXPECTED_ARCHIVE_SHA256
        kernel.Path = mapped_path
        kernel.EXPECTED_ARCHIVE_SHA256 = expected_archive
        try:
            with patch.object(kernel.subprocess, "run", intercepted):
                try:
                    kernel._kernel_work()
                except ValueError as exc:
                    if not collision or "overlaps an archive member" not in str(exc):
                        raise
                    return False, calls
                if collision:
                    raise AssertionError("Bootstrap accepted a deliberately colliding archive member")
                assert len(calls) == 1
                assert calls[0][-1].endswith("/phase_runner.py")
                experiment = working / "experiment"
                assert all(not (experiment / logical).is_symlink() for logical in manifest["external_mounts"])
                return True, calls
        finally:
            kernel.Path = original_path
            kernel.EXPECTED_ARCHIVE_SHA256 = original_digest


# Prove the actual assembled archive has 265 members while the complete input
# inventory has 287 entries; its 26 external descriptors are absent from tar.
manifest = json.loads((BUNDLE / "bundle-manifest.json").read_text())
manifest_paths = {item["path"] for item in manifest["files"]}
with tarfile.open(BUNDLE / "bundle.wnn", "r:gz") as archive:
    actual_members = {member.name for member in archive.getmembers() if member.isfile()}
external_paths = set(manifest["external_mounts"])
assert len(manifest_paths) == 287
assert len(actual_members) == 265
assert len(external_paths) == 26
assert len(manifest_paths & external_paths) == 26
assert not (actual_members & external_paths)

ok, calls = run_bootstrap(BUNDLE)
assert ok and len(calls) == 1
rejected, collision_calls = run_bootstrap(BUNDLE, collision=True)
assert not rejected and not collision_calls

# Keep independent path and exact-mount-map rejection guards around the fix.
with tempfile.TemporaryDirectory(prefix="w384-v7-mount-guard-") as tmp:
    root = Path(tmp)
    traversal = {"../escape": {"dataset": "owner/dataset", "filename": "x"}}
    try:
        kernel.validate_external_mounts(root, traversal, set(), traversal)
    except ValueError:
        pass
    else:
        raise AssertionError("Traversal mount path was accepted")
    mismatch = dict(kernel.EXPECTED_EXTERNAL_MOUNTS)
    mismatch.pop(next(iter(mismatch)))
    try:
        kernel.validate_external_mounts(root, mismatch, set(), kernel.EXPECTED_EXTERNAL_MOUNTS)
    except ValueError:
        pass
    else:
        raise AssertionError("External mount allow-map mismatch was accepted")

print(json.dumps({
    "status": "passed",
    "manifest_inventory_count": len(manifest_paths),
    "actual_archive_count": len(actual_members),
    "external_mount_count": len(external_paths),
    "external_inventory_overlap_count": len(manifest_paths & external_paths),
    "external_archive_overlap_count": len(actual_members & external_paths),
    "success_case_reached_intercepted_phase_invocation": ok,
    "collision_case_rejected_before_phase_invocation": not rejected and not collision_calls,
    "path_escape_and_mount_allowmap_guards": "passed",
}, sort_keys=True))
