"""Small stdlib-only adversarial checks for the v5 external-mount guard."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import re
import tempfile
import tarfile
from pathlib import Path


KERNEL = Path(__file__).parent / "bundle" / "kernel" / "kernel.py"
spec = importlib.util.spec_from_file_location("w384_kernel_v5_fixture", KERNEL)
if spec is None or spec.loader is None:
    raise RuntimeError("Could not load the v5 kernel source")
kernel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kernel)

repo_root = Path(__file__).resolve().parents[3]
v5_kernel_path = repo_root / "research/campaign_2026-10-04/kaggle/W384_fold1_dualT4_v5_transport_review/kernel_push/kernel.py"
v5_kernel_text = v5_kernel_path.read_text()
assert hashlib.sha256(v5_kernel_path.read_bytes()).hexdigest() == "76e2ed1c1c67d8bcd9992cd1ee3026750da228604487b6e3a4dc6e68d2736cbd"
v6_kernel_text = KERNEL.read_text()
for assignment in ("BUNDLE_SLUG", "EXPECTED_ARCHIVE_SHA256", "EXPECTED_EXTERNAL_MOUNTS"):
    old = re.search(rf"^{assignment} = .*?$", v5_kernel_text, re.MULTILINE)
    new = re.search(rf"^{assignment} = .*?$", v6_kernel_text, re.MULTILINE)
    assert old and new
    if assignment != "EXPECTED_EXTERNAL_MOUNTS":
        assert old.group(0) != new.group(0)
    v6_kernel_text = v6_kernel_text.replace(new.group(0), old.group(0), 1)
assert v6_kernel_text == v5_kernel_text
assert "timeout_seconds=25_180" in KERNEL.read_text()

mounts = kernel.EXPECTED_EXTERNAL_MOUNTS
archive_members: set[str] = set()
passed: list[str] = []


def expect_reject(name: str, call) -> None:
    try:
        call()
    except (ValueError, FileNotFoundError):
        passed.append(name)
    else:
        raise AssertionError(f"Guard accepted invalid case: {name}")


with tempfile.TemporaryDirectory(prefix="w384-mount-guard-") as temp:
    base = Path(temp)
    target_root = base / "experiment"
    target_root.mkdir()
    input_root = base / "input"

    kernel.validate_external_mounts(target_root, mounts, archive_members, mounts)
    resolved = {}
    for logical, descriptor in mounts.items():
        if isinstance(descriptor, dict):
            dataset = descriptor["dataset"]
            filename = descriptor["filename"]
        else:
            dataset = "vladkochetov07/wunder-funder"
            filename = descriptor
        owner, slug = dataset.split("/")
        asset = input_root / owner / slug / filename
        asset.parent.mkdir(parents=True, exist_ok=True)
        asset.write_bytes(b"fixture")
        resolved[logical] = kernel.find_external_candidate(input_root, dataset, filename)
    assert len(resolved) == 26
    assert all(path.is_file() for path in resolved.values())
    passed.append("all_26_declared_mounts_and_files_accepted")

    expect_reject(
        "missing_external_mount_descriptor",
        lambda: kernel.validate_external_mounts(
            target_root, dict(list(mounts.items())[1:]), archive_members, mounts
        ),
    )
    extra = dict(mounts)
    extra["extra/undeclared.bin"] = {"dataset": "owner/dataset", "filename": "x"}
    expect_reject(
        "undeclared_external_mount_descriptor",
        lambda: kernel.validate_external_mounts(target_root, extra, archive_members, mounts),
    )

    traversal = {"../escape": {"dataset": "owner/dataset", "filename": "x"}}
    expect_reject(
        "traversal_path_even_if_misdeclared",
        lambda: kernel.validate_external_mounts(
            target_root, traversal, set(), traversal
        ),
    )

    overlap_path = next(iter(mounts))
    expect_reject(
        "external_path_overlaps_archive_member",
        lambda: kernel.validate_external_mounts(
            target_root, mounts, {overlap_path}, mounts
        ),
    )

    occupied_root = base / "occupied"
    occupied_root.mkdir()
    occupied_target = occupied_root / next(iter(mounts))
    occupied_target.parent.mkdir(parents=True)
    occupied_target.write_text("preserve")
    expect_reject(
        "existing_file_target_preserved",
        lambda: kernel.validate_external_mounts(
            occupied_root, mounts, archive_members, mounts
        ),
    )

    symlink_root = base / "symlinked"
    symlink_root.mkdir()
    symlink_target = symlink_root / next(iter(mounts))
    symlink_target.parent.mkdir(parents=True)
    symlink_target.symlink_to(base / "missing-target")
    expect_reject(
        "broken_symlink_target_preserved",
        lambda: kernel.validate_external_mounts(
            symlink_root, mounts, archive_members, mounts
        ),
    )

    first_logical, first_descriptor = next(iter(mounts.items()))
    if isinstance(first_descriptor, dict):
        dataset = first_descriptor["dataset"]
        filename = first_descriptor["filename"]
    else:
        dataset = "vladkochetov07/wunder-funder"
        filename = first_descriptor
    owner, slug = dataset.split("/")
    missing_root = base / "missing-input"
    missing_root.mkdir()
    expect_reject(
        "missing_provider_external_file",
        lambda: kernel.find_external_candidate(missing_root, dataset, filename),
    )


    # The kernel cleanup removes only known external symlinks; normal files and
    # unrelated symlinks are left intact for Kaggle's output collector.
    cleanup_root = base / "cleanup"
    cleanup_root.mkdir()
    listed_link, listed_file = list(mounts)[0], list(mounts)[2]
    source = base / "mounted-source"
    source.write_bytes(b"mounted")
    (cleanup_root / listed_link).parent.mkdir(parents=True)
    (cleanup_root / listed_link).symlink_to(source)
    outside = cleanup_root / "unlisted-link"
    outside.symlink_to(source)
    (cleanup_root / listed_file).parent.mkdir(parents=True)
    (cleanup_root / listed_file).write_bytes(b"preserve")
    kernel.release_mounts(cleanup_root, {listed_link: mounts[listed_link]})
    assert not (cleanup_root / listed_link).exists() and not (cleanup_root / listed_link).is_symlink()
    assert outside.is_symlink() and (cleanup_root / listed_file).read_bytes() == b"preserve"
    passed.append("cleanup_removes_only_requested_mount_symlinks")

    # Check the real staged allow-set against the generated archive; no external
    # mounts may overlap members, and each member must occur once.
    bundle = KERNEL.parent.parent
    manifest = json.loads((bundle / "bundle-manifest.json").read_text())
    assert kernel.BUNDLE_SLUG == manifest["dataset"].split("/", 1)[1]
    assert kernel.EXPECTED_ARCHIVE_SHA256 == manifest["archive_sha256"]
    assert kernel.EXPECTED_EXTERNAL_MOUNTS == manifest["external_mounts"]
    archive = bundle / "bundle.wnn"
    allowed = {item["path"] for item in manifest["files"]} | set(manifest["spec_paths"])
    expected_members = allowed - set(manifest["external_mounts"])
    with tarfile.open(archive, "r:gz") as tar:
        infos = tar.getmembers()
    names = [item.name for item in infos]
    assert len(names) == 265 and len(names) == len(set(names)) and set(names) == expected_members
    assert all(item.isfile() and not item.name.startswith("/") and ".." not in Path(item.name).parts and "\\" not in item.name for item in infos)
    passed.append("archive_exact_allow_set_unique_safe_members")

print(json.dumps({"status": "passed", "checks": len(passed), "names": passed}, sort_keys=True))
