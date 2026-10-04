"""Small stdlib-only adversarial checks for the v5 external-mount guard."""

from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path


KERNEL = Path(__file__).parent / "kernel_push" / "kernel.py"
spec = importlib.util.spec_from_file_location("w384_kernel_v5_fixture", KERNEL)
if spec is None or spec.loader is None:
    raise RuntimeError("Could not load the v5 kernel source")
kernel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kernel)

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

print(json.dumps({"status": "passed", "checks": len(passed), "names": passed}, sort_keys=True))
