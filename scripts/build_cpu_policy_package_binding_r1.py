"""Verify four extracted xteam_h4 policy packages and write a static binding."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

SOURCE_ARCHIVE_SHA256 = "f1635d14c0e1b47848b5caeb2ec4ad9362a85800f1b2e2df80702127d3867b27"
LABELS = ("A", "B", "C", "D")
POLICIES = {
    "A": "q8vnni",
    "B": "q8vnni256",
    "C": "q8vnni_blocked",
    "D": "q8vnni_fused",
}
FIXED_FILES = ("members.json", "weights_0.npz", "weights_1.npz", "weights_2.npz", "weights_3.npz")


def sha256_file(path: Path) -> str:
    """Hash a file in bounded memory."""
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def file_hashes(package: Path) -> dict[str, str]:
    """Return package-relative file hashes, excluding interpreter cache files."""
    package = package.resolve(strict=True)
    return {
        path.relative_to(package).as_posix(): sha256_file(path)
        for path in sorted(package.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    }


def verify_archive(
    archive_path: Path, expected_files: dict[str, str], expected_sha256: str, expected_bytes: int
) -> None:
    """Check exact archive bytes, unique safe members, CRCs, and each payload hash."""
    if (
        archive_path.stat().st_size != expected_bytes
        or sha256_file(archive_path) != expected_sha256
    ):
        raise ValueError(f"package archive bytes/hash mismatch: {archive_path}")
    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise ValueError(
                f"package archive contains duplicate/overlapping member names: {archive_path}"
            )
        if (bad_member := archive.testzip()) is not None:
            raise ValueError(f"package archive CRC failure at {bad_member}: {archive_path}")
        file_infos = [info for info in infos if not info.is_dir()]
        for info in file_infos:
            name = PurePosixPath(info.filename)
            if name.is_absolute() or ".." in name.parts or "\\" in info.filename:
                raise ValueError(f"unsafe archive member path {info.filename!r}")
        if {info.filename for info in file_infos} != set(expected_files):
            raise ValueError(f"archive member inventory differs from the manifest: {archive_path}")
        for info in file_infos:
            if hashlib.sha256(archive.read(info)).hexdigest() != expected_files[info.filename]:
                raise ValueError(f"archive member hash mismatch: {archive_path}:{info.filename}")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--package-manifest", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def build_binding(args: argparse.Namespace) -> dict[str, Any]:
    """Bind all package payload hashes to the original archive and fixed weights."""
    inventory = json.loads(args.inventory.read_text())
    manifest = json.loads(args.package_manifest.read_text())
    original = inventory["packages"]["xteam_h4"]
    if (
        original["sha256"] != SOURCE_ARCHIVE_SHA256
        or manifest["source_archive_sha256"] != SOURCE_ARCHIVE_SHA256
    ):
        raise ValueError("package sources do not bind to the frozen original xteam_h4 archive")
    original_submission = next(
        entry for entry in manifest["submission_order"] if entry["label"] == "A"
    )
    archive_by_label = {"A": original_submission}
    archive_by_label.update(
        {entry["label"]: entry for entry in manifest["variants"] if entry["label"] in LABELS[1:]}
    )
    original_files = {name: spec["sha256"] for name, spec in original["files"].items()}
    repository_root = args.package_manifest.resolve().parents[4]
    package_entries: dict[str, dict[str, Any]] = {}
    fixed_files: dict[str, str] | None = None
    for label in LABELS:
        archive_entry = archive_by_label[label]
        if archive_entry["policy"] != POLICIES[label]:
            raise ValueError(f"campaign manifest policy mismatch for {label}")
        expected_files = dict(original_files)
        for relative, spec in archive_entry.get("changed_files", {}).items():
            expected_files[relative] = spec["package_sha256"]
        package = args.package_root / label
        archive_path = Path(archive_entry["archive_path"])
        if not archive_path.is_absolute():
            archive_path = repository_root / archive_path
        verify_archive(
            archive_path,
            expected_files,
            archive_entry["archive_sha256"],
            archive_entry["archive_bytes"],
        )
        actual_files = file_hashes(package)
        if actual_files != expected_files:
            absent = sorted(set(expected_files) - set(actual_files))
            extra = sorted(set(actual_files) - set(expected_files))
            wrong = sorted(
                name
                for name in set(expected_files) & set(actual_files)
                if expected_files[name] != actual_files[name]
            )
            raise ValueError(
                f"package {label} differs from bound manifest; absent={absent}, extra={extra}, wrong={wrong}"
            )
        package_fixed = {name: actual_files[name] for name in FIXED_FILES}
        if fixed_files is None:
            fixed_files = package_fixed
        elif package_fixed != fixed_files:
            raise ValueError(f"package {label} changed fixed ensemble members or weight files")
        package_entries[label] = {
            "directory": label,
            "policy": POLICIES[label],
            "archive_path": archive_entry["archive_path"],
            "archive_sha256": archive_entry["archive_sha256"],
            "archive_bytes": archive_entry["archive_bytes"],
            "files": actual_files,
        }
    return {
        "format": "wnn.cpu-policy-package-binding",
        "version": 1,
        "source_archive_path": original["archive"],
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "native_source_base_revision": manifest["native_source_base_revision"],
        "native_source_files_sha256": manifest["native_source_files_sha256"],
        "package_archive_sha256": {
            label: package_entries[label]["archive_sha256"] for label in LABELS
        },
        "fixed_weight_and_member_sha256": fixed_files,
        "packages": package_entries,
        "source_manifest_sha256": sha256_file(args.package_manifest),
        "source_inventory_sha256": sha256_file(args.inventory),
        "native_policy_scope": (
            "Weights, members/order/coefficients, preprocessing, architecture and Q8 codes/scales "
            "are hash-fixed; only precision assertion and package/native scheduling implementation vary."
        ),
    }


def main() -> None:
    """Validate the package root and write a new immutable JSON binding."""
    args = _args()
    if args.output.exists():
        raise FileExistsError(args.output)
    binding = build_binding(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(binding, indent=2, sort_keys=True) + "\n"
    with args.output.open("x", encoding="utf-8") as sink:
        sink.write(payload)
    print(json.dumps({"output": str(args.output), "sha256": sha256_file(args.output)}))


if __name__ == "__main__":
    main()
