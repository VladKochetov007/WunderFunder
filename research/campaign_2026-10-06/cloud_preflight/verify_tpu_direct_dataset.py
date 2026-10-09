from __future__ import annotations

import argparse
import hashlib
import json
import stat
from pathlib import Path, PurePosixPath

EXPECTED_DATASET = "vladkochetov07/wnn-jax-realdata-e463-tpu-20261004-v1"
EXPECTED_MANIFEST_SHA256 = "851898475dfa701f2089ec76cfe9592538aa2e7f9b70ebcb68c0a614e332d6fb"
EXPECTED_ARCHIVE_SHA256 = "a29bdf21a910a6d7b403d9ed8c52cf0761618f5ef6576b5f06d70d0a8fd5ec9e"


def sha256_file(path: Path) -> str:
    """Hash a file in bounded chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(dataset_root: Path, provider_listing_path: Path | None) -> dict[str, object]:
    """Verify the uploaded direct source tree against its frozen manifest."""
    dataset_root = dataset_root.resolve(strict=True)
    manifest_path = dataset_root / "source-manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise FileNotFoundError(f"source manifest missing or not a regular file: {manifest_path}")
    manifest_sha256 = sha256_file(manifest_path)
    if manifest_sha256 != EXPECTED_MANIFEST_SHA256:
        raise ValueError(f"source manifest hash changed: {manifest_sha256}")
    manifest_value: object = json.loads(manifest_path.read_text())
    if not isinstance(manifest_value, dict):
        raise ValueError("source manifest root is not an object")
    manifest = manifest_value
    if manifest.get("dataset") != EXPECTED_DATASET:
        raise ValueError(f"source dataset identity changed: {manifest.get('dataset')!r}")
    if (
        manifest.get("private"),
        manifest.get("internet"),
        manifest.get("gpu"),
        manifest.get("tpu"),
    ) != (
        True,
        False,
        False,
        True,
    ):
        raise ValueError("source manifest privacy or resource contract changed")
    archive_sha256 = manifest.get("archive_sha256")
    if archive_sha256 != EXPECTED_ARCHIVE_SHA256:
        raise ValueError(f"declared source archive identity changed: {archive_sha256!r}")
    expected_value = manifest.get("files")
    if not isinstance(expected_value, dict) or not expected_value:
        raise ValueError("source manifest has no file inventory")

    bundle_root = dataset_root / "source_bundle"
    if not bundle_root.is_dir() or bundle_root.is_symlink():
        raise FileNotFoundError(f"direct source bundle directory absent: {bundle_root}")
    observed_paths: set[str] = set()
    files: dict[str, dict[str, object]] = {}
    for relative_name, identity in sorted(expected_value.items()):
        if not isinstance(relative_name, str) or not isinstance(identity, dict):
            raise ValueError("source manifest file inventory has an invalid entry")
        relative = PurePosixPath(relative_name)
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise ValueError(f"unsafe source manifest path: {relative_name!r}")
        path = bundle_root.joinpath(*relative.parts)
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"manifest-listed regular source file absent: {relative_name}")
        digest = sha256_file(path)
        size = path.stat().st_size
        expected_size = identity.get("size_bytes")
        expected_sha256 = identity.get("sha256")
        if size != expected_size or digest != expected_sha256:
            raise ValueError(
                f"source file identity changed for {relative_name}: size={size}, sha256={digest}"
            )
        observed_paths.add(relative.as_posix())
        files[relative.as_posix()] = {"size_bytes": size, "sha256": digest}

    actual_paths: set[str] = set()
    for path in bundle_root.rglob("*"):
        if path.is_symlink():
            raise ValueError(
                f"unexpected symlink in direct source bundle: {path.relative_to(bundle_root)}"
            )
        if path.is_file():
            if not stat.S_ISREG(path.stat().st_mode):
                raise ValueError(
                    f"non-regular source bundle member: {path.relative_to(bundle_root)}"
                )
            actual_paths.add(path.relative_to(bundle_root).as_posix())
    if actual_paths != observed_paths:
        raise ValueError(
            "direct source tree differs from manifest: "
            f"missing={sorted(observed_paths - actual_paths)}, "
            f"unlisted={sorted(actual_paths - observed_paths)}"
        )
    result: dict[str, object] = {
        "status": "passed",
        "dataset": EXPECTED_DATASET,
        "dataset_root": str(dataset_root),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "declared_source_archive_sha256": archive_sha256,
        "archive_file_present": (dataset_root / "source_bundle.zip").is_file(),
        "direct_bundle_path": str(bundle_root),
        "manifest_file_count": len(expected_value),
        "direct_tree_file_count": len(actual_paths),
        "all_direct_bundle_files_match_manifest": True,
        "files": files,
    }
    if provider_listing_path is not None:
        listing_bytes = provider_listing_path.read_bytes()
        listing_value: object = json.loads(listing_bytes)
        if not isinstance(listing_value, list):
            raise ValueError("Kaggle dataset file listing is not a JSON array")
        provider_files: dict[str, int] = {}
        for entry in listing_value:
            if not isinstance(entry, dict):
                raise ValueError("Kaggle dataset file listing has an invalid row")
            name = entry.get("name")
            size = entry.get("size")
            if not isinstance(name, str) or type(size) is not int or name in provider_files:
                raise ValueError("Kaggle dataset file listing has an invalid or duplicate identity")
            provider_files[name] = size
        local_files = {
            path.relative_to(dataset_root).as_posix(): path.stat().st_size
            for path in dataset_root.rglob("*")
            if path.is_file()
        }
        if local_files != provider_files:
            raise ValueError(
                "downloaded direct dataset differs from Kaggle listing: "
                f"missing={sorted(set(provider_files) - set(local_files))}, "
                f"unlisted={sorted(set(local_files) - set(provider_files))}, "
                f"size_mismatch={sorted(name for name in set(local_files) & set(provider_files) if local_files[name] != provider_files[name])}"
            )
        result["provider_listing_path"] = str(provider_listing_path.resolve())
        result["provider_listing_sha256"] = hashlib.sha256(listing_bytes).hexdigest()
        result["provider_file_count"] = len(provider_files)
        result["provider_listing_matches_download"] = True
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--provider-listing", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = verify(arguments.dataset_root, arguments.provider_listing)
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if arguments.output:
        arguments.output.write_text(encoded)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
