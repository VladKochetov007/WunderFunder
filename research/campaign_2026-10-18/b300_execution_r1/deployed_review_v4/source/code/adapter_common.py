"""Shared hashes, source pins and safe JSON/symlink helpers for the tabdata bridge."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_CACHE_INPUTS = (
    "chains.npz",
    "chain_folds.npz",
    "train_targets.npz",
    "valid_targets.npz",
)
_CODE_DIR = Path(__file__).resolve().parent
_RECIPE = _CODE_DIR / "command_builder.json"
_SOURCE_MANIFEST = _CODE_DIR / "a266_source_manifest.json"
_TREE_MANIFEST = _CODE_DIR / "source_package_manifest.json"
THREAD_ENVIRONMENT = {
    "WNN_THREADS": "2",
    "OMP_NUM_THREADS": "2",
    "MKL_NUM_THREADS": "2",
    "OPENBLAS_NUM_THREADS": "2",
    "NUMEXPR_NUM_THREADS": "2",
}
_ADAPTER_BUNDLE_FILES = (
    "adapter_common.py",
    "source_package_verify.py",
    "fit_recipe.py",
    "fit_workspace.py",
    "fit_receipt.py",
    "train_cell.py",
    "export_contract.py",
    "export_cell.py",
    "gpu_entrypoint.py",
    "command_builder.json",
    "a266_source_manifest.json",
    "source_package_manifest.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def verify_source_manifest(
    source_root: Path, manifest: dict[str, Any]
) -> dict[str, Any]:
    """Verify the portable a266 file set without requiring a Git checkout."""
    if manifest.get("schema") != "tabdata-l4-source-manifest.v1":
        raise ValueError("unsupported frozen trainer source manifest")
    expected_files = manifest.get("files")
    if not isinstance(expected_files, dict) or not expected_files:
        raise ValueError("frozen trainer source manifest has no file hashes")
    actual_paths = {
        "scripts/train_rnn.py",
        "src/wnn/training/README.md",
        "pyproject.toml",
        "uv.lock",
    }
    actual_paths.update(
        path.relative_to(source_root).as_posix()
        for path in (source_root / "src/wnn/training").glob("*.py")
    )
    actual_files: dict[str, str] = {}
    for relative in sorted(actual_paths):
        path = source_root / relative
        if not path.is_file():
            if relative in expected_files:
                raise ValueError(f"frozen trainer source file is missing: {relative}")
            continue
        actual_files[relative] = _sha256(path)
    if actual_files != expected_files:
        missing = sorted(set(expected_files) - set(actual_files))
        extra = sorted(set(actual_files) - set(expected_files))
        changed = sorted(
            path
            for path in set(actual_files) & set(expected_files)
            if actual_files[path] != expected_files[path]
        )
        raise ValueError(
            f"trainer source differs from frozen manifest (missing={missing}, "
            f"extra={extra}, changed={changed})"
        )
    encoded = json.dumps(actual_files, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    identity = hashlib.sha256(encoded).hexdigest()
    if identity != manifest.get("source_identity_sha256"):
        raise ValueError(
            "trainer source aggregate identity does not match its file manifest"
        )
    return {
        "schema": manifest["schema"],
        "source_revision_expected": manifest.get("source_revision"),
        "source_identity_sha256": identity,
        "file_count": len(actual_files),
        "files": actual_files,
    }


def apply_thread_environment() -> dict[str, str]:
    """Bound CPU-side PyTorch and math-library pools before importing Torch."""
    os.environ.update(THREAD_ENVIRONMENT)
    return dict(THREAD_ENVIRONMENT)


def verify_deployed_tree_manifest(
    source_root: Path, manifest: dict[str, Any], *, manifest_sha256: str
) -> dict[str, Any]:
    """Verify the declared normalized subset; retain the upstream archive lineage."""
    from source_package_verify import verify_source_package

    return verify_source_package(source_root, manifest, manifest_sha256=manifest_sha256)


def _link(link: Path, target: Path, *, directory: bool) -> None:
    target = target.resolve(strict=True)
    if link.is_symlink():
        if link.resolve(strict=True) != target:
            raise ValueError(f"workspace input link points elsewhere: {link}")
        return
    if link.exists():
        raise ValueError(f"refusing to replace a pre-existing workspace path: {link}")
    link.symlink_to(target, target_is_directory=directory)


def adapter_bundle_identity() -> dict[str, Any]:
    """Hash the complete producer/export adapter set and its frozen input pins."""
    files = {name: _sha256(_CODE_DIR / name) for name in sorted(_ADAPTER_BUNDLE_FILES)}
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    recipe = _json(_RECIPE)
    tree_manifest = _json(_TREE_MANIFEST)
    tree_manifest_sha256 = _sha256(_TREE_MANIFEST)
    if recipe.get("source_tree_manifest_sha256") != tree_manifest_sha256 or recipe.get(
        "expected_source_tree_identity_sha256"
    ) != tree_manifest.get("deployed_tree_identity_sha256"):
        raise ValueError(
            "adapter bundle and deployed source-tree manifest pins disagree"
        )
    return {
        "schema": "tabdata-b300-adapter-bundle.v1",
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "files": files,
        "source_tree_manifest_sha256": tree_manifest_sha256,
        "source_tree_identity_sha256": tree_manifest["deployed_tree_identity_sha256"],
        "source_tree_archive_sha256": tree_manifest["archive_sha256"],
    }
