from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _write_json(path: Path, value: dict[str, object]) -> None:
    """Persist the standalone cache-gate result atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _required_regular_file(path: Path, root: Path, label: str) -> None:
    """Require a regular file whose resolved path remains under the reviewed root."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} is absent, linked, or not a regular file: {path}")
    if not path.resolve(strict=True).is_relative_to(root):
        raise ValueError(f"{label} escapes its reviewed root: {path}")


def validate_mounted_cache(
    project_root: Path,
    cache_relative: str,
    variant: str,
    require_families: bool,
) -> dict[str, object]:
    """Validate the exact mounted cache and split before any training phase."""
    project = project_root.resolve(strict=True)
    cache_path = Path(cache_relative)
    if cache_path.is_absolute() or ".." in cache_path.parts:
        raise ValueError("cache root must be project-relative")
    declared_cache_root = project / cache_path
    if declared_cache_root.is_symlink() or not declared_cache_root.is_dir():
        raise ValueError("cache root is absent, linked, or not a directory")
    cache_root = declared_cache_root.resolve(strict=True)
    if not cache_root.is_relative_to(project):
        raise ValueError("cache root escapes the project")

    metadata_path = cache_root / "metadata.json"
    _required_regular_file(metadata_path, cache_root, "cache metadata")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    teacher_split = metadata.get("teacher_source_split")
    if not isinstance(teacher_split, dict):
        raise ValueError("cache metadata lacks its teacher source split")
    split_relative = Path(str(teacher_split.get("path", "")))
    if split_relative.is_absolute() or ".." in split_relative.parts:
        raise ValueError("teacher source split must be project-relative")
    split_path = project / split_relative
    _required_regular_file(split_path, project, "teacher source split")

    required_keys = [f"main_{variant}_train", f"main_{variant}_valid"]
    if require_families:
        required_keys.extend(("families_train", "families_valid"))
    files = metadata.get("files")
    if not isinstance(files, dict):
        raise ValueError("cache metadata lacks its file records")
    regular_arrays: dict[str, dict[str, object]] = {}
    for key in required_keys:
        entry = files.get(key)
        if not isinstance(entry, dict):
            raise ValueError(f"cache metadata lacks required array {key}")
        filename = Path(str(entry.get("path", "")))
        if filename.is_absolute() or len(filename.parts) != 1:
            raise ValueError(f"cache array path is not a single filename: {key}")
        array_path = cache_root / filename
        _required_regular_file(array_path, cache_root, f"cache array {key}")
        regular_arrays[key] = {
            "path": str(filename),
            "bytes": array_path.stat().st_size,
            "sha256": entry.get("sha256"),
        }

    sys.path.insert(0, str(project / "src"))
    try:
        from wnn.all_data_distillation_cache import validate_all_data_distillation_cache

        validated = validate_all_data_distillation_cache(
            cache_root,
            project,
            variant,
            require_families=require_families,
        )
    finally:
        sys.path.remove(str(project / "src"))
    return {
        "status": "passed",
        "cache_root": str(cache_path),
        "variant": variant,
        "require_families": require_families,
        "metadata_sha256": validated["metadata_sha256"],
        "verified_files": validated["verified_files"],
        "regular_arrays_under_cache_root": regular_arrays,
        "teacher_source_split": {
            "path": str(split_relative),
            "sha256": teacher_split.get("sha256"),
            "size_bytes": split_path.stat().st_size,
        },
    }


def main() -> int:
    """Run and persist the mounted-cache preflight without entering fit code."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--variant", choices=("source", "normalized", "homogeneous"), required=True)
    parser.add_argument("--require-families", action="store_true")
    parser.add_argument("--receipt-path", required=True)
    args = parser.parse_args()
    project = args.project_root.resolve(strict=True)
    receipt_relative = Path(args.receipt_path)
    if receipt_relative.is_absolute() or ".." in receipt_relative.parts:
        raise ValueError("preflight receipt path must be project-relative")
    receipt_path = project / receipt_relative
    started = time.monotonic()
    try:
        result = validate_mounted_cache(
            project,
            args.cache_root,
            args.variant,
            args.require_families,
        )
        result["elapsed_seconds"] = time.monotonic() - started
        _write_json(receipt_path, result)
        return 0
    except Exception as error:
        _write_json(
            receipt_path,
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "elapsed_seconds": time.monotonic() - started,
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
