from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


def verify_source_capsule(root: Path, manifest: Path, expected_sha256: str, commit: str) -> str:
    """Verify a trusted Git-blob inventory without requiring Git on the receiver.

    The manifest digest must come from the sending operator, independently of the
    received capsule. Every inventoried file is checked, and additional importable
    Python under src/scripts is rejected rather than silently imported.
    """
    raw = manifest.read_bytes()
    if len(raw) > 8_000_000 or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("source capsule manifest digest or size mismatch")
    value = json.loads(raw)
    if (
        not isinstance(value, dict)
        or set(value) != {"schema", "commit", "files"}
        or value["schema"] != "wnn.git-source-capsule.v1"
        or value["commit"] != commit
    ):
        raise ValueError("source capsule is not the pinned Git source inventory")
    files = value["files"]
    if not isinstance(files, dict) or not files:
        raise ValueError("source capsule has no file inventory")
    root = root.resolve(strict=True)
    for relative, digest in files.items():
        if not isinstance(relative, str) or not isinstance(digest, str):
            raise ValueError("source capsule file entry is malformed")
        parts = PurePosixPath(relative)
        if parts.is_absolute() or ".." in parts.parts or "\\" in relative:
            raise ValueError("source capsule file path is unsafe")
        path = root / relative
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
            raise ValueError(f"source capsule file is missing or unsafe: {relative}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"source capsule file changed: {relative}")
    python_files = {
        path.relative_to(root).as_posix()
        for directory in (root / "src", root / "scripts")
        for path in directory.rglob("*.py")
    }
    expected_python = {
        name for name in files if name.endswith(".py") and name.startswith(("src/", "scripts/"))
    }
    if python_files != expected_python:
        raise ValueError("source capsule has additional or missing importable Python files")
    if "scripts/predict_run.py" not in files or not any(n.startswith("src/") for n in files):
        raise ValueError("source capsule omits its predictor or implementation")
    return commit


def main() -> None:
    """Check one operator-bound immutable source capsule before invoking receiver stages."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    verify_source_capsule(args.root, args.manifest, args.manifest_sha256, args.commit)


if __name__ == "__main__":
    main()
