from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


def digest(path: Path) -> str:
    """Hash one artifact without loading it into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    """Import the delivered bundle without the repository on Python's path."""
    if sys.version_info[:2] != (3, 11) or not sys.flags.isolated:
        raise RuntimeError("requires Python3.11 -I")
    archive_path, manifest_path, temporary_root, output = map(Path, sys.argv[1:])
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads(manifest_path.read_text())
    archive_hash = digest(archive_path)
    if archive_hash != manifest["artifact"]["sha256"]:
        raise ValueError("artifact hash differs from the build receipt")
    temporary_root = temporary_root.resolve(strict=True)
    import_path = list(sys.path)
    os.environ["WNN_NATIVE_LEVEL"] = "v3"
    os.environ.pop("WNN_PRECISION", None)
    with tempfile.TemporaryDirectory(dir=temporary_root) as directory:
        package = Path(directory)
        with zipfile.ZipFile(archive_path) as archive:
            if sum(item.file_size for item in archive.infolist()) > 64_000_000:
                raise ValueError("unexpected expanded package size")
            for item in archive.infolist():
                name = PurePosixPath(item.filename)
                if name.is_absolute() or ".." in name.parts or item.is_dir():
                    raise ValueError("unexpected package path")
                content = archive.read(item)
                if (
                    hashlib.sha256(content).hexdigest()
                    != manifest["artifact"]["members"][str(name)]
                ):
                    raise ValueError("entry hash differs from the build receipt")
                destination = package / str(name)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(content)
        spec = importlib.util.spec_from_file_location("checked_solution", package / "solution.py")
        if spec is None or spec.loader is None:
            raise ImportError("missing solution entry point")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = module.PredictionModel().model
        if model.active_members != 4 or list(model.coefficients) != [0.25] * 4:
            raise ValueError("loaded member count or coefficients differ from recipe")
        if model.precision != "q8vnni":
            raise ValueError("unexpected runtime precision")
        native = Path(module.wnn_native.__file__).resolve()
        if native.parent != package / "v3":
            raise ValueError("native import resolved outside the extracted package")
        report = {
            "passed": True,
            "zip_sha256": archive_hash,
            "python": sys.version,
            "isolated_mode": bool(sys.flags.isolated),
            "initial_sys_path": import_path,
            "native_sha256": digest(native),
            "active_members": model.active_members,
            "coefficients": list(model.coefficients),
            "precision": model.precision,
            "checker_sha256": digest(Path(__file__)),
            "scope": "isolated dependency import and constructor; no install, network request or score",
        }
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
