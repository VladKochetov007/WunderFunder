"""Record the isolated Python 3.11 wheel environment without network access."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import numpy
import onnxruntime
import pyarrow


def main() -> None:
    """Bind dependency bytes and demonstrate imports in the target interpreter."""
    directory = Path(__file__).resolve().parent
    site = directory / "python311-site"
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("verification requires Python 3.11")
    modules = [numpy, onnxruntime, pyarrow]
    if any(not Path(module.__file__).resolve().is_relative_to(site) for module in modules):
        raise RuntimeError("import escaped the isolated dependency directory")
    files = {
        str(path.relative_to(site)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(site.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    }
    distributions = sorted(
        (item.metadata["Name"], item.version)
        for item in importlib.metadata.distributions(path=[str(site)])
    )
    result = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "distributions": distributions,
        "file_sha256": files,
        "imports": {module.__name__: module.__version__ for module in modules},
        "numpy_cpu_probe": numpy.array([1.0, 2.0]).sum().item(),
        "scope": "local isolated dependencies; no submission or portal runtime claim",
    }
    with (directory / "python311_dependencies_v1.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"imports": result["imports"], "hashed_files": len(files)}))


if __name__ == "__main__":
    main()
