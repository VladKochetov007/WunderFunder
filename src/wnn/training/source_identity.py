from __future__ import annotations

import hashlib
import json
from importlib.metadata import version

from wnn.splits import file_sha256
from wnn.training.environment import TrainingEnvironment


def training_source_identity(environment: TrainingEnvironment) -> dict[str, object]:
    """Bind the thin CLI and every implementation module to one source digest.

    Args:
        environment: Actual source root and public entrypoint for this run.

    Returns:
        A JSON-safe manifest of file hashes, aggregate digest and explicit metadata gaps.
        Source-only packets may omit dependency files; required implementation files
        and the public entrypoint are still hashed. The actual attrs version is recorded.
    """
    root = environment.source_root
    paths = [environment.entrypoint]
    dependency_paths = [root / "pyproject.toml", root / "uv.lock"]
    missing_dependencies = [path.name for path in dependency_paths if not path.is_file()]
    paths.extend(path for path in dependency_paths if path.is_file())
    paths.extend(sorted((root / "src/wnn/training").glob("*.py")))
    paths.append(root / "src/wnn/training_dynamics.py")
    files = {path.relative_to(root).as_posix(): file_sha256(path) for path in paths}
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "schema": "wnn.training-source.v1",
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "files": files,
        "missing_dependency_files": missing_dependencies,
        "runtime_dependencies": {"attrs": version("attrs")},
    }
